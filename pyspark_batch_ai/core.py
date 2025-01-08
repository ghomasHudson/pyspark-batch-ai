'''Library to process a dataframe in batches using openai's batch api'''

import pyspark
from pyspark.sql import types as T
from pyspark.sql import functions as F
import pandas as pd
from loguru import logger
import json
import json_repair
import uuid
import tempfile
import os
import time
import datetime
from openai import AzureOpenAI
from pyspark.sql import SparkSession
import sys
from typing import Union
import openai


# Initialize the logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def process_dataframe(
    df: Union[pd.DataFrame, pyspark.sql.DataFrame],
    client: openai.Client,
    response_schema: pyspark.sql.types.DataType = T.StringType(),
    prompt_is_json: bool = False,
    output_format: str = "auto",
    model: str = "gpt-3.5-turbo-0125"
) -> pyspark.sql.DataFrame:
    """
    Processes a dataframe of prompts in batches using OpenAI's batch API, submitting tasks for LLM processing
    and returning the results.

    This function handles dataframe-to-batch submission by converting each row's 'prompt' field into a request task.
    It determines the appropriate output format based on the content of the first prompt (auto-detection) and submits
    the tasks in parallel batches to OpenAI's API. The results are processed and joined back to the original dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame or pyspark.sql.DataFrame
        The input dataframe containing prompts. It must include a column named 'prompt' with the string or JSON prompts
        to be submitted.

    client : openai.Client
        The OpenAI client used to submit and retrieve tasks from the batch API.

    response_schema : pyspark.sql.types.DataType, optional
        The schema of the response column. Defaults to StringType(). This defines how the API response will be
        structured in the resulting dataframe. If none is specified, the response column will be a string.

    prompt_is_json : bool, optional
        If True, the 'prompt' column contains JSON-formatted prompts. The function will parse these prompts
        before submission. Defaults to False (simple string prompts)

    output_format : str, optional
        The output format of the responses as specified in the prompt. Can be 'auto', 'json', 'dspy', or 'plain'. If set to 'auto', the
        function will infer the format from the first prompt. Defaults to 'auto'.

    model : str, optional
        The model to use for batch processing. Defaults to 'gpt-4o-mini-batch-5B'.

    Returns:
    --------
    pyspark.sql.DataFrame
        A dataframe containing the original input data along with the processed responses from the API.

    Raises:
    -------
    ValueError
        If the input dataframe does not contain a 'prompt' column.

    Notes:
    ------
    - The function internally uses Spark for distributed processing and converts pandas dataframes to Spark dataframes.
    - A 'batch_processing_index' column is temporarily added to manage row tracking during batch submission.
    - The output format is inferred based on the content of the first prompt if 'auto' is selected.
    - After processing, results are joined back to the original dataframe based on the batch index.
    - The function logs the processing duration and completion status.

    Example:
    --------
    >>> import pandas as pd
    >>> data = {'prompt': ['translate this to french: hello', 'summarize this text in one sentence.']}
    >>> df = pd.dataframe(data)
    >>> client = openai.client(api_key="sk-...")
    >>> result_df = process_dataframe(df, client)
    >>> result_df.show()
    """

    spark = SparkSession.builder.appName("pd_to_batch_llm").getOrCreate()
    start = time.time()

    if isinstance(df, pd.DataFrame):
        df = spark.createDataFrame(df)

    if "prompt" not in df.columns:
        raise ValueError("The dataframe must have a 'prompt' column")

    df = df.withColumn("batch_processing_index", F.monotonically_increasing_id())
    random_id = uuid.uuid4().hex.upper()[0:6]

    # Use the first prompt to determine the output format
    if output_format == "auto":
        prompt_1 = df.select("prompt").first().prompt
        if prompt_is_json:
            if isinstance(prompt_1, str):
                prompt_1 = json.loads(prompt_1)
            prompt_1 = " ".join(json.dumps(m["content"]) for m in prompt_1["body"]["messages"])
        prompt_1 = prompt_1.lower()
        if "json" in prompt_1:
            output_format = "json"
        elif "[[ ##" in prompt_1:
            output_format = "dspy"
        else:
            output_format = "plain"
        logger.info(f"Detected output format: {output_format}")

    # Prepare the tasks
    tasks = []
    for row in df.collect():
        if prompt_is_json:
            task_json = json.loads(row.prompt)
            task_json["custom_id"] = f"{random_id}_{row.batch_processing_index}"
            tasks.append(task_json)
        else:
            response_type = "json_object" if output_format == "json" else "text"
            if response_type == "json_object":
                assert "json" in row.prompt, "Prompt must contain the word 'json' to use json_object response format"
            tasks.append({
                "custom_id": f"{random_id}_{row.batch_processing_index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0.1,
                    "response_format": {
                        "type": response_type
                    },
                    "messages": [{
                        "role": "user",
                        "content": row.prompt
                    }]
                }
            })

    # Submit the tasks in batches
    batch_files = _lines_to_batches(tasks)
    result_file_ids = _submit_and_process(client, batch_files).values()

    # Post-process the results
    rows = []
    for file_response_id in result_file_ids:
        content = client.files.content(file_response_id).text
        rows += _process_and_repair(content, output_format=output_format)

    # Join the results back to the original dataframe
    rows = spark.createDataFrame(rows, schema=T.StructType([
            T.StructField("batch_processing_index", T.StringType()),
            T.StructField("response", response_schema)
    ]), verifySchema=False)
    df = df.join(rows, df.batch_processing_index == rows.batch_processing_index)
    df = df.drop("batch_processing_index")

    logger.info(f"Processing complete. Time taken: {time.time() - start:.2f} seconds")
    return df



# #####################
# Helper functions
# #####################

def _process_and_repair(response_str, output_format="json"):
    rows = []
    response_split = response_str.strip().split('\n')
    for i, raw_response in enumerate(response_split):
        try:
            json_response = json_repair.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON on line {i + 1}: {e}")
            continue

        try:
            content = json_response["response"]["body"]["choices"][0]["message"]["content"]
        except:
            continue

        if isinstance(output_format, str):
            match output_format:
                case "plain":
                    content = content
                case "json":
                    try:
                        content = json_repair.loads(content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to repair JSON on line {i + 1}: {e}")
                        continue
                case "dspy":
                    c_dict = {}
                    for c_split in content.split("[[ ##"):
                        title, text = c_split.split("## ]]", 1)
                        c_dict[title] = text
                    if "completed" in c_dict:
                        del c_dict["completed"]
                    content = c_dict
        elif callable(output_format):
            try:
                content = output_format(content)
            except Exception as e:
                logger.error(f"Failed to process output on line {i + 1}: {e}")
        rows.append({
            "batch_processing_index": json_response["custom_id"].split("_")[-1],
            "response": content
        })
    return rows


def _lines_to_batches(task_json_list):
    '''Takes a list of task jsons and batches them as openai batch files'''
    MAX_LINES_PER_FILE = 40000
    MAX_SIZE_BYTES_PER_FILE = 150 * 1024 * 1024

    output_base_path = tempfile.mkdtemp()

    def open_new_file(file_count):
        part_path = f"{output_base_path}/{file_count}.jsonl"
        return open(part_path, 'w'), part_path

    file_count = 1
    line_count = 0
    current_file_size = 0
    output_file = None

    output_file, part_path = open_new_file(file_count)
    logger.debug(f"Writing to {part_path}")

    filepaths = [output_file]
    for item in task_json_list:
        if isinstance(item, dict):
            item = json.dumps(item)
        line = f"{item}\n"
        line_size = len(line.encode('utf-8'))

        if (line_count >= MAX_LINES_PER_FILE) or (current_file_size + line_size > MAX_SIZE_BYTES_PER_FILE):
            output_file.close()
            logger.debug(f"Part {file_count} written to {part_path} (Lines: {line_count}, Size: {current_file_size / (1024 * 1024):.2f} MB)")

            file_count += 1
            line_count = 0
            current_file_size = 0
            output_file, part_path = open_new_file(file_count)
            filepaths.append(output_file)
            logger.debug(f"Writing to {part_path}")

        output_file.write(line)
        line_count += 1
        current_file_size += line_size

    if output_file:
        output_file.close()
        logger.debug(f"Part {file_count} written to {part_path} (Lines: {line_count}, Size: {current_file_size / (1024 * 1024):.2f} MB)")
    return filepaths


def _submit_file(client, file_path):
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="batch")

    logger.debug(f"Uploaded file: {file_path}")
    file_id = file.id

    file_status = client.files.retrieve(file.id)
    while file_status.status == "pending":
        file_status = client.files.retrieve(file.id)

    if file_status.status == "error":
        raise Exception(f"Error uploading file: {file_path}: {file_status.status_details}")

    def submit_batch_job_with_retry(file_id, max_retries=3, delay=5, endpoint="/v1/chat/completions"):
        retries = 0
        while retries < max_retries:
            try:
                batch_response = client.batches.create(input_file_id=file_id, endpoint=endpoint, completion_window="24h")
                logger.debug(f"Batch job submitted successfully for file {file_id}. Batch ID: {batch_response.id}")
                return batch_response
            except Exception as e:
                retries += 1
                logger.error(f"Attempt {retries} to submit batch job for file {file_id} failed: {e}")
                if retries < max_retries:
                    logger.debug(f"Retrying in {delay * (retries+1)} seconds...")
                    time.sleep(delay * (retries+1))
                else:
                    logger.error(f"Max retries reached for file {file_id}. Unable to submit the batch job.")
                    raise

    batch_response = submit_batch_job_with_retry(file_id)
    return batch_response.id


def _submit_and_process(client, batch_files, postprocess_fn=None, max_concurrent_jobs=20):
    split_files = [b.name for b in batch_files]
    total_jobs = len(split_files)

    def monitor_batches(batch_statuses, total_jobs):
        output_file_ids = {}

        while batch_statuses:
            completed_batches = []
            for batch_id, info in batch_statuses.items():
                status = info['status']
                if status not in ("completed", "failed", "cancelled"):
                    batch_response = client.batches.retrieve(batch_id)
                    new_status = batch_response.status
                    if new_status != status:
                        logger.info(f"Batch ID: {batch_id}, Status changed from {status} to {new_status}")
                        batch_statuses[batch_id]['status'] = new_status

                        if new_status == "failed":
                            for error in batch_response.errors.data:
                                logger.error(f"Batch ID {batch_id} - Error code {error.code}: {error.message}")

                        if new_status == "completed":
                            output_file_id = batch_response.output_file_id
                            if not output_file_id:
                                output_file_id = batch_response.error_file_id
                            logger.info(f'Batch ID {batch_id} completed. Output file ID: {output_file_id}')
                            output_file_ids[batch_id] = output_file_id
                            completed_batches.append(batch_id)

                            jobs_left = total_jobs - len(output_file_ids)
                            logger.info(f"Jobs left to process: {jobs_left}")

                else:
                    completed_batches.append(batch_id)

            for batch_id in completed_batches:
                del batch_statuses[batch_id]

            while len(batch_statuses) < max_concurrent_jobs and split_files_queue:
                next_file = split_files_queue.pop(0)
                logger.info(f"Submitting next job for file: {next_file}")
                batch_id = _submit_file(client, next_file)
                batch_statuses[batch_id] = {'status': 'validating', 'file': next_file}
                logger.info(f"Total jobs: {total_jobs}, Currently running: {len(batch_statuses)}, Jobs left in queue: {len(split_files_queue)}")

            if not batch_statuses:
                logger.info("All batch jobs have completed.")
                break

            # Sleep before checking again
            time.sleep(60)  # Adjust the sleep time as needed

        return output_file_ids  # Return the dictionary of output_file_ids

    # Main Logic
    batch_statuses = {}
    split_files_queue = split_files.copy()

    if split_files_queue:
        # Total number of jobs to run
        total_jobs = len(split_files)
        logger.info(f"Total number of jobs to run: {total_jobs}")

        # Submit initial batch of jobs up to MAX_CONCURRENT_JOBS
        for _ in range(min(max_concurrent_jobs, len(split_files_queue))):
            split_file_path = split_files_queue.pop(0)
            logger.debug(f"Submitting split file: {split_file_path}")
            batch_id = _submit_file(client, split_file_path)
            batch_statuses[batch_id] = {'status': 'validating', 'file': split_file_path}
            logger.info(f"Currently running: {len(batch_statuses)}, Jobs left in queue: {len(split_files_queue)}")

        # Monitor batches and submit new jobs as others complete
        output_file_ids = monitor_batches(batch_statuses, total_jobs)
    else:
        logger.info(f"No files to process")

    # At this point, all batch jobs have completed
    logger.info("Processing complete.")
    logger.info(f"Output file IDs: {output_file_ids}")
    return output_file_ids
