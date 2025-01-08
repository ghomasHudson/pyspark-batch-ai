# Pyspark Batch AI
> Batch process Spark DataFrames with LLMs

Start with a table with a `prompt` column:

| Name     | Age | prompt       |
| -------- | --- | ------------ |
| A. Smith | 40  | What is 2+2? |
| B. Jones | 45  | What is 9*4? |

And end up with the same table with a `response` column:

| Name     | Age | response     |
| -------- | --- | ------------ |
| A. Smith | 40  | 4            |
| B. Jones | 45  | 36           |


## Install

`pip install pyspark-batch-ai`

## How to use

```python
import pandas as pd
from pyspark_batch_ai import process_dataframe
data = {'prompt': ['translate this to french: hello', 'summarize this text in one sentence.']}
df = pd.dataframe(data)
client = openai.client(api_key="sk-...")
result_df = process_dataframe(df, client)
result_df.show()
```
