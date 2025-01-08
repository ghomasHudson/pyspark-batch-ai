from setuptools import setup, find_packages

setup(
    name='pyspark-batch-ai',
    version='0.1.0',
    description='Batch process Spark DataFrames with OpenAI API',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/ghomasHudson/pyspark-batch-ai',
    author='ghomasHudson',
    author_email='ghomasHudson@github.com',
    packages=find_packages(),
    install_requires=[
        "openai>=1.57.0",
        "json_repair>=0.30.3",
        "pandas>=2.2.3",
        "pyspark>=3.5.3",
        "loguru>=0.5.3"
    ],
)
