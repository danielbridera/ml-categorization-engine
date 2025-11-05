import os

from dotenv import find_dotenv, load_dotenv

# Find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())


# OpenAI credentials (for batch labeling)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Cloud Platform Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
