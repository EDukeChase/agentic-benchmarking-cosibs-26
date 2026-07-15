import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

TAVILY_API_KEY = "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"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)