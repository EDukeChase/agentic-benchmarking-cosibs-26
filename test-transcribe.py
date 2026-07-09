# pip install --user azure-identity openai
# https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/AAD_Integration/aad_integration_example_sdk.ipynb
# https://github.com/openai/openai-python/blob/main/examples/azure_ad.py

# must successfully complete "az login" CLI command before trying this program

# transcription example from https://learn.microsoft.com/en-us/azure/ai-services/openai/whisper-quickstart

import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

deployment_name = "whisper"

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://bpsmar-ai-openai-1.openai.azure.com",
    azure_ad_token_provider=token_provider,
)

audio_test_file = "sample.mp3"
result = client.audio.transcriptions.create(
    file=open(audio_test_file, "rb"), model=deployment_name
)

print(result)
