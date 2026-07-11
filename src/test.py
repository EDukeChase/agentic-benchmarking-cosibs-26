# pip install --user azure-identity openai
# https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/AAD_Integration/aad_integration_example_sdk.ipynb
# https://github.com/openai/openai-python/blob/main/examples/azure_ad.py

# must successfully complete "az login" CLI command before trying this program

import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

deployment_name = "gpt-5.4-mini"
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://bpsmar-ai-openai-1.openai.azure.com",
    azure_ad_token_provider=token_provider,
)

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "system",
            "content": "Assistant is a large language model who answers technical questions in detail.",
        },
        {
            "role": "user",
            "content": "Are you able to search on the internet alone?",
        },
    ],
)

print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)
