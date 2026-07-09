# pip install --user azure-identity openai

# informed by, see for a more complex example:
# https://medium.com/totalenergies-digital-factory/open-ai-multi-images-prompting-python-api-6148dcd9dcc5

# https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/AAD_Integration/aad_integration_example_sdk.ipynb
# https://github.com/openai/openai-python/blob/main/examples/azure_ad.py

# must successfully complete "az login" CLI command before trying this program

import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import base64

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

deployment_name = "FIXME-PUT-YOURS-HERE"
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://bpsmar-ai-openai-1.openai.azure.com",
    azure_ad_token_provider=token_provider,
)

with open("sample.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Your are an expert in computer vision."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see on that image \n "},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + encoded_image},
                },
            ],
        },
    ],
)

print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)

print(response)
