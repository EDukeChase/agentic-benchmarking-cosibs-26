from .. import authentication.token_provider as token_provider
from openai import AzureOpenAI
from langchain_openai import ChatOpenAI

literature_llm = ChatOpenAI(
    model = "gpt-5.4-mini",
    base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
    api_key = token_provider,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = literature_llm.invoke(messages)
print(ai_msg.text)