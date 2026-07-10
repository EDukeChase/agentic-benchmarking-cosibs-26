from authentication import token_provider
from openai import AzureOpenAI
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

NUMBER_OF_MODELS_TO_BENCHMARK = 5

benchmarking_llm = ChatOpenAI(
    model = "gpt-5.4-mini",
    base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
    api_key = token_provider,
)

messages = [
    SystemMessage(
        """
        You are an expert scientist in the field of biostatistics who is working on a research project. 
        Your research group is tasked with benchmarking the performance of various new machine learning models to predict patient outcomes based on clinical data, specifically data in the format of EHRSHOT. 
        Your task is to provide a list of candidate models to benchmark, along with a summary of the documentation for each model.
        """
    ),
    HumanMessage(
        f"""
        Please look online for a list of {NUMBER_OF_MODELS_TO_BENCHMARK} candidate models to benchmark, and provide a summary of the documentation for each model.
        Make sure that there is enough information in the documentation to allow the next scientist in the research group to implement the model using just your information.
        """
    )
]
ai_msg = benchmarking_llm.invoke(messages)
print(ai_msg.text)