from .authentication import token_provider
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from src.schemas import LiteratureReviewResult
from langchain_tavily import TavilySearch

def literature_agent(number_of_models: int, max_search_results: int = 10, additional_context: list[str] = []) -> list[AIMessage]:
    """
    This defines and runs a literature agent that searches online for candidate machine learning models to benchmark for predicting patient outcomes based on clinical data in the format of EHRSHOT.
    """
    llm = ChatOpenAI(
        model = "gpt-5.4-mini",
        base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key = token_provider,
    )
    search_tool = TavilySearch(
        max_results = max_search_results,
        topic = "general",
    )
    literature_agent = create_agent(
        model = llm,
        tools = [search_tool],
        response_format=ToolStrategy(LiteratureReviewResult)
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
            Please look online for a list of {number_of_models} candidate models to benchmark, and provide a summary of the documentation for each model.
        Make sure that there is enough information in the documentation to allow the next scientist in the research group to implement the model using just your information.
            """
        )
    ]
    for context in additional_context:
        messages.append(HumanMessage(context))

    trajectory = literature_agent.invoke({
        "messages": messages
    })
    return trajectory