from .authentication import token_provider
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from src.schemas import LiteratureReviewResult
from langchain_tavily import TavilySearch

SYSTEM_PROMPT = """
You are an expert scientist in the field of biostatistics who is working on a research project.
Your research group is tasked with benchmarking the performance of various new machine learning
models to predict patient outcomes based on clinical data, specifically data in the format of
EHRSHOT. Your task is to provide a list of candidate models to benchmark, along with a summary
of the documentation for each model.
"""

def build_literature_agent(max_search_results: int = 10):
    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        base_url="https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
        api_key=token_provider,
    )
    search_tool = TavilySearch(max_results=max_search_results, topic="general")
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        response_format=ToolStrategy(LiteratureReviewResult),   # <-- the key line
    )
    return agent

def run_literature_review(agent, num_models: int) -> LiteratureReviewResult:
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"Please look online for {num_models} candidate models to benchmark, and "
            "provide a summary of the documentation for each. Make sure there is enough "
            "information for the next scientist to implement the model using just your "
            "information."
        ),
    ]
    result = agent.invoke({"messages": messages})
    return result["structured_response"]