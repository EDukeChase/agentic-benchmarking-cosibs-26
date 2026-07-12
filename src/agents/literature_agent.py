from authentication import token_provider
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch

NUMBER_OF_MODELS = 5
MAX_SEARCH_RESULTS = 10

llm = ChatOpenAI(
    model = "gpt-5.4-mini",
    base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
    api_key = token_provider,
)

search_tool = TavilySearch(
    max_results = MAX_SEARCH_RESULTS,
    topic = "general",
)

literature_agent = create_agent(
    model = llm,
    tools = [search_tool],
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
        Please look online for a list of {NUMBER_OF_MODELS} candidate models to benchmark, and provide a summary of the documentation for each model.
        Make sure that there is enough information in the documentation to allow the next scientist in the research group to implement the model using just your information.
        """
    )
]
trajectory = literature_agent.invoke({
    "messages": messages
})
response = trajectory["messages"][-1].content
print(response)