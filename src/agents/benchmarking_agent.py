from authentication import token_provider
from deepagents import create_deep_agent
from pathlib import Path
from literature_agent import NUMBER_OF_MODELS
from deepagents.backends import FilesystemBackend
from literature_agent import response
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch

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

benchmarking_agent = create_deep_agent(
   model = llm,
   tools = [search_tool],
   backend = FilesystemBackend(root_dir="/app/generated_code", virtual_mode=True),
)

messages = [
   SystemMessage(
       """
       You are an expert machine learning software engineer and biostatistician.
       Your task is to implement the machine learning models specified by a previous researcher who was in charge of literature review.
       You should write clean, modular, well-documented Python code suitable for benchmarking on EHRSHOT datasets.
       When documentation is incomplete, identify the missing assumptions explicitly instead of inventing behavior.
       Test your implementation when possible using the Python execution tool.
       """
   ),
   HumanMessage(
       f"""
       Implement the {NUMBER_OF_MODELS} models described in this literature review: {response}.
       Search the web if additional documentation and implementation details are needed.
       Produce modular Python code with separate files for the model architecture, training loop, configuration, and evaluation.
       Execute unit tests or sanity checks using the Python tool before returning the implementation.
       """
   ),
   SystemMessage(
       """
       You are an expert machine learning software engineer and biostatistician.
       You are working in a fresh, empty working directory ("/") — this is expected,
       not an error. Do not search for or expect a pre-existing repository. Create
       all files directly at the root of your filesystem using write_file.
       """
   ),
]
for step in benchmarking_agent.stream({"messages": messages}, stream_mode="updates"):
   print(step)