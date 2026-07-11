# NEED TO FIND OUT HOW TO ALLOW THIS AGENT TO IMPLEMENT CODE IN SANDBOX
from authentication import token_provider
from langsmith.sandbox import SandboxClient
from deepagents.backends.langsmith import LangSmithSandbox
from deepagents.backends.langsmith import LangSmithSandboxBackend
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch

client = SandboxClient()
sandbox = client.create_sandbox()
backend = LangSmithSandboxBackend(sandbox=sandbox)


llm = ChatOpenAI(
    model = "gpt-5.4-mini",
    base_url = "https://bpsmar-ai-openai-1.openai.azure.com/openai/v1/",
    api_key = token_provider,
)

search_tool = TavilySearch(
    max_results = MAX_SEARCH_RESULTS,
    topic = "general",
)

programming_agent = create_agent(
    model = llm,
    tools = [search_tool],
    backend=backend,
)
try:
    messages = [
        SystemMessage(
            """
            You are an expert machine learning engineer working in computational healthcare.
            Your task is to implement machine learning models described by another researcher.
            You have access to:
            - internet search
            - a Python execution environment
            - a filesystem

            Your responsibilities:
            1. Inspect model documentation.
            2. Resolve missing implementation details.
            3. Write clean reproducible code.
            4. Run experiments.
            5. Debug failures.
            6. Report implementation decisions and limitations.

            Do not merely describe code. Create and execute it.
            """
        ),
        HumanMessage(
            f"""
            Please provide a detailed implementation of the model, including all necessary code and documentation to allow the next scientist in the research group to benchmark the model using just your information.
            Please do this in python using all necessary libraries.
            """
        )
    ]
    trajectory = programming_agent.invoke({
        "messages": messages
    })
    response = trajectory["messages"][-1].content
    print(response)
finally:
    client.delete_sandbox(sandbox.name)