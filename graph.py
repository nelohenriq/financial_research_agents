import os
import json
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
import matplotlib
from pydantic import SecretStr
import requests
from langchain.schema.runnable import RunnableConfig
from langsmith import traecable

# Load environment variables
load_dotenv()
max_tokens = 2000

# Create the LLM
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

llm = ChatOpenAI(
    base_url=os.environ.get("BASE_URL"),
    api_key=SecretStr(groq_api_key),
    model=str(os.environ.get("MODEL")),
    temperature=0,
    timeout=None,
    max_retries=2,
)

# Define the tools
@traecable
@tool
def llm_tool(query: Annotated[str, "The query to search for."]):
    """A tool to call an LLM model to search for a query"""
    try:
        result = llm.invoke(query)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return result.content

# File management tools
file_tools = FileManagementToolkit(
    root_dir=str("./data"),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
read_tool, write_tool, list_tool = file_tools

# Python REPL tool
repl = PythonREPL()
@traecable
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute user instructions such as generate CSV or charts."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    If you need to save a plot, you should save it to the ./data folder. Assume the most default values for charts and plots. If the user has not indicated a preference, make an assumption and create the plot. Do not use a sandboxed environment. Write the files to the ./data folder, residing in the current folder.

    Clean the data provided before plotting a chart. If arrays are of unequal length, substitute missing data points with 0 or the average of the array.

    Example:
        Do not save the plot to (sandbox:/data/plot.png) but to (./data/plot.png)

    Example:
    ``` from matplotlib import pyplot as plt
        plt.savefig('./data/foo.png')
    ```
    """
    try:
        matplotlib.use('agg')
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

# Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)

# Alpha Vantage tool
@traecable
def get_natural_gas():
    response = requests.get(
        "https://www.alphavantage.co/query/",
        params={
            "function": "NATURAL_GAS",
            "apikey": os.getenv("ALPHAVANTAGE_API_KEY"),
        },
    )
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"API Error: {data['Error Message']}")

    return data
@traecable
@tool
def alpha_vantage_tool():
    """A tool to get Natural Gas prices from AlphaVantage API"""
    try:
        result = get_natural_gas()
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return result

# GDP tool
def get_gdp_data(country_code: str):
    response = requests.get(
        "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH",
    )
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"API Error: {data['Error Message']}")

    return data["values"]["NGDP_RPCH"][country_code]
@traecable
@tool
def gdp_tool(country_code: str):
    """A tool to get the GDP data for a given country code"""
    try:
        result = get_gdp_data(country_code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return result

# Agent state
class AgentState(TypedDict):
    messages: list
    next: str

# Member agent names
members = ["llm", "file_writer", "coder", "researcher", "alpha_vantage", "gdp_researcher"]

# System prompt for supervisor
system_prompt = f"""
You are a supervisor managing workers: {members}. Given the user's request, decide which worker should act next. Respond with a JSON object:
{{
    "next": "worker_name"
}}
where "worker_name" is one of {members} or "FINISH". Start by assessing if the query can be answered without tools; if not, proceed with tool usage as specified.
"""

# Supervisor node
@traecable
def supervisor_node(state: AgentState) -> AgentState:
    print("----------------- SUPERVISOR NODE START -----------------\n")
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    print(f"Prompt being sent to LLM: {messages}")
    
    # Get raw LLM response
    response = llm.invoke(messages)
    response_content = response.content.strip()
    
    # Attempt to parse response as JSON
    try:
        response_dict = json.loads(response_content)
        next_ = response_dict["next"]
    except json.JSONDecodeError:
        # Fallback in case of parsing error
        next_ = "llm"  # Default to LLM if parsing fails
        print(f"Failed to parse JSON from LLM response: {response_content}")
    
    if next_ == "FINISH":
        next_ = END
    
    print(f"Routing to {next_}")
    print("----------------- SUPERVISOR NODE END -----------------\n")
    return {"messages": state["messages"], "next": next_}

# Define agents
llm_agent = create_react_agent(
    llm, tools=[llm_tool], state_modifier="You are a highly-trained research analyst and can provide the user with the information they need. You are tasked with finding the answer to the user's question without using any tools. Answer the user's question to the best of your ability."
)

file_agent = create_react_agent(llm, tools=[write_tool])

@traecable
def file_node(state: AgentState) -> AgentState:
    result = file_agent.invoke(state)
    return AgentState(
        next="supervisor",
        messages=[HumanMessage(content=result["messages"][-1].content, name="file_writer")]
    )

@traecable
def llm_node(state: AgentState) -> AgentState:
    result = llm_agent.invoke(state)
    return AgentState(next="supervisor", messages=[HumanMessage(content=result["messages"][-1].content, name="llm_node")])

code_agent = create_react_agent(llm, tools=[python_repl_tool])

@traecable
def code_node(state: AgentState) -> AgentState:
    result = code_agent.invoke(state)
    return AgentState(
        next="supervisor",
        messages=[HumanMessage(content=result["messages"][-1].content, name="coder")]
    )

research_agent = create_react_agent(
    llm, tools=[tavily_tool], state_modifier="You are a highly-trained researcher. DO NOT do any math. You are tasked with finding the answer to the user's question. You have access to the following tools: Tavily Search. Use them wisely."
)

@traecable
def research_node(state: AgentState) -> AgentState:
    result = research_agent.invoke(state)
    return AgentState(next="researcher", messages=[HumanMessage(content=result["messages"][-1].content, name="researcher")])

alpha_vantage_agent = create_react_agent(llm, tools=[alpha_vantage_tool])

@traecable
def alpha_vantage_node(state: AgentState) -> AgentState:
    result = alpha_vantage_agent.invoke(state)
    return AgentState(next="alpha_vantage", messages=[HumanMessage(content=result["messages"][-1].content, name="alpha_vantage")])

gdp_agent = create_react_agent(llm, tools=[gdp_tool])

@traecable
def gdp_node(state: AgentState) -> AgentState:
    result = gdp_agent.invoke(state)
    return AgentState(
        next="gdp_researcher",
        messages=[HumanMessage(content=result["messages"][-1].content, name="gdp_researcher")]
    )

# Build the graph
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_edge(START, "supervisor")
builder.add_node("llm", llm_node)
builder.add_node("file_writer", file_node)
builder.add_node("coder", code_node)
builder.add_node("researcher", research_node)
builder.add_node("alpha_vantage", alpha_vantage_node)
builder.add_node("gdp_researcher", gdp_node)

# Add edges for all members to report back to supervisor
for member in members:
    builder.add_edge(member, "supervisor")

# Conditional edges for supervisor
builder.add_conditional_edges("supervisor", lambda state: state["next"])

# Compile the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Function to generate RunnableConfig
@traecable
def get_runnable_config(thread_id: str = "thread-1", run_name: str = "example_run") -> RunnableConfig:
    """
    Generates a RunnableConfig dictionary for the graph.

    Args:
        thread_id (str): The ID of the thread. Defaults to "thread-1".
        run_name (str): The name of the run. Defaults to "example_run".

    Returns:
        RunnableConfig: A dictionary containing the configuration for the graph.
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "placeholder_ns",  # Placeholder namespace
            "checkpoint_id": "placeholder_id"   # Placeholder checkpoint ID
        },
        "run_name": run_name,
        "tags": [],
        "metadata": {},
        "callbacks": [],
        "max_concurrency": None,
        "recursion_limit": 25,
        "run_id": None
    }

# Main loop
@traecable
def main_loop():
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Generate the config dynamically
        config = get_runnable_config(thread_id="user_thread", run_name="user_run")

        # Stream the graph with the user input
        for s in graph.stream(
            {
                "messages": [
                    HumanMessage(content=user_input, additional_kwargs={}, response_metadata={}, id='user_input_id')
                ]
            },
            config=config,
        ):
            print(s)
            print("----")

# Run the main loop
if __name__ == "__main__":
    main_loop()