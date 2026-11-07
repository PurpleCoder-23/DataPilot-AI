import os
import sys

# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import ETLAgentSchema

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START, END
from langchain.tools import tool


# ---------------------------------------------------------
# ETL AGENT TOOLS
# ---------------------------------------------------------

@tool
def extract_load_tool(
    url: str,
    output_folder: str = "data/extract",
    format: str = "csv",
) -> str:
    """
    Extract data from a public JSON API and save it.

    Args:
        url: Public HTTP/HTTPS API endpoint.
        output_folder: Project-relative destination folder.
        format: Output format: csv, json, or parquet.
    """
    return ETLTools().extract_load(url, output_folder, format)




@tool
def inspect_dataset_tool(input_file_path: str) -> str:
    """
    Inspect a dataset before transforming it.

    Args:
        input_file_path: Project-relative path to CSV, Excel, JSON, or Parquet file.

    Returns:
        Dataset row/column counts, exact column names, and a small preview.
    """
    return ETLTools().transform_load_context(
        file_path=input_file_path,
        output_folder="data/transform",
        output_format="csv",
    )


@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str = "data/transform",
    output_format: str = "csv",
    select_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    filter_column: str | None = None,
    filter_operator: str | None = None,
    filter_value: str | int | float | bool | None = None,
    sort_by: list[str] | None = None,
    ascending: bool = True,
    drop_duplicates: bool = False,
    limit: int | None = None,
) -> str:
    """
    Transform a dataset using a controlled set of Pandas operations.

    Supported operations:
    - Select columns
    - Drop columns
    - Filter rows using supported operators
    - Sort rows
    - Remove duplicate rows
    - Limit number of rows
    - Save as CSV, JSON, or Parquet

    This tool does not execute generated Python code.
    """
    etl_tools = ETLTools()

    return etl_tools.transform_dataset(
        input_file_path=input_file_path,
        output_folder=output_folder,
        output_format=output_format,
        select_columns=select_columns,
        drop_columns=drop_columns,
        filter_column=filter_column,
        filter_operator=filter_operator,
        filter_value=filter_value,
        sort_by=sort_by,
        ascending=ascending,
        drop_duplicates=drop_duplicates,
        limit=limit,
    )


# ---------------------------------------------------------
# TOOLKIT AND LLM
# ---------------------------------------------------------

tools = [
    extract_load_tool,
    inspect_dataset_tool,
    transform_load_tool,
]

llm = pick_llm("high")
llm_bind = llm.bind_tools(tools)


# ---------------------------------------------------------
# LLM NODE
# ---------------------------------------------------------

def llm_node(state: ETLAgentSchema):
    """Generate a response and request ETL tools when needed."""

    messages = state.messages or []

    # Prevent repeated tool-calling cycles
    tool_call_count = sum(
        1
        for message in messages
        if isinstance(message, AIMessage)
        and bool(message.tool_calls)
    )

    if tool_call_count >= 4:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Stopped after four tool-calling rounds. "
                        "Please check the previous results and try "
                        "a more specific request."
                    )
                )
            ]
        }

    # Limit long message content while preserving message metadata
    bounded_messages = []

    for message in messages:
        content = getattr(message, "content", "")

        if isinstance(content, str) and len(content) > 2000:
            message = message.model_copy(
                update={
                    "content": content[:2000] + "\n[Content truncated]"
                }
            )

        bounded_messages.append(message)

    system_message = SystemMessage(
        content=(
            "You are a Data Analyst assistant with tools for extracting "
            "and transforming datasets.\n\n"

            "IMPORTANT SAFETY RULE:\n"
            "Never generate, request, or execute Python code. "
            "The transformation tool supports only structured operations.\n\n"

            "For API extraction, call extract_load_tool with the API URL, "
            "destination folder, and format. Only use public HTTP/HTTPS "
            "JSON API endpoints.\n\n"

            "For file-based dataset transformations, first call "
            "inspect_dataset_tool when exact column names are not already "
            "known. Then call transform_load_tool with exact columns and "
            "structured operation arguments.\n\n"

            "IMPORTANT: Do not invent file paths. If the user refers to "
            "an uploaded file but no accessible project-relative path is "
            "provided in the conversation, ask for the file path or wait "
            "for the application to pass the uploaded file into the ETL "
            "workflow. Do not ask for a file path if one is already present.\n\n"

            "This ETL tool currently reads project files using a "
            "project-relative path. It does not directly access Streamlit's "
            "temporary upload object or the Data Analyst's in-memory "
            "DataFrame. The app must pass an accessible path or handle "
            "uploaded-DataFrame transformations directly.\n\n"

            "Supported transformation operations:\n"
            "- select_columns: list of column names to keep\n"
            "- drop_columns: list of column names to remove\n"
            "- filter_column + filter_operator + filter_value\n"
            "- sort_by: list of column names\n"
            "- ascending: true or false\n"
            "- drop_duplicates: true or false\n"
            "- limit: positive integer\n\n"

            "Supported filter_operator values:\n"
            "equals, not_equals, greater_than, greater_than_or_equal, "
            "less_than, less_than_or_equal, contains, is_null, not_null.\n\n"

            "Use only column names that are present in the dataset preview "
            "or explicitly provided by the user. If a requested operation "
            "cannot be represented by these options, explain the limitation "
            "and ask the user to choose a supported operation.\n\n"

            "Defaults:\n"
            "- output_folder: data/transform\n"
            "- output_format: csv\n\n"

            "Do not claim an operation succeeded unless the tool result "
            "confirms success. If a tool returns TRANSFORMATION_FAILED or "
            "EXECUTION_BLOCKED, explain that it failed and do not claim "
            "the file was saved.\n\n"

            "After a successful tool result, summarize the output path, "
            "row count, and columns when available. Do not repeat the "
            "same tool call unnecessarily. If a tool result says failed, "
            "report the failure and its cause plainly."
        )
    )

    response = llm_bind.invoke(
        [system_message] + bounded_messages
    )

    return {"messages": [response]}


# ---------------------------------------------------------
# TOOL NODE
# ---------------------------------------------------------

def tool_node(state: ETLAgentSchema):
    """Execute requested tools and return their results."""

    tools_results = []

    tools_by_name = {
        current_tool.name: current_tool
        for current_tool in tools
    }

    last_message = state.messages[-1]

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        current_tool = tools_by_name.get(tool_name)

        if current_tool is None:
            result = f"Unknown tool: {tool_name}"
        else:
            try:
                result = current_tool.invoke(tool_args)
            except Exception as error:
                result = (
                    f"Tool execution failed: "
                    f"{type(error).__name__}: {error}"
                )

        tools_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tools_results}


# ---------------------------------------------------------
# GRAPH AND EDGES
# ---------------------------------------------------------

etl_analyst_graph = StateGraph(ETLAgentSchema)

etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)


def is_tool_call(state: ETLAgentSchema):
    """Route to the tool node only when the latest AI message has tool calls."""

    if not state.messages:
        return "end"

    last_message = state.messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"

    return "end"


etl_analyst_graph.add_edge(START, "llm_node")

etl_analyst_graph.add_conditional_edges(
    "llm_node",
    is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END,
    },
)

etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()


# ---------------------------------------------------------
# MAIN: TEST TRANSFORMATION
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Transform tool schema:")
    print(transform_load_tool.args)

    try:
        from IPython.display import Image

        graph_image = Image(
            etl_analyst.get_graph().draw_mermaid_png()
        )

        with open("etl_analyst_graph.png", "wb") as file:
            file.write(graph_image.data)

        print("Graph diagram saved to etl_analyst_graph.png")

    except Exception as error:
        print(f"Could not save graph diagram: {error}")

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Transform the extracted Pokémon dataset at "
                        "'data/extract/extracted_data.csv'. "
                        "Keep useful fields such as name, height, weight, "
                        "and base_experience if they exist. "
                        "Save the transformed dataset as CSV in "
                        "'data/transform'."
                    )
                )
            ]
        },
        config={"recursion_limit": 20},
    )

    print("\nETL Agent Response:")

    for message in response["messages"]:
        print(message)
