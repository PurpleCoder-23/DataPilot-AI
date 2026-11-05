import os
import sys
import re

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.SchemA import ETLAgentSchema

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
    output_folder: str,
    format: str
) -> str:
    """
    Extract data from an API endpoint and save it to a folder.

    Args:
        url: The API endpoint from which to extract data.
        output_folder: Folder where the extracted data will be saved.
        format: Output format: csv, json, or parquet.

    Returns:
        A message indicating the extraction result.
    """
    etl_tools = ETLTools()

    return etl_tools.extract_load(
        url,
        output_folder,
        format
    )


@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str = "data/transform",
    output_format: str = "csv",
    user_question: str = (
        "Select useful columns and save the transformed dataset"
    ),
) -> str:
    """
    Transform a dataset using pandas and save the result.

    Args:
        input_file_path: Path to the input data file.
        output_folder: Destination folder. Defaults to data/transform.
        output_format: Output format: csv, json, or parquet.
            Defaults to csv.
        user_question: Specific instructions describing the requested
            data transformation.

    Returns:
        The transformation result and execution details.
    """
    etl_tools = ETLTools()

    # Get a preview/context of the input dataset.
    top_3_rows = etl_tools.transform_load_context(
        input_file_path,
        output_folder,
        output_format,
    )

    # Select the LLM used to generate pandas code.
    llm = pick_llm("high")

    prompt = f"""
You are a Python data analyst who uses pandas.

Generate executable Python pandas code to perform the requested
data transformation.

Rules:
1. Read the input dataset from the exact path provided.
2. Use pandas to inspect and transform the dataset.
3. Follow the user's requested transformation.
4. Save the transformed dataset inside the specified output folder.
5. Use the requested output format: {output_format}.
6. Create the output folder if it does not exist.
7. Use only columns that actually exist in the dataset.
8. Do not invent columns or data.
9. Output only executable Python code.
10. Do not include explanations or Markdown code fences.

Input file:
{input_file_path}

Output folder:
{output_folder}

Output format:
{output_format}

User's transformation request:
{user_question}

Dataset context:
{top_3_rows}
"""

    response = llm.invoke(prompt).content

    # Extract code from the model response.
    if not isinstance(response, str):
        raise TypeError(
            "The code-generation LLM did not return text."
        )

    pandas_code = response.strip()

    # Remove Markdown fences if the model includes them.
    pandas_code = re.sub(
        r"^\s*```(?:python)?\s*",
        "",
        pandas_code,
        flags=re.IGNORECASE,
    )
    pandas_code = re.sub(
        r"\s*```\s*$",
        "",
        pandas_code,
    ).strip()

    if not pandas_code:
        raise ValueError(
            "The LLM returned empty pandas code."
        )

    # Execute the generated pandas code.
    results = etl_tools.execute_code(pandas_code)

    return (
        f"Transformation execution completed.\n"
        f"Input file: {input_file_path}\n"
        f"Output folder: {output_folder}\n"
        f"Output format: {output_format}\n\n"
        f"Pandas code executed:\n{pandas_code}\n\n"
        f"Execution results:\n{results}"
    )


# ---------------------------------------------------------
# TOOLKIT AND LLM
# ---------------------------------------------------------

tools = [
    extract_load_tool,
    transform_load_tool,
]

llm = pick_llm("high")
llm_bind = llm.bind_tools(tools)


# ---------------------------------------------------------
# LLM NODE
# ---------------------------------------------------------

def llm_node(state: ETLAgentSchema):
    """
    Generate a response and request ETL tools when necessary.
    """

    messages = state.messages or []

    # Prevent repeated tool-calling cycles.
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
                        "Check the previous tool results and try "
                        "a more specific request."
                    )
                )
            ]
        }

    # Limit long message content while preserving message metadata.
    bounded_messages = []

    for message in messages:
        content = getattr(message, "content", "")

        if isinstance(content, str) and len(content) > 2000:
            message = message.model_copy(
                update={
                    "content": (
                        content[:2000] + "\n[Content truncated]"
                    )
                }
            )

        bounded_messages.append(message)

    system_message = SystemMessage(
        content=(
            "You are a Python Data Analyst with tools for extracting "
            "and transforming datasets.\n\n"

            "Choose the appropriate tool for the user's request.\n\n"

            "When calling extract_load_tool, provide url, "
            "output_folder, and format.\n\n"

            "When calling transform_load_tool, provide "
            "input_file_path, output_folder, output_format, "
            "and user_question whenever possible.\n"
            "For output_folder, use 'data/transform' if the user "
            "does not specify a destination.\n"
            "For output_format, use 'csv' if the user does not "
            "specify a format.\n"
            "Preserve the user's requested transformation in "
            "user_question.\n\n"

            "Do not claim that an operation succeeded unless "
            "the tool result supports that claim.\n"
            "After receiving a successful tool result, summarize "
            "the outcome instead of repeating the same tool call."
        )
    )

    final_answer = llm_bind.invoke(
        [system_message] + bounded_messages
    )

    return {
        "messages": [final_answer]
    }


# ---------------------------------------------------------
# TOOL NODE
# ---------------------------------------------------------

def tool_node(state: ETLAgentSchema):
    """
    Execute requested tools and return their results.
    """

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
                # StructuredTool.invoke applies declared defaults
                # to optional arguments omitted from the argument dict.
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

    # ETLAgentSchema.messages uses an append reducer.
    # Return only new messages, not the entire existing state.
    return {
        "messages": tools_results
    }


# ---------------------------------------------------------
# GRAPH AND EDGES
# ---------------------------------------------------------

etl_analyst_graph = StateGraph(ETLAgentSchema)

etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)


def is_tool_call(state: ETLAgentSchema):
    """
    Route to the tool node only if the latest AI message
    contains tool calls.
    """

    if not state.messages:
        return "end"

    last_message = state.messages[-1]

    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tool_node"

    return "end"


etl_analyst_graph.add_edge(
    START,
    "llm_node",
)

etl_analyst_graph.add_conditional_edges(
    "llm_node",
    is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END,
    },
)

etl_analyst_graph.add_edge(
    "tool_node",
    "llm_node",
)


etl_analyst = etl_analyst_graph.compile()
# ---------------------------------------------------------
# MAIN: TEST TRANSFORMATION
# ---------------------------------------------------------

if __name__ == "__main__":

    # Compile the graph.
    etl_analyst = etl_analyst_graph.compile()

    # Print the generated tool schema for debugging.
    print("Transform tool schema:")
    print(transform_load_tool.args)

    # Save the graph diagram.
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

    # Test the transformation workflow.
    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Transform the extracted Pokémon dataset at "
                        "'data/extract/extracted_data.csv'. "
                        "Inspect the columns and select useful fields "
                        "such as name, height, weight, and "
                        "base_experience if available. "
                        "Save the transformed dataset as CSV in "
                        "'data/transform'."
                    )
                )
            ]
        },
        config={
            "recursion_limit": 20
        },
    )

    print("\nETL Agent Response:")
    for message in response["messages"]:
        print(message)
