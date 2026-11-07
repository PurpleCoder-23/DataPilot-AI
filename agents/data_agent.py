import os
import re
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst
from agents.data_analyst import answer_dataset_question

from utils.llm_pick import pick_llm
from Models.schema import RouterSchema, DataAgentSchema

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# ERROR RESPONSE HELPER
# ---------------------------------------------------------

def create_error_response(agent_name: str, error: Exception) -> str:
    """Create a readable error message for the user."""

    print(f"{agent_name} error: {type(error).__name__}: {error}")

    return (
        f"{agent_name} could not complete the request.\n"
        f"Error type: {type(error).__name__}\n"
        "Please check the application logs and configuration."
    )


# ---------------------------------------------------------
# LLM ROUTER
# ---------------------------------------------------------

llm = pick_llm("high")

llm_router = llm.with_structured_output(
    RouterSchema,
    method="json_schema",
)


# ---------------------------------------------------------
# SIMPLE CONVERSATIONAL GREETINGS
# ---------------------------------------------------------

def is_general_message(message: str) -> bool:
    """Recognize standalone greetings and simple conversational messages."""

    normalized = " ".join(message.lower().strip().split())
    normalized = normalized.strip("!?., ")

    general_messages = {
        "hi", "hello", "hey", "hiya", "hello there", "hi there",
        "good morning", "good afternoon", "good evening",
        "how are you", "who are you", "what can you do",
        "thanks", "thank you", "thanks a lot",
    }

    return normalized in general_messages


def general_node(state: DataAgentSchema):
    """Respond to greetings without invoking a specialist agent."""

    message = " ".join(state.messages[-1].content.lower().strip().split())
    message = message.strip("!?., ")

    if message in {"hi", "hello", "hey", "hiya", "hello there", "hi there"}:
        response = (
            "Hello! I'm your Data Agent. I can help analyze your uploaded "
            "dataset, work with SQL Server, or perform ETL tasks. "
            "What would you like to do?"
        )
    elif message in {"good morning", "good afternoon", "good evening"}:
        response = (
            "Hello! I'm your Data Agent. You can ask me about your uploaded "
            "dataset, SQL Server, or ETL workflows."
        )
    elif message in {"how are you", "who are you", "what can you do"}:
        response = (
            "I'm your Main Data Agent. I can route questions to the Data "
            "Analyst for uploaded files, SQL Analyst for database questions, "
            "or ETL Analyst for extraction and transformation tasks."
        )
    else:
        response = "You're welcome! What would you like to work on next?"

    return {"messages": [AIMessage(content=response)]}


# ---------------------------------------------------------
# ROUTER NODE
# ---------------------------------------------------------

def router_node(state: DataAgentSchema, config):
    """Route requests while distinguishing uploaded-file SQL from SQL Server."""

    try:
        if not state.messages:
            raise ValueError("No user message was provided.")

        message = state.messages[-1].content

        if not isinstance(message, str) or not message.strip():
            raise ValueError("The user message is empty or invalid.")

        # Handle simple greetings without calling the LLM router.
        if is_general_message(message):
            return {"route_response": "general"}

        configurable = config.get("configurable", {})
        has_uploaded_dataset = configurable.get("dataset") is not None
        normalized = message.lower()

        # SQL wording alone does not mean the user wants SQL Server.
        # If a file is attached and the user asks to WRITE/GENERATE SQL,
        # route to Data Analyst so it can use that file's actual columns.
        sql_generation_phrases = (
            "write sql", "generate sql", "create sql", "sql query",
            "write a query", "generate a query", "create a query",
            "give me sql", "give me a sql",
        )
        explicit_database_phrases = (
            "sql server", "database", "db table", "database table",
            "execute this query", "run this query", "query the database",
            "in the database",
        )

        asks_to_generate_sql = any(
            phrase in normalized for phrase in sql_generation_phrases
        )
        explicitly_targets_database = any(
            phrase in normalized for phrase in explicit_database_phrases
        )

        named_table_reference = bool(
            re.search(
                r"\\b(?:from|in|query|display|show|select)\\s+(?:the\\s+)?"
                r"[a-z_][a-z0-9_]*\\s+table\\b",
                normalized,
            )
        ) or bool(re.search(r"\\b[a-z_][a-z0-9_]*\\s+table\\b", normalized))

        explicitly_targets_database = (
            explicitly_targets_database or named_table_reference
        )

        if (
            has_uploaded_dataset
            and asks_to_generate_sql
            and not explicitly_targets_database
        ):
            print("Router selected: data_analysis (SQL for uploaded dataset)")
            return {"route_response": "data_analysis"}

        # Handle simple greetings and thanks deterministically. This avoids
        # asking the LLM to return "general" through RouterSchema, whose
        # specialist-only choices are sql/etl/data_analysis.
        if is_general_message(message):
            return {"route_response": "general"}

        route_result = llm_router.invoke(
            f"""
Classify the user's request into exactly one route.

Routes:
- sql: questions about SQL Server/database tables, executing queries against
  the connected database, or retrieving database records
- etl: extracting, transforming, or loading data from files or URLs
- data_analysis: questions about the uploaded dataset, including generating
  SQL statements based on its columns, rows, statistics, missing values,
  duplicates, or patterns
- general: standalone greetings, thanks, or simple conversation

Uploaded dataset available: {has_uploaded_dataset}

Important routing rules:
- If the user asks to write/generate SQL for the uploaded file, choose
  data_analysis, not sql.
- Choose sql when the user explicitly names a database table (for example,
  "users table" or "rides table"), requests SQL Server, or asks to execute
  a query against the connected database.
- If an uploaded dataset is available and the question refers to its
  contents, choose data_analysis.
- Do not treat the phrase "SQL query" by itself as SQL Server execution.

User request:
{message}
"""
        )

        if route_result.answer not in {"sql", "etl", "data_analysis"}:
            raise ValueError(
                f"Invalid routing decision: {route_result.answer}"
            )

        print(f"Router selected: {route_result.answer}")

        return {
            "route_response": route_result.answer
        }

    except Exception as error:
        error_message = create_error_response("Router", error)

        return {
            "route_response": "error",
            "messages": [AIMessage(content=error_message)]
        }


# ---------------------------------------------------------
# ETL NODE
# ---------------------------------------------------------

def etl_node(state: DataAgentSchema):
    """Execute the ETL workflow using the user's request."""

    try:
        message = state.messages[-1].content

        response = etl_analyst.invoke(
            {
                "messages": [HumanMessage(content=message)]
            },
            config={"recursion_limit": 20},
        )

        etl_messages = response.get("messages", [])

        final_response = next(
            (
                msg.content
                for msg in reversed(etl_messages)
                if isinstance(msg, AIMessage)
                and isinstance(msg.content, str)
                and msg.content.strip()
            ),
            "ETL workflow completed without a final response.",
        )

        return {
            "messages": [AIMessage(content=final_response)]
        }

    except Exception as error:
        error_message = create_error_response("ETL Analyst", error)

        return {
            "messages": [AIMessage(content=error_message)]
        }


# ---------------------------------------------------------
# SQL NODE
# ---------------------------------------------------------

def sql_node(state: DataAgentSchema):
    """Execute the SQL workflow using the user's request."""

    try:
        message = state.messages[-1].content

        input_schema = {
            "messages": [],
            "user_question": message,
            "curated_query": "",
            "prompt_query_context": "",
            "is_safe": "No",
            "comments": "",
            "generated_sql_query": "",
            "sql_query_execution_result": "",
            "final_response": "",
        }

        response = sql_analyst.invoke(input_schema)

        final_response = response.get("final_response", "")

        if not final_response and response.get("messages"):
            final_response = response["messages"][-1].content

        if not final_response:
            final_response = "SQL Analyst completed without a final response."

        return {
            "messages": [AIMessage(content=str(final_response))]
        }

    except Exception as error:
        error_message = create_error_response("SQL Analyst", error)

        return {
            "messages": [AIMessage(content=error_message)]
        }


# ---------------------------------------------------------
# DATA ANALYST NODE
# ---------------------------------------------------------

def data_analysis_node(state: DataAgentSchema, config):
    """
    Answer questions about the DataFrame supplied by Streamlit.

    Streamlit passes the current uploaded DataFrame through:
    config["configurable"]["dataset"]
    """

    try:
        message = state.messages[-1].content

        configurable = config.get("configurable", {})
        dataset = configurable.get("dataset")

        if dataset is None:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Please upload a CSV, Excel, or supported dataset "
                            "in the dashboard before asking questions about it."
                        )
                    )
                ]
            }

        final_response = answer_dataset_question(
            df=dataset,
            question=message,
        )

        return {
            "messages": [AIMessage(content=str(final_response))]
        }

    except Exception as error:
        error_message = create_error_response("Data Analyst", error)

        return {
            "messages": [AIMessage(content=error_message)]
        }


# ---------------------------------------------------------
# ERROR NODE
# ---------------------------------------------------------

def error_node(state: DataAgentSchema):
    """Handle routing failures."""

    return {}


# ---------------------------------------------------------
# ROUTING FUNCTION
# ---------------------------------------------------------

def route_request(state: DataAgentSchema):
    """Select the next node using the router's decision."""

    if state.route_response == "sql":
        return "sql"

    if state.route_response == "etl":
        return "etl"

    if state.route_response == "data_analysis":
        return "data_analysis"

    if state.route_response == "general":
        return "general"

    if state.route_response == "error":
        return "error"

    raise ValueError(
        f"Unknown routing decision: {state.route_response}"
    )


# ---------------------------------------------------------
# BUILD DATA AGENT GRAPH
# ---------------------------------------------------------

data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)
data_agent_graph.add_node("data_analysis_node", data_analysis_node)
data_agent_graph.add_node("error_node", error_node)
data_agent_graph.add_node("general_node", general_node)

data_agent_graph.add_edge(START, "router_node")

data_agent_graph.add_conditional_edges(
    "router_node",
    route_request,
    {
        "sql": "sql_node",
        "etl": "etl_node",
        "data_analysis": "data_analysis_node",
        "general": "general_node",
        "error": "error_node",
    },
)

data_agent_graph.add_edge("sql_node", END)
data_agent_graph.add_edge("etl_node", END)
data_agent_graph.add_edge("data_analysis_node", END)
data_agent_graph.add_edge("general_node", END)
data_agent_graph.add_edge("error_node", END)

data_agent = data_agent_graph.compile()


# ---------------------------------------------------------
# SAVE GRAPH DIAGRAM
# ---------------------------------------------------------

def save_graph_diagram():
    """Save the Main Data Agent graph visualization."""

    try:
        from IPython.display import Image

        img = Image(
            data_agent.get_graph().draw_mermaid_png()
        )

        with open("data_agent_graph.png", "wb") as f:
            f.write(img.data)

        print("Graph diagram saved to data_agent_graph.png")

    except Exception as error:
        print(
            f"Could not save graph diagram: "
            f"{type(error).__name__}: {error}"
        )


# ---------------------------------------------------------
# TEST DATA AGENT
# ---------------------------------------------------------

if __name__ == "__main__":

    save_graph_diagram()

    # Change this request to test SQL, ETL, or uploaded-data routing.
    test_question = "How many rows and columns are in my uploaded dataset?"

    try:
        result = data_agent.invoke(
            {
                "messages": [HumanMessage(content=test_question)]
            },
            config={
                "recursion_limit": 20,
                "configurable": {
                    # Standalone test has no uploaded DataFrame.
                    "dataset": None
                },
            },
        )

        print("\nFinal Data Agent Response:")

        final_response = next(
            (
                message.content
                for message in reversed(result["messages"])
                if isinstance(message, AIMessage)
                and message.content
            ),
            "No final response was generated.",
        )

        print(final_response)

        print("\n--- Data Agent State Inspection ---")
        print("Total messages:", len(result["messages"]))
        print("Routing decision:", result.get("route_response"))

        for index, message in enumerate(result["messages"]):
            content = str(message.content)
            print(
                f"Message {index + 1}: "
                f"{type(message).__name__} - {content[:100]}"
            )

    except Exception as error:
        print("\nMain Data Agent execution failed.")
        print(f"Error type: {type(error).__name__}")
        print("Please check the application logs and configuration.")
