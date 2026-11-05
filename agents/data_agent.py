import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from agents.ETL_analyst import etl_analyst
from agents.sql_analyst import sql_analyst

from utils.llm_pick import pick_llm
from Models.SchemA import RouterSchema, DataAgentSchema

from langchain_core.messages import HumanMessage ,AIMessage
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# LLM ROUTER
# ---------------------------------------------------------

llm = pick_llm("high")

llm_router = llm.with_structured_output(
    RouterSchema,
    method="json_schema"
)


# ---------------------------------------------------------
# ROUTER NODE
# ---------------------------------------------------------

def router_node(state: DataAgentSchema):
    """Classify the user request as SQL or ETL."""

    message = state.messages[-1].content

    route_result = llm_router.invoke(message)

    return {
        "route_response": route_result.answer
    }


# ---------------------------------------------------------
# ETL NODE
# ---------------------------------------------------------

def etl_node(state: DataAgentSchema):
    """Execute the ETL workflow using the user's request."""

    message = state.messages[-1].content

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(content=message)
            ]
        },
        config={"recursion_limit": 20},
    )

    # Return only the new agent messages.
    return {
        "messages": response["messages"]
    }


# ---------------------------------------------------------
# SQL NODE
# ---------------------------------------------------------

def sql_node(state: DataAgentSchema):
    """Execute the SQL workflow using the user's request."""

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

    # Extract the final response from the SQL agent's messages
    # or its final_response state field, depending on its output.
    final_response = response.get("final_response", "")

    if not final_response and response.get("messages"):
        final_response = response["messages"][-1].content

    from langchain_core.messages import AIMessage

    return {
        "messages": [
            AIMessage(content=str(final_response))
        ]
    }


# ---------------------------------------------------------
# ROUTING FUNCTION
# ---------------------------------------------------------

def route_request(state: DataAgentSchema):
    """Select the next node using the router's decision."""

    if state.route_response == "sql":
        return "sql"

    if state.route_response == "etl":
        return "etl"

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

data_agent_graph.add_edge(START, "router_node")

data_agent_graph.add_conditional_edges(
    "router_node",
    route_request,
    {
        "sql": "sql_node",
        "etl": "etl_node",
    },
)

data_agent_graph.add_edge("sql_node", END)
data_agent_graph.add_edge("etl_node", END)

data_agent = data_agent_graph.compile()


# ---------------------------------------------------------
# SAVE GRAPH DIAGRAM
# ---------------------------------------------------------

if __name__ == "__main__":

    try:
        from IPython.display import Image

        img = Image(
            data_agent.get_graph().draw_mermaid_png()
        )

        with open("data_agent_graph.png", "wb") as f:
            f.write(img.data)

        print("Graph diagram saved to data_agent_graph.png")

    except Exception as error:
        print(f"Could not save graph diagram: {error}")

    # Test the complete workflow.
    result = data_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Extract data from "
                        "https://pokeapi.co/api/v2/pokemon/pikachu "
                        "and save it as a CSV file."
                    )
                )
            ]
        },
        config={"recursion_limit": 20},
    )

    print("\nFinal Data Agent Response:")

    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            print(message.content)
            break
