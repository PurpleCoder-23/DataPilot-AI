from langchain_core.messages import HumanMessage, AIMessage
from agents.data_agent import data_agent


def main():
    """Run the Data Agent in continuous terminal mode."""

    print("\n===================================")
    print("       DATA AGENT APPLICATION")
    print("===================================")
    print("Ask questions about your data.")
    print("Type 'exit' to quit.\n")

    while True:

        user_question = input("You: ").strip()

        # Exit the application
        if user_question.lower() == "exit":
            print("Exiting Data Agent. Goodbye!")
            break

        # Handle empty input
        if not user_question:
            print("Please enter a question.\n")
            continue

        try:
            # Invoke the existing LangGraph workflow
            result = data_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_question)
                    ]
                },
                config={"recursion_limit": 20},
            )

            # Extract the final AI response
            final_response = next(
                (
                    message.content
                    for message in reversed(result["messages"])
                    if isinstance(message, AIMessage)
                    and isinstance(message.content, str)
                    and message.content.strip()
                ),
                "No response was generated.",
            )

            print(f"\nData Agent: {final_response}\n")

        except Exception as error:
            print("\nThe Data Agent encountered an error.")
            print(f"Error type: {type(error).__name__}")
            print("Please check the application logs.\n")


if __name__ == "__main__":
    main()
