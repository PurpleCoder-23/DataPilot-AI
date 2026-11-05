from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file


def pick_llm(level:str):
    """
    Function to pick the appropriate LLM based on the level of complexity of the task or question.
    
    Args:
        level (str): The level of complexity of the task. Can be "simple", "intermediate", or "complex".
        
    Returns:
        str: The name of the LLM to be used for the task.
    """
    if level.lower() == "low":
        llm=ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    elif level.lower()== "medium":
        llm=ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    elif level.lower() == "high":
        llm=ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
    else:
        raise ValueError("Invalid level specified. Choose from 'low', 'medium', or 'high'.")
    return llm


if __name__ == "__main__":
    llm_obj=pick_llm("low")
    print(llm_obj.invoke("What is the capital of France?"))


##llm_obj=pick_llm("low")
##print(llm_obj.invoke("What is the capital of France?"))