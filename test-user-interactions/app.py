import os
import subprocess
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Get environment variables
PWD = os.getenv('PWD')

# Ensure that the PWD is set
if not PWD:
    st.error("PWD is not set. Please set it in your environment variables.")
    st.stop()

# Initialize the OpenAI LLM
llm = ChatOpenAI(model='gpt-4o-mini')

def user_query(prompt: str, embedder: str) -> str:
    """
    Executes the search_news.py script with the given prompt and embedder type.

    Args:
        prompt (str): The search prompt or query.
        embedder (str): The embedder type to use (e.g., 'minilm').

    Returns:
        str: The output from the script execution.
    """
    try:
        script_path = Path(f"{PWD}/news-generated-by-rag/data-retrieval/scripts/search_news.py")
        # Build the command
        command = [
            "python", str(script_path),
            "--prompt", prompt,
            "--embedder", embedder
        ]

        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        return result.stdout  # Return the script output
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

# Streamlit app
def main():
    st.title("News Search App with GPT Integration")

    # User inputs
    user_input = st.text_input("Enter your search query:", "")
    embedder = st.selectbox("Select an embedder:", ["minilm", "tfidf", "bm25", "dpr", "sbert"])

    if st.button("Search"):
        if not user_input:
            st.warning("Please enter a search query.")
        else:
            # Display a loading message
            with st.spinner("Searching..."):
                # Get the search results from your local script
                search_results = user_query(user_input, embedder)

                # Check for errors in search_results
                if search_results.startswith("Error:"):
                    st.error(search_results)
                    return

                # Prepare the prompt
                system_template = """You are an AI assistant that answers question about news articles in Spanish."""
                prompt = f"""
                            User Query: {user_input}

                            Search Results:
                            {search_results}

                            Please provide a concise summary of the most relevant information in professional and neutral language

                        """

                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", system_template), ("user", "{prompt}")]
                )


                try:
                    chain = prompt_template | llm
                    result = chain.invoke({"prompt": prompt})
                except Exception as e:
                    st.error(f"Error communicating with OpenAI API: {e}")
                    return

            # Display the results
            st.subheader("Summary")
            st.write(result.content)

            st.subheader("Raw Search Results")
            st.write(search_results)

if __name__ == "__main__":
    main()
