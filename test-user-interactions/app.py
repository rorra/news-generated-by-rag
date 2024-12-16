import os
import subprocess
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from models.db_models import Article

# Load environment variables from .env file
load_dotenv()

# Get environment variables
PWD = os.getenv('PWD')
print(PWD)

# Ensure that the PWD is set
if not PWD:
    st.error("PWD is not set. Please set it in your environment variables.")
    st.stop()

# Initialize the OpenAI LLM
llm = ChatOpenAI(model='gpt-4o-mini')

# Load environment variables from .env file
load_dotenv()

# Get the database URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(bind=engine)

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
        script_path = Path(f"{PWD}/../data-retrieval/scripts/search_news.py")
        # Build the command
        command = [
            "python", str(script_path),
            "--prompt", prompt,
            "--embedder", embedder,
            "--limit", "5",
            "--local",
            "--json",
        ]

        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        data = json.loads(str(result.stdout))

        original_ids = [result['id'] for result in data['results']]

        db = SessionLocal()

        articles = db.query(Article).filter(
            Article.id.in_(original_ids)
        ).all()

        db.close()

        return "\n".join([f"{article.title}\n{article.content}\n\n\n" for article in articles])
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
                system_template = """Eres un asistente de inteligencia artificial que responde preguntas sobre artículos de noticias en español."""
                prompt = f"""
                            Consulta del Usuario:  {user_input}

                            Resultados de la Búsqueda:
                            {search_results}

                            Por favor, proporcioná un resumen conciso de la información más relevante con un tono profesional y neutral.
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
