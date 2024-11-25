# News Search App with GPT Integration

This project is a Streamlit application that allows users to search for news articles using a custom search script and summarizes the results using OpenAI's GPT model through LangChain integration.

## Features

- **User Query Input**: Accepts a search query from the user.
- **Embedder Selection**: Allows the user to select the embedder type.
- **GPT Summarization**: Uses LangChain and GPT to provide a concise summary of the search results.
- **Interactive UI**: Provides an easy-to-use interface built with Streamlit.

## Prerequisites
Set in a .env the variables

PWD=(path work directory before "news-generated-by-rag")
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY=*****
OPENAI_API_KEY=*****

# Usage
```bash
streamlit run app.py
```

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.100.3:8501

