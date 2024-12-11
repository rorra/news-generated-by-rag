## Installation

Set up environment variables:
    Create a `.env` file in the root directory and add the following:
    ```env
    QDRANT_URL=your_qdrant_url
    QDRANT_API_KEY=your_qdrant_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. **Generate Test Dataset:**
    Run the `test_generator.py` script to generate a test dataset:
    ```sh
    python test_generator.py
    ```

2. **Run Retrieval Tests:**
    Open the `test_retrieval_lab.ipynb` Jupyter Notebook and run the cells to test the retrieval system and plot the metrics.
