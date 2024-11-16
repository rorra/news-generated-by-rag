import os
import json

def get_preprocessed_json_file(date: str) -> dict:
    """
    Dynamically retrieves the JSON content of a preprocessed file based on the given date.

    Args:
        date (str): The date in YYYYMMDD format (e.g., "20241116").

    Returns:
        dict: The loaded JSON data.

    Raises:
        FileNotFoundError: If no file matching the date is found.
    """
    # Get the absolute path to the current script directory
    script_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"Script directory: {script_dir}")

    # Move up one directory to the parent directory
    parent_dir = os.path.dirname(script_dir)
    print(f"Parent directory: {parent_dir}")

    # Construct the absolute path to data-preprocessing/preprocessed_files
    directory = os.path.join(parent_dir, 'data-preprocessing', 'preprocessed_files')
    print(f"Preprocessed files directory: {directory}")

    # Construct the file name
    file_name = f"{date}_preprocessed_files.json"

    # Construct the full file path
    file_path = os.path.join(directory, file_name)
    print(f"Looking for file at: {file_path}")

    # Check if the file exists
    if os.path.isfile(file_path):
        print(f"File found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"No preprocessed file found for the date: {date}")

