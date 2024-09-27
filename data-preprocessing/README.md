## News Preprocessor
This project is a Python script designed to preprocess news articles stored in a database. The script applies various cleaning and normalization functions to the content of the articles and saves the preprocessed data into a JSON file. It is intended to be run from the console and can be scheduled with cron to automate its execution.

## Prerequisites
Python 3.8 or higher.
Access to a SQLAlchemy-compatible database containing the Article, Newspaper, and Section tables.
Have the packages listed in requirements.txt installed.
Install dependencies:
   ```
   pip install -r requirements.txt
   ```

Download NLTK data:
In the code the following line is necessary to donwlaod NLTK data:
#nltk.download('punkt')
#nltk.download('punkt_tab')

## Usage
Run the script from the console you could specify the publication date:
example of YYYY-MM-DD is 2024-01-01 will run from that day to the present.
 ```
python main.py --date YYYY-MM-DD
```
or just run 
```
 python main.py
``` 
will take the current day.