# News Website

A simple Flask web application for displaying generated news articles organized by sections. The application allows
users to browse news by date and view articles categorized by sections.

## Features

- Display news articles organized by sections
- Navigate news by date
- View full article content
- Responsive design using Bootstrap
- Spanish language interface
- Date navigation dropdown

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
# Copy the sample environment file
cp env.sample .env

# Edit .env with your credentials
# DATABASE_URL=mysql+mysqlconnector://user:password@localhost/news_db
```

## Running the Application

```bash
# Method 1: Using Flask command
export FLASK_APP=app.py 
export FLASK_ENV=development
flask run

# Method 2: Direct Python execution
python3 app.py
```

The application will be available at http://127.0.0.1:5000/

## Development

- The application runs in debug mode by default when using `python app.py`
- Templates use Jinja2 templating engine
- Bootstrap 5 is used for styling
- Custom CSS is minimal and builds upon Bootstrap
