# News Scraper

This project is a news scraper that scrapes news articles from different newspapers and saves them to a database. It uses BeautifulSoup for web scraping and SQLAlchemy for database operations.

## Requirements

- Python 3.8 or higher
- MySQL 8.0 or higher
- BeautifulSoup4

## Instructions

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up the database:
   - Create a new database in MySQL.
   - Update the `DATABASE_URL` in the `.env` file with the appropriate credentials.
   - Run alembic to create tha database schema
     ```
     alembic upgrade head
     ```

3. Create the database data:
   ```
   python3 db_init.py
   ```

4. Run the scraper:
   ```
   python3 main.py
   ```

# Data Mining

# Sitios de noticias a parsear

### Página 12
https://www.pagina12.com.ar/
Economía: https://www.pagina12.com.ar/secciones/economia
Internacional: https://www.pagina12.com.ar/secciones/el-mundo
Sociedad: https://www.pagina12.com.ar/secciones/sociedad

### TN
https://tn.com.ar/
Economía: https://tn.com.ar/economia/
Internacional: https://tn.com.ar/internacional/
Política: https://tn.com.ar/politica/
Sociedad: https://tn.com.ar/sociedad/

### Perfil
https://www.perfil.com/
Economía: https://www.perfil.com/seccion/economia
Internacional: https://www.perfil.com/seccion/internacional
Política: https://www.perfil.com/seccion/politica
Sociedad: https://www.perfil.com/seccion/sociedad

### Infobae
https://www.infobae.com/?noredirect
Economía: https://www.infobae.com/economia/
Internacional: https://www.infobae.com/america/
Política: https://www.infobae.com/politica/
Sociedad: https://www.infobae.com/cultura/

### El economista
https://eleconomista.com.ar/
Economía: https://eleconomista.com.ar/economia/
Internacional: https://eleconomista.com.ar/internacional/
Politica: https://eleconomista.com.ar/politica/

### Ámbito Financiero
https://www.ambito.com/
Economía: https://www.ambito.com/contenidos/economia.html
Politica: https://www.ambito.com/contenidos/politica.html



