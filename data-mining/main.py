from scrapers.el_economista_scraper import ElEconomistaScraper
from scrapers.pagina_12_scraper import Pagina12Scraper
from scrapers.tn import TN
from config import SessionLocal
from models.db_models import Article, Newspaper, Section
from logger import logger
import time

def save_article(article_data, db_session, newspaper_name, section_name):
    """
    Save the article into the database if it doesn't already exist.
    Store foreign keys for newspaper and section, along with the publication datetime.
    """
    existing_article = db_session.query(Article).filter_by(link=article_data['url']).first()
    if not existing_article:
        newspaper = db_session.query(Newspaper).filter_by(name=newspaper_name).first()
        section = db_session.query(Section).filter_by(name=section_name).first()

        if not newspaper or not section:
            print(f"Error: Newspaper or Section not found.")
            return
        
        # Save the article with publication datetime
        new_article = Article(
            title=article_data['title'],
            link=article_data['url'],
            content=article_data['content'],
            newspaper=newspaper,
            section=section,
            published_at=article_data['published_at']  # Save the publication datetime
        )
        db_session.add(new_article)
        db_session.commit()
        print(f"Inserted article: {article_data['title']}")
    else:
        print(f"Article already exists: {article_data['title']}")



def run_scrapers():
    """
    Run all scrapers and store articles in the database for multiple sections.
    """
    db_session = SessionLocal()

    scrapers = [
        ElEconomistaScraper(),
        Pagina12Scraper(),
        TN(),
    ]
    
    for scraper in scrapers:
        logger.info("Starting to scrape %s", scraper.__class__.__name__)
        for section_name, section_url in scraper.section_urls.items():
            logger.info("Scraping section: %s", section_name)
            articles = scraper.scrape_section(section_name, section_url)
            logger.info("Found %s articles", len(articles))
            for article_url in articles:
                time.sleep(1) # Add a delay to avoid hitting the server too frequently
                logger.info("Scraping article: %s", article_url)
                article_data = scraper.scrape_article(article_url)
                if article_data:
                    article_data['section'] = section_name
                    save_article(article_data, db_session, scraper.newspaper, section_name)
    
    db_session.close()

if __name__ == "__main__":
    run_scrapers()
