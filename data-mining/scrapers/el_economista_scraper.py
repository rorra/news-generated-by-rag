from db_init import newspapers
from scrapers.base_scraper import BaseScraper
from datetime import datetime
from logger import logger
import pytz
import json
from typing import Optional


class ElEconomistaScraper(BaseScraper):
    def __init__(self):
        self.newspaper = "El Economista"
        section_urls = {
            'Economía': '/economia/',
            'Sociedad': '/sociedad-redes/',
            'Política': '/politica/'
        }
        super().__init__(base_url='https://eleconomista.com.ar', section_urls=section_urls)

    def scrape_section(self, section_name, section_url):
        soup = self.get_soup(self.base_url + section_url)
        articles = []

        for article in soup.find_all('article', class_='noti-box'):
            title_tag = article.find('h2', class_='tit')
            if title_tag and title_tag.a:
                article_url = title_tag.a['href']
                full_url = f'{self.base_url}{article_url}' if article_url.startswith('/') else article_url
                articles.append(full_url)

        return articles

    def scrape_article(self, article_url):
        soup = self.get_soup(article_url)

        # Parse title
        title = soup.find('h1', class_='tit-ficha').get_text(strip=True)

        # Parse the content
        content_div = soup.find('article', class_='content')
        content = self.clean_and_get_text(content_div)

        # Extract the publication datetime
        published_at = self.extract_published_datetime(soup, article_url)

        return {
            'title': title,
            'url': article_url,
            'content': content,
            'published_at': published_at
        }

    def extract_published_datetime(self, soup, article_url: str) -> Optional[datetime]:
        """Extract the publication datetime from the article's JSON-LD data."""
        script_tags = soup.find_all('script', type='application/ld+json')

        for script_tag in script_tags:
            try:
                json_data = json.loads(script_tag.string, strict=False)

                # Check if this is a NewsArticle
                if isinstance(json_data, dict) and json_data.get('@type') == 'NewsArticle':
                    date_published = json_data.get('datePublished')
                    if date_published:
                        published_at = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                        return published_at.astimezone(pytz.UTC)
            except Exception as e:
                pass  # Fail silently

        logger.warning(f"No valid datePublished found in any JSON-LD script for article: {article_url}")
        return None
