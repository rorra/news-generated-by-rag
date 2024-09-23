from scrapers.base_scraper import BaseScraper
from typing import Optional
from datetime import datetime
from logger import logger


class Infobae(BaseScraper):
    def __init__(self):
        self.newspaper = "Infobae"
        section_urls = {
            'Economía': '/economia',
            'Internacional': '/america',
            'Política': '/politica/',
            'Sociedad': '/cultura',
        }
        super().__init__(base_url='https://www.infobae.com', section_urls=section_urls)

    def scrape_section(self, section_name, section_url):
        """
        Scrape the section page to get article URLs for a specific section.
        """
        soup = self.get_soup(self.base_url + section_url)

        articles = []

        articles_html = soup.find_all('a', class_='story-card-ctn')
        if not articles_html:
            articles_html = soup.find_all('a', class_='feed-list-card')

        for article in articles_html:
            article_url = article['href']
            full_url = f'{self.base_url}{article_url}' if article_url.startswith('/') else article_url
            articles.append(full_url)

        return articles

    def scrape_article(self, article_url):
        """
        Visit each article page and extract the detailed content and publication datetime.
        """
        soup = self.get_soup(article_url)

        # Parse title and content
        title = soup.find('h1', class_='article-headline').get_text(strip=True)
        content_div = soup.find('div', class_='body-article')
        if not content_div:
            content_div = soup.find('div', class_='body-blogging-article')
        content = self.clean_and_get_text(content_div)

        # Extract the publication datetime
        published_at = self.extract_published_datetime(soup, article_url)

        return {
            'title': title,
            'url': article_url,
            'content': content,
            'published_at': published_at  # Return the publication datetime
        }

    def extract_published_datetime(self, soup, article_url: str) -> Optional[datetime]:
        """Extract the publication datetime from the article's JSON-LD data."""
        try:
            meta_tag = soup.find('meta', property='article:published_time')
            datetime_str = meta_tag['content']
            published_time = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return published_time
        except Exception as e:
            raise e
            logger.warning(f"Couldn't find published time: {article_url}")
            return None
