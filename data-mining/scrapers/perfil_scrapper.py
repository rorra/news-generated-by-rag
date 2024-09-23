from scrapers.base_scraper import BaseScraper
from typing import Optional
from datetime import datetime
from logger import logger


class Perfil(BaseScraper):
    def __init__(self):
        self.newspaper = "Perfil"
        section_urls = {
            'Economía': '/seccion/economia',
            'Internacional': '/seccion/internacional',
            'Política': '/seccion/politica',
            'Sociedad': '/seccion/sociedad',
        }
        super().__init__(base_url='https://www.perfil.com', section_urls=section_urls)

    def scrape_section(self, section_name, section_url):
        """
        Scrape the section page to get article URLs for a specific section.
        """
        soup = self.get_soup(self.base_url + section_url)

        articles = []

        for article in soup.find_all('article', class_='news'):
            title_tags = article.find_all(['h2', 'h3', 'h4'], class_='news__title')
            for title_tag in title_tags:
                parent_a = title_tag.find_parent('a')
                if parent_a and 'href' in parent_a.attrs:
                    article_url = parent_a['href']
                    full_url = f'{self.base_url}{article_url}' if article_url.startswith('/') else article_url
                    articles.append(full_url)

        return articles

    def scrape_article(self, article_url):
        """
        Visit each article page and extract the detailed content and publication datetime.
        """
        soup = self.get_soup(article_url)

        # Parse title and content
        title = soup.find('h1', class_='article__title').get_text(strip=True)
        content_div = soup.find('div', class_='article__content')
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
            meta_tag = soup.find('meta', {'name': 'datepublished'})
            datetime_str = meta_tag['content']
            published_time = datetime.strptime(datetime_str, '%B, %d %Y %H:%M:%S %z')
            return published_time
        except Exception as e:
            raise e
            logger.warning(f"Couldn't find published time: {article_url}")
            return None
