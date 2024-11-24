from scrapers.base_scraper import BaseScraper
from typing import Optional
from datetime import datetime
from logger import logger


class TN(BaseScraper):
    def __init__(self):
        self.newspaper = "TN"
        section_urls = {
            'Economía': '/economia',
            'Internacional': '/internacional',
            'Política': '/politica',
            'Sociedad': '/sociedad',
        }
        super().__init__(base_url='https://www.tn.com.ar', section_urls=section_urls)

    def scrape_section(self, section_name, section_url):
        """
        Scrape the section page to get article URLs for a specific section.
        """
        soup = self.get_soup(self.base_url + section_url)

        articles = []

        for article in soup.find_all('article', class_='card__container'):
            title_tags = article.find_all(['h2', 'h3', 'h4'], class_='card__headline')
            for title_tag in title_tags:
                if title_tag and title_tag.a:
                    article_url = title_tag.a['href']
                    full_url = f'{self.base_url}{article_url}' if article_url.startswith('/') else article_url
                    articles.append(full_url)

        return articles

    def scrape_article(self, article_url):
        """
        Visit each article page and extract the detailed content and publication datetime.
        """
        soup = self.get_soup(article_url)
        content = ''
        # Skip live articles
        if "envivo/24hs/" in article_url:
            return None

        # Parse title and content
        title = soup.find('h1', class_='article__title').get_text(strip=True)

        # Extract the subheading if it exists
        dropline_div = soup.find('h2', class_='article__dropline')
        if dropline_div:
            content += self.clean_and_get_text(dropline_div)

        # Extract the main content
        content_div = soup.find('div', class_='article__body')
        if content_div is None:
            return None
        content = self.clean_and_get_text(content_div)

        # Iterate over the children of the content div to extract the text
        for element in content_div.children:
            # Ignore figure elements (Photo captions)
            if element.name == 'figure':
                continue
            # Ignore "Leé también" paragraphs
            if element.name == 'p' and 'Leé' in element.get_text():
                continue
            content += " " + self.clean_and_get_text(element)

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
        except Exception:
            logger.warning(f"Couldn't find published time: {article_url}")
            return None
