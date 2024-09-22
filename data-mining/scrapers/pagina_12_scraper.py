from scrapers.base_scraper import BaseScraper
from datetime import datetime


class Pagina12Scraper(BaseScraper):
    def __init__(self):
        self.newspaper = "Página 12"
        section_urls = {
            'Economía': '/secciones/economia',
            'Sociedad': '/secciones/sociedad',
            'Política': '/secciones/politica'
        }
        super().__init__(base_url='https://www.pagina12.com.ar', section_urls=section_urls)

    def scrape_section(self, section_name, section_url):
        """
        Scrape the section page to get article URLs for a specific section.
        """
        soup = self.get_soup(self.base_url + section_url)
        articles = []

        for article in soup.find_all('div', class_='article-item'):
            title_tag = article.find('h2', class_='title')
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

        # Parse title and content
        title = soup.find('h1', class_='article-title').get_text(strip=True)
        content = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))

        # Extract the publication datetime
        pub_date_str = soup.find('time', class_='date')['datetime']  # Adapt to the correct HTML structure
        published_at = datetime.strptime(pub_date_str, '%Y-%m-%dT%H:%M:%S')  # Adjust format as necessary

        return {
            'title': title,
            'url': article_url,
            'content': content,
            'published_at': published_at  # Return the publication datetime
        }
