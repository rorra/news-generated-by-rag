import requests
from bs4 import BeautifulSoup, NavigableString, Tag


class BaseScraper:
    def __init__(self, base_url, section_urls):
        """
        Base scraper class for parsing multiple sections per newspaper.
        :param base_url: The base URL of the newspaper
        :param section_urls: A dictionary where keys are section names and values are section URLs
        """
        self.base_url = base_url
        self.section_urls = section_urls

    def get_soup(self, url):
        """
        Fetch a page and parse it with BeautifulSoup.
        """
        response = requests.get(url)
        response.raise_for_status()  # Ensure we get a valid response
        return BeautifulSoup(response.text, 'html.parser')

    def scrape_section(self, section_name, section_url):
        """
        Scrape a specific section page to get article URLs.
        This should be overridden in subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def scrape_article(self, article_url):
        """
        Scrape an article page to get full content.
        This should be overridden in subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def clean_and_get_text(self, element):
        """
        Extract text from an element while preserving spaces between different elements.
        """
        # Remove any script tag from the content
        if element.find('script'):
            for script in element.find_all("script"):
                script.decompose()

        text = []
        for item in element.contents:
            if isinstance(item, NavigableString):
                text.append(item.strip())
            elif isinstance(item, Tag):
                text.append(self.clean_and_get_text(item))
        return ' '.join(filter(bool, text)).replace('\xa0', ' ')
