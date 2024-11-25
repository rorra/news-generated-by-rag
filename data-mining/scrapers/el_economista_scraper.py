from scrapers.base_scraper import BaseScraper


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
            title_tags = article.find_all(['h2', 'h3', 'h4'], class_='tit')
            for title_tag in title_tags:
                if title_tag and title_tag.a:
                    article_url = title_tag.a['href']
                    full_url = f'{self.base_url}{article_url}' if article_url.startswith('/') else article_url
                    articles.append(full_url)

        return articles

    def scrape_article(self, article_url):
        soup = self.get_soup(article_url)

        content = ''

        # Find the title and subheadline
        title = soup.find('h1', class_='tit-ficha')
        if title:
            title = title.get_text(strip=True)
        subheadline = soup.find('h2', class_='sufix-ficha')
        if subheadline:
            subheadline = subheadline.get_text(strip=True)
            content += subheadline + ('. ' if subheadline[-1]!='.' else ' ')

        # # Find the main content div
        content_div = soup.find('article', class_='content')
        if not content_div:
            return None


        # # Get unwanted tags and classes
        unwanted_tags = ['figure', 'div', 'span', 'link']
        unwanted_classes = ['image-detail', 'rela', 'image', 'date', 'media']
        unwanted_elements = content_div.findAll(name=unwanted_tags, class_=unwanted_classes)

        for element in content_div.children:
            if element in unwanted_elements: # Skip unwanted elements
                continue
            if element.name == 'link':  # Skip CSS link tags
                continue
            if element.name is not None:
                content += self.clean_and_get_text(element) + ' '

        # Extract the publication datetime
        published_at = self.extract_published_datetime(soup, article_url)

        return {
            'title': title,
            'url': article_url,
            'content': content,
            'published_at': published_at
        }
