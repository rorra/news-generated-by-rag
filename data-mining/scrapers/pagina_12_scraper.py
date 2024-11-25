from scrapers.base_scraper import BaseScraper


class Pagina12Scraper(BaseScraper):
    def __init__(self):
        self.newspaper = "Página 12"
        section_urls = {
            'Economía': '/secciones/economia',
            'Internacional': '/secciones/el-mundo',
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

        for article in soup.find_all('article', class_='article-item'):
            title_tags = article.find_all(['h2', 'h3', 'h4'], class_='title')
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
        title = ''

        # Parse title
        header_div = soup.find('div', class_='article-header')
        if header_div:
            title = header_div.find('h1').get_text(strip=True)
            # Maintain both subheadings
            subencabezados = header_div.find_all('h2')
            for subencabezado in subencabezados:
                if subencabezado is not None:
                    text = subencabezado.get_text(strip=True)
                    if len(text) > 0:
                        content += text + (' ' if text[-1]=='.' else '. ')


            # parse content
            content_div = soup.find('div', class_='article-main-content')
            if content_div is None:
                print( None)

            # Get unwanted tags and classes
            unwanted_tags = ['div']
            unwanted_classes = ['paywall', 'member-banner']
            unwanted_elements = content_div.findAll(name=unwanted_tags, class_=unwanted_classes)

            for el in unwanted_elements:
                el.decompose()

            for element in content_div.children:
                if element in unwanted_elements: # Skip unwanted elements
                    continue

                if element.name is not None:
                    content += self.clean_and_get_text(element) + ' '

        # Extract the publication datetime
        published_at = self.extract_published_datetime(soup, article_url)

        return {
            'title': title,
            'url': article_url,
            'content': content,
            'published_at': published_at  # Return the publication datetime
        }
