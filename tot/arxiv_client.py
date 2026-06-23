import urllib.parse
import xml.etree.ElementTree as ET
import requests

class ArxivPaper:
    """Represents a research paper fetched from arXiv."""
    
    def __init__(self, title: str, summary: str, authors: list, url: str, pdf_url: str, published: str):
        self.title = title
        self.summary = summary
        self.authors = authors
        self.url = url
        self.pdf_url = pdf_url
        self.published = published

    def to_dict(self):
        """Converts the paper details into a dictionary format."""
        return {
            "title": self.title,
            "summary": self.summary,
            "authors": self.authors,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "published": self.published
        }

    def __str__(self):
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"[{self.published}] {self.title} - {authors_str}"

class ArxivClient:
    """Client to query the arXiv API and parse responses into structured Python objects."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 5) -> list:
        """
        Queries arXiv with a search string and parses the response.
        
        Args:
            query (str): The search query keywords.
            max_results (int): Maximum number of results to return.
            
        Returns:
            list[ArxivPaper]: List of parsed paper objects.
        """
        # Ensure query is clean and urlencoded. Search in all fields ('all:').
        # Using double quotes in the arXiv search string ensures exact phrase matches where needed.
        safe_query = urllib.parse.quote(f'all:"{query}"')
        url = f"{self.base_url}?search_query={safe_query}&max_results={max_results}"
        
        try:
            # Short timeout to keep queries snappy
            response = requests.get(url, timeout=50)
            response.raise_for_status()
            return self._parse_xml(response.text)
        except Exception as e:
            print(f"Error querying arXiv API: {e}")
            # Try a fallback query without exact phrase quotes in case it failed
            try:
                fallback_query = urllib.parse.quote(f"all:{query}")
                fallback_url = f"{self.base_url}?search_query={fallback_query}&max_results={max_results}"
                response = requests.get(fallback_url, timeout=35)
                response.raise_for_status()
                return self._parse_xml(response.text)
            except Exception as e2:
                print(f"Fallback arXiv query also failed: {e2}")
                return []

    def _parse_xml(self, xml_data: str) -> list:
        """Parses the Atom XML feed returned by arXiv."""
        papers = []
        try:
            root = ET.fromstring(xml_data)
            # Atom namespaces are defined in the XML feed root element
            namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', namespaces):
                # Title
                title_el = entry.find('atom:title', namespaces)
                title = title_el.text.strip().replace('\n', ' ') if title_el is not None else "No Title"
                title = " ".join(title.split())  # Clean extra spacing
                
                # Summary (Abstract)
                summary_el = entry.find('atom:summary', namespaces)
                summary = summary_el.text.strip().replace('\n', ' ') if summary_el is not None else "No Summary"
                summary = " ".join(summary.split())
                
                # Authors
                authors = []
                for author_el in entry.findall('atom:author', namespaces):
                    name_el = author_el.find('atom:name', namespaces)
                    if name_el is not None:
                        authors.append(name_el.text.strip())
                
                # URL / ID
                id_el = entry.find('atom:id', namespaces)
                url = id_el.text.strip() if id_el is not None else ""
                
                # Published date (ISO format like 2021-02-26T20:41:40Z)
                pub_el = entry.find('atom:published', namespaces)
                published = pub_el.text.strip() if pub_el is not None else ""
                if published:
                    published = published.split('T')[0]
                
                # Extract alternate (HTML web page) and PDF links
                pdf_url = ""
                for link in entry.findall('atom:link', namespaces):
                    rel = link.get('rel')
                    title_attr = link.get('title')
                    href = link.get('href')
                    
                    if rel == 'alternate':
                        url = href
                    elif rel == 'related' and title_attr == 'pdf':
                        pdf_url = href
                
                # Construct logical PDF URL if it wasn't explicitly labeled
                if not pdf_url and "/abs/" in url:
                    pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
                
                papers.append(ArxivPaper(
                    title=title,
                    summary=summary,
                    authors=authors,
                    url=url,
                    pdf_url=pdf_url,
                    published=published
                ))
        except Exception as e:
            print(f"Error parsing arXiv XML response: {e}")
            
        return papers
