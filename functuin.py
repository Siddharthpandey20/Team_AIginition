import whisper
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
import requests
import os
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
import re
import easyocr
from PIL import Image
import numpy as np

# Initialize whisper model
whisper_model = whisper.load_model("base")

# Initialize EasyOCR reader (only do this once)
# You can add more languages if needed: ['en', 'fr', 'es', etc.]
reader = easyocr.Reader(['en'])

def extract_text_from_pdf(file_path):
    try:
        text = ""
        if not os.path.isfile(file_path):
            return f"❗ File not found: {file_path}"

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n" + page_text
                else:
                    print(f"❗ Skipping OCR for page {page_num + 1} (OCR disabled for speed).")

        return text.strip() if text.strip() else "❗ No text found in PDF."

    except Exception as e:
        return f"❗ Error reading PDF: {str(e)}"

def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        return text.strip() if text.strip() else "❗ No text found in CSV."
    except Exception as e:
        return f"❗ Error reading CSV: {e}"

def extract_text_from_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path)
        return result['text'].strip() if result['text'].strip() else "❗ No speech detected in audio."
    except Exception as e:
        return f"❗ Error processing audio: {e}"

def extract_text_from_image(file_path):
    try:
        # Read the image with PIL first to ensure it's a valid image
        img = Image.open(file_path)
        img_np = np.array(img)
        
        # Use EasyOCR to extract text
        results = reader.readtext(img_np)
        
        # Extract all detected text
        text = "\n".join([result[1] for result in results])
        
        return text.strip() if text.strip() else "❗ No text found in the image."
    except Exception as e:
        return f"❗ Error processing image: {e}"

def extract_text_from_url_simple(url):
    try:
        text = ""
        urls = []

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return "❗ Invalid URL format."

        # Fetch the page
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for tag in ['script', 'style', 'nav', 'footer', 'iframe']:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract title and text
        title = soup.title.string if soup.title else 'No Title'
        body_text = "\n".join(p.get_text(strip=True) for p in soup.find_all('p'))

        text += f"\n=== Page Title ===\n{title}\n\n=== Content ===\n{body_text}\n"

        # Extract all valid URLs from the page
        urls = [link.get('href') for link in soup.find_all('a', href=True)
                if link.get('href').startswith(('http://', 'https://'))]

        # Extract from nested URLs (limit to first 5)
        if urls:
            text += "\n\n=== Nested URLs Content ===\n"
            for nested_url in urls[:5]:
                try:
                    nested_response = requests.get(nested_url, timeout=5, headers=headers)
                    nested_soup = BeautifulSoup(nested_response.text, 'html.parser')
                    nested_text = nested_soup.get_text(strip=True)[:500]
                    text += f"\n\n[From {nested_url}]\n{nested_text}\n"
                except Exception as e:
                    text += f"\n❗ Could not extract from {nested_url}: {e}\n"

        return text.strip() if text.strip() else "❗ No content extracted from URL."

    except Exception as e:
        return f"❗ Error processing URL: {e}"


def extract_text_from_js_rendered_url(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            content = page.content()
            browser.close()
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text(strip=True)
    except Exception as e:
        return f"❗ Error processing JavaScript-rendered URL: {e}"

def extract_text_auto(file_path=None, url=None, js_render=False):
    if file_path:
        ext = os.path.splitext(file_path)[-1].lower()
        print(f"🔍 Processing file type: {ext}")

        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext == '.csv':
            return extract_text_from_csv(file_path)
        elif ext in ['.mp3', '.wav', '.m4a']:
            return extract_text_from_audio(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return extract_text_from_image(file_path)
        else:
            try:
                # Try to read as a text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return "❗ Unsupported file type or unable to read file."

    elif url:
        print(f"🔍 Processing URL: {url}")
        if js_render:
            return extract_text_from_js_rendered_url(url)
        else:
            return extract_text_from_url_simple(url)

    else:
        return "❗ Please provide a file path or URL."

# Optional: If you still want the web crawling functionality
visited_links = set()

def extract_clean_text(url, depth=1, max_depth=2):
    """
    Extracts necessary information from a documentation page, structures it properly,
    and removes deprecated or warning messages.
    """
    global visited_links
    if url in visited_links or depth > max_depth:
        return ""

    print(f"Extracting: {url} (Depth: {depth})")
    visited_links.add(url)

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in ["nav", "footer", "aside", "script", "style", "form", "header", "noscript"]:
            for element in soup.find_all(tag):
                element.decompose()

        structured_text = f"\n\n[ Section: {url} ]\n\n"  # Section title per link
        headers = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

        filtered_paragraphs = [p for p in paragraphs if not re.search(r"(deprecated|warning)", p, re.IGNORECASE)]

        if headers:
            structured_text += "\n".join(headers) + "\n\n"
        if filtered_paragraphs:
            structured_text += "\n".join(filtered_paragraphs)

        structured_text += "\n\n" + "=" * 50 + "\n\n"

        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(url, href)
            if urlparse(full_url).netloc == urlparse(url).netloc and full_url not in visited_links:
                structured_text += extract_clean_text(full_url, depth + 1, max_depth)

        return structured_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return f"\n[ Error fetching content from {url} ]\n"

if __name__ == "__main__":
    __all__ = ['extract_text_from_url_simple', 'extract_text_auto']
