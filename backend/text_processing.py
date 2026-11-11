import re
from bs4 import BeautifulSoup


def clean_readme(readme):

    if not readme or readme == "README_NOT_FOUND":
        return ""

    # Parse HTML
    soup = BeautifulSoup(readme, "html.parser")

    # Remove images, scripts, styles and SVGs
    for element in soup(["img", "script", "style", "svg"]):
        element.decompose()

    # Extract readable text
    text = soup.get_text(separator=" ")

    # Remove Markdown images and badges
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)

    # Remove reference-style Markdown images
    text = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", " ", text)

    # Preserve link text, remove link URLs
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # Remove empty Markdown links
    text = re.sub(r"\[\]\([^)]*\)", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Markdown headings and formatting
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"[*_`~]", "", text)

    # Remove unnecessary backslashes
    text = re.sub(r"\\(?=[^\w\s])", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()