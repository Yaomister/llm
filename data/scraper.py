import json
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "llm/1.0"
}


def scrape(topic):
    print(f"scraping {topic}")
    html = requests.get(f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}', headers=headers).text

    body = BeautifulSoup(html, "html.parser").select_one("div.mw-parser-output")

    for tag in body.select("sup.reference, .reflist, .navbox, table, .mw-editsection, style, .hatnote, .thumb"):
        tag.decompose()

    text = "\n\n".join([p.get_text(" ", strip=True) for p in body.find_all("p", recursive=True)])

    return text

def save(extracted_text):
    with open("data/datasets/text.txt", "w") as f:
        f.write("".join(extracted_text))

if __name__ == "__main__":

    extracted_text = []

    with open("data/pages.json", "r") as f:
        pages = json.load(f)
        for topic in pages:
            extracted_text.append(scrape(topic))

    save(extracted_text)