import csv
import re
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, date

API_URL = "https://export.arxiv.org/api/query"

QUERIES = [
    "machine learning",
    "deep learning",
    "computer vision",
    "natural language processing",
    "robotics",
    "reinforcement learning",
    "multimodal learning",
]

MAX_PAPERS = 70
BATCH_SIZE = 70
CUTOFF_DATE = date(2025, 1, 1)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}


def extract_paper(entry):

    title = entry.findtext("atom:title", "", NS).strip()
    abstract = entry.findtext("atom:summary", "", NS).strip()
    paper_id = entry.findtext("atom:id", "", NS)
    published = entry.findtext("atom:published", "", NS)

    if not (title and abstract and paper_id and published):
        return None

    pub_date = datetime.fromisoformat(
        published.replace("Z", "+00:00")
    ).date()

    if pub_date < CUTOFF_DATE:
        return None

    arxiv_id = re.sub(r"v\d+$", "", paper_id.split("/")[-1])

    authors = [
        author.findtext("atom:name", "", NS)
        for author in entry.findall("atom:author", NS)
    ]

    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(title.split()),
        "abstract": " ".join(abstract.split()),
        "authors": "; ".join(authors),
        "publication_date": pub_date.isoformat(),
        "html_url": f"https://arxiv.org/html/{arxiv_id}"
    }


def search_arxiv(query, papers):

    start = 0
    while len(papers) < MAX_PAPERS:

        params = {
            "search_query": f'all:"{query}"',
            "start": start,
            "max_results": BATCH_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        response = requests.get(API_URL, params=params, timeout=60)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        entries = root.findall("atom:entry", NS)

        if not entries:
            break

        for entry in entries:
            paper = extract_paper(entry)

            if paper:
                papers.setdefault(paper["arxiv_id"], paper)

            if len(papers) >= MAX_PAPERS:
                return

        start += BATCH_SIZE
        time.sleep(3)


def save_csv(papers):

    columns = [
        "paper_id",
        "arxiv_id",
        "title",
        "publication_date",
        "authors",
        "abstract",
        "html_url",
    ]

    with open("paper_manifest.csv", "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for i, paper in enumerate(papers.values(), start=1):
            writer.writerow({
                "paper_id": i,
                **paper
            })


def main():

    papers = {}

    for query in QUERIES:

        print(f"Searching: {query}")
        search_arxiv(query, papers)

        print(f"Collected: {len(papers)} papers")

        if len(papers) >= MAX_PAPERS:
            break

    save_csv(papers)

    print("file saved in csv")


if __name__ == "__main__":
    main()
