
import csv
import re
import time
import requests
import xml.etree.ElementTree as ET

from datetime import date, datetime


# SETTINGS

API_URL = "https://export.arxiv.org/api/query"

MAX_PAPERS = 70
BATCH_SIZE = 70
CUTOFF_DATE = date(2025, 1, 1)

OUTPUT_FILE = "paper_manifest.csv"

QUERIES = [
    "machine learning",
    "deep learning",
    "computer vision",
    "natural language processing",
    "robotics",
    "reinforcement learning",
    "multimodal learning",
]

HEADERS = {"User-Agent": "ResearchQuestionDataset/1.0"}

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}


# EXTRACT PAPER INFORMATION

def extract_paper(entry):

    paper_id = entry.findtext("atom:id", namespaces=NS)
    title = entry.findtext("atom:title", namespaces=NS)
    abstract = entry.findtext("atom:summary", namespaces=NS)
    published = entry.findtext("atom:published", namespaces=NS)
    doi = entry.findtext("arxiv:doi", namespaces=NS)

    if not paper_id or not title or not abstract or not published:
        return None

    publication_date = datetime.fromisoformat(
        published.replace("Z", "+00:00")
    ).date()

    if publication_date < CUTOFF_DATE:
        return None

    arxiv_id = re.sub(r"v\d+$", "", paper_id.split("/")[-1])

    authors = []

    for author in entry.findall("atom:author", NS):
        name = author.findtext("atom:name", namespaces=NS)

        if name:
            authors.append(name)

    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(title.split()),
        "abstract": " ".join(abstract.split()),
        "doi": doi.strip() if doi else "",
        "publication_date": publication_date.isoformat(),
        "authors": "; ".join(authors),
        "html_url": f"https://arxiv.org/html/{arxiv_id}"
    }


# SEARCH ARXIV

def search_arxiv(query, target):

    papers = {}
    start = 0

    while len(papers) < target:

        params = {
            "search_query": f'all:"{query}"',
            "start": start,
            "max_results": BATCH_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        print(f"Fetching {query}: batch starting at {start}")

        response = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=60
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)
        entries = root.findall("atom:entry", NS)

        if not entries:
            break

        for entry in entries:

            paper = extract_paper(entry)

            if paper and paper["arxiv_id"] not in papers:
                papers[paper["arxiv_id"]] = paper

            if len(papers) >= target:
                break

        print(f"Collected: {len(papers)}")

        start += len(entries)

        if len(entries) < BATCH_SIZE:
            break

        time.sleep(3)

    return papers


# BUILD AND SAVE MANIFEST

def build_manifest():

    all_papers = {}

    for query in QUERIES:

        if len(all_papers) >= MAX_PAPERS:
            break

        remaining = MAX_PAPERS - len(all_papers)

        try:
            results = search_arxiv(query, remaining)

            for arxiv_id, paper in results.items():
                if arxiv_id not in all_papers:
                    all_papers[arxiv_id] = paper

        except (requests.RequestException, ET.ParseError) as error:
            print(f"Error searching {query}: {error}")

        print(f"Total unique papers: {len(all_papers)}")

    columns = [
        "paper_id",
        "arxiv_id",
        "title",
        "doi",
        "publication_date",
        "authors",
        "abstract",
        "html_url"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for number, paper in enumerate(all_papers.values(), start=1):
            writer.writerow({
                "paper_id": number,
                **paper
            })

    print("\nManifest completed!")
    print(f"Total papers: {len(all_papers)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_manifest()
