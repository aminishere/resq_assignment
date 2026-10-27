
import csv
import json
import re
import time
import requests

from pathlib import Path
from bs4 import BeautifulSoup


# CONFIGURATION

MANIFEST_FILE = "paper_manifest.csv"
OUTPUT_FILE = "extraction_candidates.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:0.6b"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 180
SLEEP_SECONDS = 1

MAX_ABSTRACT_CHARS = 3000
MAX_INTRO_CHARS = 10000
MAX_CONCLUSION_CHARS = 5000

HEADERS = {
    "User-Agent": "ResearchQuestionDataset/1.0"
}


# PROMPT

PROMPT = """
You are helping construct a dataset of research questions
from academic research papers.

Identify the central research question addressed by the paper.

Instructions:

1. Identify the main problem or objective investigated.
2. Write one clear, concise interrogative sentence.
3. The question may be implicit. Reconstruct it using the
   title, abstract, introduction and conclusion.
4. Be permissive. Generate a reasonable question even when
   information is incomplete.
5. Focus on the underlying research problem, not just the
   proposed technique or model.
6. Do not simply turn the title into a question.
7. Do not invent unsupported details.
8. Return an empty question only when virtually no research
   objective can be identified.

Return ONLY valid JSON:

{
    "extracted_question": "One research question?",
    "reasoning": "Brief explanation."
}
"""


# TEXT UTILITIES

def clean_text(text):
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def normalize_arxiv_id(paper_id):
    paper_id = str(paper_id).strip().rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", paper_id)


# HTML EXTRACTION

def extract_html(arxiv_id):

    response = requests.get(
        f"https://arxiv.org/html/{arxiv_id}",
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    abstract_element = soup.find(class_="ltx_abstract") or soup.find("abstract")

    abstract = ""
    if abstract_element:
        abstract = clean_text(abstract_element.get_text(" ", strip=True))

    introduction = ""
    conclusion = ""

    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):

        heading_text = clean_text(heading.get_text(" ", strip=True)).lower()

        section = heading.find_parent(
            lambda tag: tag.name == "section"
            or (tag.get("class") and any("ltx_section" in c for c in tag["class"]))
        )

        if section is None:
            continue

        section_text = clean_text(section.get_text(" ", strip=True))

        if heading_text.startswith("1 introduction") or heading_text == "introduction":
            introduction = section_text

        if "conclusion" in heading_text or "concluding remarks" in heading_text:
            conclusion = section_text

    full_text = clean_text(soup.get_text(" ", strip=True))

    return {
        "abstract": abstract,
        "introduction": introduction or full_text[:MAX_INTRO_CHARS],
        "conclusion": conclusion or full_text[-MAX_CONCLUSION_CHARS:],
        "source": "arXiv HTML"
    }


# GET PAPER CONTENT

def get_paper_text(paper):

    content = {
        "abstract": paper.get("abstract", ""),
        "introduction": "",
        "conclusion": "",
        "source": "Manifest abstract only"
    }

    try:
        html = extract_html(normalize_arxiv_id(paper["arxiv_id"]))

        if html["abstract"]:
            content["abstract"] = html["abstract"]

        content["introduction"] = html["introduction"]
        content["conclusion"] = html["conclusion"]
        content["source"] = "arXiv HTML + manifest abstract"

    except Exception as error:
        print(f"  HTML unavailable: {error}")
        print("  Using manifest abstract.")

    return content


# MODEL INPUT

def build_model_input(paper, content):

    return f"""
PAPER TITLE:
{paper.get("title", "")}

ABSTRACT:
{content["abstract"][:MAX_ABSTRACT_CHARS]}

INTRODUCTION:
{content["introduction"][:MAX_INTRO_CHARS]}

CONCLUSION:
{content["conclusion"][:MAX_CONCLUSION_CHARS]}

Identify the central research question.
"""


# OLLAMA

def call_ollama(paper, content):

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": build_model_input(paper, content)}
                    ],
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_ctx": 8192,
                        "num_predict": 512
                    }
                },
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            return json.loads(response.json()["message"]["content"])

        except Exception as error:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES}: {error}")

            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(3)


# VALIDATE MODEL OUTPUT

def validate_extraction(result):

    if not isinstance(result, dict):
        raise ValueError("Invalid model response.")

    question = str(result.get("extracted_question", "")).strip()
    reasoning = str(result.get("reasoning", "")).strip()

    if question and not question.endswith("?"):
        question += "?"

    return question, reasoning


# CHECKPOINTING

def load_completed_ids():

    completed = set()
    path = Path(OUTPUT_FILE)

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)

                if record.get("status") in ["pending", "completed", "no_question"]:
                    completed.add(record["arxiv_id"])

            except (json.JSONDecodeError, KeyError):
                continue

    return completed


def save_record(record):

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# MAIN PIPELINE

def main():

    if not Path(MANIFEST_FILE).exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    with open(MANIFEST_FILE, "r", encoding="utf-8", newline="") as file:
        papers = list(csv.DictReader(file))

    if not papers:
        raise ValueError("Manifest is empty.")

    completed_ids = load_completed_ids()

    print(f"Manifest papers: {len(papers)}")
    print(f"Already extracted: {len(completed_ids)}")
    print(f"Model: {MODEL}")

    successful = skipped = failed = 0

    for index, paper in enumerate(papers, start=1):

        arxiv_id = normalize_arxiv_id(paper.get("arxiv_id", ""))

        print(f"\n[{index}/{len(papers)}] Processing {arxiv_id}")

        if not arxiv_id:
            print("  Missing arXiv ID.")
            failed += 1
            continue

        if arxiv_id in completed_ids:
            print("  Already processed. Skipping.")
            skipped += 1
            continue

        try:
            content = get_paper_text(paper)

            print(f"  Source: {content['source']}")

            result = call_ollama(paper, content)
            question, reasoning = validate_extraction(result)

            record = {
                "paper_id": paper.get("paper_id", ""),
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "publication_date": paper.get("publication_date", ""),
                "extracted_question": question,
                "reasoning": reasoning,
                "human_review": "",
                "status": "pending",
                "model": MODEL
            }

            save_record(record)
            completed_ids.add(arxiv_id)

            successful += 1
            print(f"  Question: {question}")

        except Exception as error:
            print(f"  Failed: {error}")
            failed += 1

        time.sleep(SLEEP_SECONDS)

    print("\n========== SUMMARY ==========")
    print(f"Extracted: {successful}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Output:    {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
