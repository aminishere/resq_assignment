
import csv
import json
import re
import time
import requests

from bs4 import BeautifulSoup


INPUT_FILE = "paper_manifest.csv"
OUTPUT_FILE = "questions.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:0.6b"
PROMPT = """
You are a senior reviewer for a top-tier AI conference
(e.g., ICLR, NeurIPS, ACL).

You will receive a paper abstract and other sections of the paper.

Your task is to identify the central research question.

GUIDELINES:

1. Identify the underlying scientific problem being addressed.

2. Consider the research idea and its contributions together.

3. Focus on the knowledge gap, limitation, or challenge
   that motivates the research.

4. Do not simply rewrite the research idea as a question.

5. Do not generate a list of questions.

6. The question should capture the central objective
   of the research.

7. Do not invent motivations or research gaps that are
   unsupported by the supplied information.

8. Write one concise, specific, meaningful research question.

REASONING:

Briefly explain why this question captures the central
research objective.

Connect the question to the research idea and its
main contributions.

Return ONLY valid JSON in this exact format:

{
    "question": "The central research question?",
    "reasoning": "Explanation of why this question represents the research objective."
}
"""


def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def get_paper(arxiv_id):

    url = f"https://arxiv.org/html/{arxiv_id}"

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    full_text = clean(soup.get_text(" ", strip=True))

    introduction = ""
    conclusion = ""

    for heading in soup.find_all(re.compile(r"^h[1-6]$")):

        heading_text = clean(heading.get_text()).lower()

        if "introduction" in heading_text and not introduction:
            introduction = clean(
                heading.parent.get_text(" ", strip=True)
            )

        if any(word in heading_text for word in [
            "conclusion", "concluding", "discussion", "summary"
        ]) and not conclusion:

            conclusion = clean(
                heading.parent.get_text(" ", strip=True)
            )

    return introduction, conclusion, full_text

def extract_question(title, abstract, introduction, conclusion, full_text):

    paper = f"""
TITLE:
{title}

ABSTRACT:
{abstract}

INTRODUCTION:
{introduction or "Not available"}

CONCLUSION:
{conclusion or "Not available"}

ADDITIONAL TEXT:
{full_text[:10000]}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": paper}
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192
            }
        },
        timeout=120
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    return json.loads(content)


def main():

    with open(INPUT_FILE, encoding="utf-8") as file:
        papers = list(csv.DictReader(file))

    if not papers:
        print("Manifest is empty. Nothing to process.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for i, paper in enumerate(papers, 1):

            arxiv_id = paper.get("arxiv_id", "").strip()
            title = paper.get("title", "").strip()
            abstract = clean(paper.get("abstract", "") or "")

            print(f"\nProcessing {i}/{len(papers)}: {arxiv_id}")

            if not arxiv_id or not title or not abstract:
                print("Skipped: Missing ID, title, or abstract.")
                continue

            try:

                introduction, conclusion, full_text = get_paper(arxiv_id)

                result = extract_question(
                    title,
                    abstract,
                    introduction,
                    conclusion,
                    full_text
                )

                record = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "question": result.get("question", ""),
                    "reasoning": result.get("reasoning", ""),
                    "human_reviewed_question": "",
                    "status": "pending"
                }

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

                print("Question:", record["question"])

            except Exception as error:
                print(f"Failed: {arxiv_id} | {error}")

            time.sleep(2)

    print("\nExtraction completed.")


if __name__ == "__main__":
    main()
