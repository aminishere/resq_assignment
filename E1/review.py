import json
from pathlib import Path

QUESTIONS = Path(__file__).parent / "questions.jsonl"

def save_questions(questions):
    with open(QUESTIONS, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

def main():
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    for i, q in enumerate(questions):
        if q.get("status") == "approved":
            continue

        print(f"QUESTION -- {i + 1}/{len(questions)}")
        for key, value in q.items():
            print(f"\n{key}:\n{value}")

        print("Edit human_reviewed_question")
        edited = input("> ")

        if edited.strip():
            q["human_reviewed_question"] = edited
        q["status"] = (
            "approved"
            if q.get("human_reviewed_question", "").strip()
            else "pending"
        )

        save_questions(questions)
        print(f" updated status: {q['status']}")

    print("\nEntries completed")

if __name__ == "__main__":
    main()
