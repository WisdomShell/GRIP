import json
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

#Configuration your openai api key
client = openai.OpenAI(
    base_url='',
    api_key=''
)

INPUT_FILE = '/path/to/your/data/C.jsonl'
OUTPUT_FILE = INPUT_FILE.replace('.jsonl', '_tmp.jsonl')
MAX_WORKERS = 64
WRITE_EVERY = 64 

def build_prompt(question, retrieval, initial_intermediary):
    docs_str = retrieval
    return f"""
System:
Given:
  • Question: the original user question
  • Retrieved docs: 3 context list of retrieved documents
  • Initial_intermediary: the model's first attempt at a partial answer
Task:
1. Re-summarize the intermediary fact that is most useful for answering the question,
   by combining the question, the retrieved documents, and initial_intermediary.
   – If initial_intermediary was incorrect or incomplete, discard or correct it.
   – Verify whether any retrieved document contains incorrect or misleading information
     based on your own knowledge, and correct the intermediary fact accordingly.
   – If it was correct, you may refine it.
2. Based on that finalized intermediary fact, plus the question and retrieved documents,
   generate a new search query that is highly likely to return evidence needed
   for the full answer.
Output:
Produce exactly one line in the following format, with no extra text:
[INTERMEDIARY] <your refined known fact> [RETRIEVE] <your new query>
---
Question:
{question}
Retrieved docs:
{docs_str}
Initial_intermediary:
{initial_intermediary}
Your output:
"""


def process_entry(entry):
    try:
        question = entry["Question"]
        retrieval = entry.get("Retrieved_Context", "")
        initial_intermediary = entry.get("Intermediate_Answer", "")

        prompt = build_prompt(question, retrieval, initial_intermediary)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert query-generation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=80
        )
        entry["Output"] = response.choices[0].message.content.strip()
        return entry
    except Exception as e:
        print(f"[Skip] {entry.get('Question','<no question>')[:60]}... | {str(e)}")
        
        return None


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file does not exist: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_entries = [json.loads(line) for line in f]

    total = len(all_entries)

    
    processed_questions = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_questions.add(item.get("Question"))

    
    to_process = [e for e in all_entries if e.get("Question") not in processed_questions]

    if not to_process:
        
        print("All data is fully processed. Replacing input file...")
        if os.path.exists(INPUT_FILE):
            os.remove(INPUT_FILE)
        os.rename(OUTPUT_FILE, INPUT_FILE)
        return

    buffer = []
    success_count = 0
    skipped_count = 0  

    def flush_buffer(buf):
        if not buf:
            return
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
            for item in buf:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        buf.clear()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, e): e for e in to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="line"):
            result = future.result()
            if result is not None:
                buffer.append(result)
                success_count += 1

                if len(buffer) >= WRITE_EVERY:
                    flush_buffer(buffer)
            else:
                skipped_count += 1

    if buffer:
        flush_buffer(buffer)

    if success_count == len(to_process) - skipped_count:
        print(f"Execution finished. Success: {success_count}, Skipped/Failed: {skipped_count}.")
        print("Replacing input file with processed output...")
        if os.path.exists(INPUT_FILE):
            os.remove(INPUT_FILE)
        os.rename(OUTPUT_FILE, INPUT_FILE)
        print("Replacement done. Note: Total rows in the new file are fewer due to skipped items.")
    else:
        print(f"Run interrupted. Processed {success_count} success entries, {skipped_count} skips this run. Resume by running again.")

if __name__ == "__main__":
    main()