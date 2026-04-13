import json
import os
import re
import glob
import string
import argparse
import unicodedata
from typing import List

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import LlamaForCausalLM, AutoTokenizer
from beir.retrieval.search.lexical.elastic_search import ElasticSearch


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

TEMP_SUFFIX = ".tmp"
PROGRESS_FILE = "progress.json"


def load_progress(output_dir: str) -> dict:
    """Load saved progress counters, return defaults if not found."""
    path = os.path.join(output_dir, PROGRESS_FILE)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"count_A": 0, "count_B": 0, "count_C": 0, "count_D": 0, "processed_lines": 0}


def save_progress(output_dir: str, progress: dict) -> None:
    """Atomically save progress counters (write temp → rename)."""
    path = os.path.join(output_dir, PROGRESS_FILE)
    tmp_path = path + TEMP_SUFFIX
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def cleanup_temp_files(output_dir: str) -> None:
    """Remove any leftover .tmp files from a previous crashed run."""
    for tmp in glob.glob(os.path.join(output_dir, f"*{TEMP_SUFFIX}")):
        os.remove(tmp)
        print(f"[resume] Removed stale temp file: {tmp}")


def open_append(path: str):
    """Open a file in append mode (creates if missing)."""
    return open(path, "a", encoding="utf-8")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        device_map={"": local_rank},
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_batch(
    questions: List[str],
    model,
    tokenizer,
    max_input_length: int = 1024,
    max_new_tokens: int = 32,
) -> List[str]:
    system_prompt = (
        "You are a helpful assistant."
        "Please answer the following question, just output the answer without any explanation.\n "
        "Example: \n"
        "question:what college does everyone in gossip girl go to?\n"
        "answer:New York University \n "
        "question:who plays the mother of howard on big bang theory?\n"
        "answer:Carol Ann Susi\n"
        "question:when did the first pair of yeezys come out?\n"
        "answer:February 14, 2015 "
    )
    prompts = [
        (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"question:{q}\nanswer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        for q in questions
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
        )

    seq_len = inputs["input_ids"].shape[1]
    predictions = []
    for i in range(len(questions)):
        gen_ids = outputs[i][seq_len:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        predictions.append(raw_text)
    return predictions


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace."""
    text = unicodedata.normalize("NFD", text)
    # lower
    text = text.lower()
    # remove punctuation
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # fix whitespace
    return ' '.join(text.split())

def exact_match(prediction: str, refs: str) -> bool:
    if isinstance(refs, str):
        refs = [refs]
    normalized_pred = normalize_answer(prediction)
    return any(normalize_answer(r) == normalized_pred for r in refs)

def cover_em(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def cover_em_check(prediction: str, reference) -> bool:
    if isinstance(reference, str):
        reference = [reference]
    return any(cover_em(prediction, ans) for ans in reference)


def classify_prediction(prediction: str, reference) -> str:
    if isinstance(reference, str):
        reference = [reference]
    if exact_match(prediction, reference):
        return "A"
    if cover_em_check(prediction, reference):
        return "B"
    return "C"


def find_matched_answer(prediction: str, reference) -> str:
    if isinstance(reference, str):
        reference = [reference]
    for ans in reference:
        if prediction.strip().lower() == ans.strip().lower():
            return ans
    for ans in reference:
        if cover_em(prediction, ans):
            return ans
    return reference[0] if reference else ""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_batch(
    es: ElasticSearch,
    index_name: str,
    body_field: str,
    queries: List[str],
    top_k: int = 3,
) -> List[str]:
    if not queries:
        return []

    msearch_res = es.lexical_multisearch(texts=queries, top_hits=top_k)

    all_doc_ids = [doc_id for res in msearch_res for doc_id, _ in res["hits"]]
    unique_ids = list(set(all_doc_ids))

    if unique_ids:
        docs = es.es.mget(body={"ids": unique_ids}, index=index_name)["docs"]
        doc_map = {doc["_id"]: doc.get("_source", {}) for doc in docs}
    else:
        doc_map = {}

    results = []
    for res in msearch_res:
        bodies = [
            doc_map.get(doc_id, {}).get(body_field, "")
            for doc_id, _ in res["hits"]
        ]
        formatted = []
        for j, b in enumerate(bodies, 1):
            if b.strip():  
                formatted.append(f"[{j}]. {b}")
        results.append("\n".join(formatted))
    return results


def normalise_answer_field(raw):
    return raw  


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classification and Processing with torchrun support")
    parser.add_argument("--model_dir", required=True, help="Path to the model directory")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for classified files")
    parser.add_argument("--es_host", default="localhost", help="ElasticSearch hostname")
    parser.add_argument("--es_index", default="wiki", help="ElasticSearch index name")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--target", type=int, default=10000, help="Target count per category")
    parser.add_argument("--max_input_length", type=int, default=1024)
    args = parser.parse_args()
    args.max_new_tokens = 32

    # ---- distributed init ----
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
    rank = get_rank()
    world_size = get_world_size()

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        cleanup_temp_files(args.output_dir)

    if dist.is_initialized():
        dist.barrier()

    # ---- output paths ----
    output_file_A = os.path.join(args.output_dir, "A.jsonl")
    output_file_B = os.path.join(args.output_dir, "B.jsonl")
    output_file_C = os.path.join(args.output_dir, "C.jsonl")
    output_file_D = os.path.join(args.output_dir, "D.jsonl")

    # ---- resume ----
    progress = load_progress(args.output_dir)
    if is_main_process():
        print(f"[main] Loaded progress: A={progress['count_A']}, B={progress['count_B']}, "
              f"C={progress['count_C']}, D={progress['count_D']}, processed={progress['processed_lines']}")

    # Here global_count_X represents the global true quantity
    global_count_A = progress["count_A"]
    global_count_B = progress["count_B"]
    global_count_C = progress["count_C"]
    global_count_D = progress["count_D"]
    processed_lines = progress["processed_lines"]
    current_processed = processed_lines

    target_A = target_B = target_C = target_D = args.target

    # ---- load model ----
    if is_main_process():
        print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # ---- elasticsearch ----
    es_config = {
        "hostname": args.es_host,
        "index_name": args.es_index,
        "keys": {"title": "title", "body": "txt"},
        "timeout": 100,
        "retry_on_timeout": True,
        "number_of_shards": "default",
        "maxsize": 24,
        "language": "english",
    }
    es = ElasticSearch(es_config)

    # ---- load data ----
    if is_main_process():
        print(f"Loading data (skipping first {processed_lines} lines)...")

    with open(args.input_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    all_data = all_data[processed_lines:]

    for item in all_data:
        ans = item.get("answer")
        if ans is None or (isinstance(ans, list) and not ans):
            item["answer"] = ""
            if is_main_process():
                print(f"[Warning] Missing or empty 'answer' field in item: {item.get('question', 'N/A')}")

    # ---- shard data across GPUs ----
    shard_data = all_data[rank::world_size]

    # ---- open output files ----
    fa = open_append(output_file_A)
    fb = open_append(output_file_B)
    fc = open_append(output_file_C)
    fd = open_append(output_file_D)

    try:
        iterator = range(0, len(shard_data), args.batch_size)
        if is_main_process():
            iterator = tqdm(iterator, desc="Processing batches")

        for i in iterator:
            if global_count_A >= target_A and global_count_B >= target_B and global_count_C >= target_C and global_count_D >= target_D:
                if is_main_process():
                    print("All targets reached. Terminating early.")
                break

            try:
                batch = shard_data[i: i + args.batch_size]
                questions = [item["question"] for item in batch]
                references = [item.get("answer", "") for item in batch]

                predictions = generate_batch(
                    questions, model, tokenizer,
                    max_input_length=args.max_input_length,
                    max_new_tokens=args.max_new_tokens,
                )

                batch_A, batch_B, batch_C = [], [], []
                batch_D_candidates = []  
                
                # Create a tensor to record the newly generated counts of each category in this batch on the current GPU (for multi-GPU synchronization)
                local_counts = torch.zeros(4, dtype=torch.int64, device=model.device)

                for q, ref, pred in zip(questions, references, predictions):
                    if global_count_A >= target_A and global_count_B >= target_B and global_count_C >= target_C and global_count_D >= target_D:
                        break

                    classification = classify_prediction(pred, ref)
                    matched_answer = find_matched_answer(pred, ref)

                    if classification == "A":
                        if global_count_A < target_A:
                            batch_A.append({
                                "Question": q,
                                "Output": f"[ANSWER] {matched_answer} [SOLVED]",
                                "Intermediate_Answer": "",
                                "Retrieved_Context": "",
                            })

                    elif classification == "B":
                        if global_count_B < target_B:
                            batch_B.append({
                                "Question": q,
                                "Output": f"[INTERMEDIARY] {pred} [RETRIEVE] {q}",
                                "Intermediate_Answer": "",          
                                "Retrieved_Context": "",
                            })
                        else: # If B is full, the samples where the model outputs CoverEM can be used as retrieval candidates for D
                            if global_count_D < target_D:
                                batch_D_candidates.append({
                                    "Question": q,
                                    "ref": ref,
                                    "pred": pred,
                                    "matched_answer": matched_answer,
                                })

                    else:  # classification == "C" 
                        if global_count_C < target_C:
                            batch_C.append({
                                "Question": q,
                                "Output": "", 
                                "Intermediate_Answer": pred,
                                "Retrieved_Context": "",
                            })
                        else: # If C is full, samples that are neither CoverEM nor EM are also used as retrieval candidates for D
                            if global_count_D < target_D:
                                batch_D_candidates.append({
                                    "Question": q,
                                    "ref": ref,
                                    "pred": pred,
                                    "matched_answer": matched_answer,
                                })

                # 1. Write to A and B
                for record in batch_A:
                    fa.write(json.dumps(record, ensure_ascii=False) + "\n")
                local_counts[0] = len(batch_A)

                for record in batch_B:
                    fb.write(json.dumps(record, ensure_ascii=False) + "\n")
                local_counts[1] = len(batch_B)

                # 2. Retrieve and write to C
                if batch_C:
                    queries_C = [item["Question"] for item in batch_C]
                    retrievals_C = retrieve_batch(
                        es, es_config["index_name"], es_config["keys"]["body"], queries_C
                    )
                    for record, retrieval in zip(batch_C, retrievals_C):
                        record["Retrieved_Context"] = retrieval   
                        fc.write(json.dumps(record, ensure_ascii=False) + "\n")
                    local_counts[2] = len(batch_C)

                # 3. Process retrieval candidates for D (sources can be the B branch after it's full, or the C branch after it's full)
                if batch_D_candidates:
                    queries_D = [item["Question"] for item in batch_D_candidates]
                    retrievals_D = retrieve_batch(
                        es, es_config["index_name"], es_config["keys"]["body"], queries_D
                    )
                    
                    count_d_this_batch = 0
                    for cand, retrieval in zip(batch_D_candidates, retrievals_D):
                        # Check if the retrieved document is CoverEM, if so, write to D
                        if cover_em_check(retrieval, cand["ref"]):
                            record = {
                                "Question": cand["Question"],
                                "Output": f"[ANSWER] {cand['matched_answer']} [SOLVED]",
                                "Intermediate_Answer": cand["pred"],
                                "Retrieved_Context": retrieval,   
                            }
                            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count_d_this_batch += 1
                    local_counts[3] = count_d_this_batch

                # ---- Force flush to disk ----
                fa.flush()
                fb.flush()
                fc.flush()
                fd.flush()

                # ---- [Key] Synchronize the growth of each category across GPUs! ----
                if dist.is_initialized():
                    dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)

                global_count_A += local_counts[0].item()
                global_count_B += local_counts[1].item()
                global_count_C += local_counts[2].item()
                global_count_D += local_counts[3].item()

                if is_main_process():
                    # Calculate total progress (recovered count + (current batch index + actual batch size) * world_size)
                    current_processed = processed_lines + min((i + len(batch)) * world_size, len(all_data))
                    save_progress(args.output_dir, {
                        "count_A": global_count_A,
                        "count_B": global_count_B,
                        "count_C": global_count_C,
                        "count_D": global_count_D,
                        "processed_lines": current_processed,
                    })

            except Exception as e:
                print(f"[rank {rank}] Error at batch {i}: {e}")
                continue

    finally:
        fa.close()
        fb.close()
        fc.close()
        fd.close()

        if is_main_process():
            save_progress(args.output_dir, {
                "count_A": global_count_A,
                "count_B": global_count_B,
                "count_C": global_count_C,
                "count_D": global_count_D,
                "processed_lines": current_processed,
            })

    # ===========================================================================
    # Second Pass: Find unwritten data and process it specifically for File B
    # ===========================================================================
    if global_count_B < target_B:
        if is_main_process():
            print(f"Target B not reached ({global_count_B}/{target_B}). Starting second pass for unwritten data...")
        
        if dist.is_initialized():
            dist.barrier()

        # Gather all written questions across A, B, C, D to filter out processed items
        written_questions = set()
        for filename in [output_file_A, output_file_B, output_file_C, output_file_D]:
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                if "Question" in record:
                                    written_questions.add(record["Question"])
                            except Exception:
                                pass

        # Read original dataset again to accurately find the missing elements
        with open(args.input_file, "r", encoding="utf-8") as f:
            full_data = [json.loads(line) for line in f]
        
        # Filter samples that are not present in any output file
        unwritten_data = [item for item in full_data if item.get("question") not in written_questions]

        # Shard unwritten data across GPUs
        shard_unwritten = unwritten_data[rank::world_size]
        
        fb = open_append(output_file_B)
        try:
            iterator_2 = range(0, len(shard_unwritten), args.batch_size)
            if is_main_process():
                iterator_2 = tqdm(iterator_2, desc="Second pass for File B")

            for i in iterator_2:
                if global_count_B >= target_B:
                    if is_main_process():
                        print("Target B reached during the second pass. Terminating early.")
                    break
                
                batch = shard_unwritten[i: i + args.batch_size]
                questions = [item["question"] for item in batch]
                references = [item.get("answer", "") for item in batch]

                predictions = generate_batch(
                    questions, model, tokenizer,
                    max_input_length=args.max_input_length,
                    max_new_tokens=args.max_new_tokens,
                )

                batch_B = []
                local_counts_B = torch.zeros(1, dtype=torch.int64, device=model.device)

                for q, ref, pred in zip(questions, references, predictions):
                    if global_count_B + local_counts_B[0].item() >= target_B:
                        break

                    classification = classify_prediction(pred, ref)
                    
                    # If model's output is neither CoverEM nor EM (i.e. 'C'), it can ALSO be treated as 'B'
                    if classification in ["B", "C"]:
                        batch_B.append({
                            "Question": q,
                            "Output": f"[INTERMEDIARY] {pred} [RETRIEVE] {q}",
                            "Intermediate_Answer": "",          
                            "Retrieved_Context": "",
                        })
                        local_counts_B[0] += 1

                for record in batch_B:
                    fb.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                fb.flush()

                if dist.is_initialized():
                    dist.all_reduce(local_counts_B, op=dist.ReduceOp.SUM)

                global_count_B += local_counts_B[0].item()

                if is_main_process():
                    save_progress(args.output_dir, {
                        "count_A": global_count_A,
                        "count_B": global_count_B,
                        "count_C": global_count_C,
                        "count_D": global_count_D,
                        "processed_lines": current_processed,
                    })
        except Exception as e:
            print(f"[rank {rank}] Error at second pass batch {i}: {e}")
        finally:
            fb.close()

    # ---- cleanup ----
    if dist.is_initialized():
        dist.destroy_process_group()

    if is_main_process():
        print(f"Done! A={global_count_A}, B={global_count_B}, C={global_count_C}, D={global_count_D}")


if __name__ == "__main__":
    main()