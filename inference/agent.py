import json
import re
import os
import argparse
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Tuple
from beir.retrieval.search.lexical.elastic_search import ElasticSearch
from tqdm import tqdm
import math
import threading
import time

# ==========================================
# Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(description="Distributed GRIP Evaluation")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
parser.add_argument("--max_round", type=int, default=4, help="Maximum number of reasoning rounds")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")

# Use parse_known_args to avoid potential conflicts with torchrun arguments
args, _ = parser.parse_known_args()

# ==========================================
# DDP Setup
# ==========================================
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    # Fallback to single GPU/CPU if not using torchrun
    rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global cache and thread lock
_retrieval_cache = {}
_cache_lock = threading.Lock()

# ==========================================
# Retrieval Module Optimization
# ==========================================
es_instance = None
def init_es() -> ElasticSearch:
    global es_instance
    if es_instance is None:
        config = {
            'hostname': 'localhost',
            'index_name': 'wiki',
            'keys': {'title': 'title', 'body': 'txt'},
            'timeout': 100,
            'retry_on_timeout': True,
            'number_of_shards': 'default',
            'maxsize': 24,
            'language': 'english',
        }
        es_instance = ElasticSearch(config), config['index_name'], config['keys']['title'], config['keys']['body']
    return es_instance

def clean_query(query: str) -> str:
    """Clean the query string"""
    cleaned = re.sub(r'[^\w\s]', ' ', query).strip()
    return cleaned if cleaned else query

def batch_retrieve(queries: List[str]) -> List[str]:
    """Batch retrieval with cache optimization"""
    if not queries:
        return []
    
    results = []
    uncached_queries = []
    uncached_indices = []
    
    # Check cache
    with _cache_lock:
        for i, query in enumerate(queries):
            if query in _retrieval_cache:
                results.append(_retrieval_cache[query])
            else:
                results.append(None)
                uncached_queries.append(query)
                uncached_indices.append(i)
    
    # Process uncached queries in batch
    if uncached_queries:
        try:
            es, index_name, title_field, body_field = init_es()
            cleaned_queries = [clean_query(q) for q in uncached_queries]
            
            # Actual batch retrieval
            msearch_results = es.lexical_multisearch(texts=cleaned_queries, top_hits=3)
            
            for i, (original_idx, query, msearch_res) in enumerate(zip(uncached_indices, uncached_queries, msearch_results)):
                retrieved_text = ""
                
                if msearch_res and 'hits' in msearch_res and len(msearch_res['hits']) > 0:
                    doc_ids = [doc_id for doc_id, _ in msearch_res["hits"]]
                    if doc_ids:
                        docs = es.es.mget(body={"ids": doc_ids}, index=index_name)["docs"]
                        doc_map = {doc["_id"]: doc.get("_source", {}) for doc in docs}
                        bodies = [doc_map.get(doc_id, {}).get(body_field, "").strip()
                                for doc_id, _ in msearch_res["hits"] if doc_map.get(doc_id)]
                        retrieved_text = '\n'.join(bodies)
                
                # Update cache and results
                with _cache_lock:
                    _retrieval_cache[query] = retrieved_text
                results[original_idx] = retrieved_text
                
        except Exception as e:
            print(f"Batch retrieval error: {str(e)}")
            # Fill empty strings for failed queries
            for idx in uncached_indices:
                results[idx] = ""
    
    return results

# ==========================================
# Model Loading
# ==========================================
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path, use_fast=False, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the full model directly
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
)
model.to(device)
model.eval()

INSTRUCTION_NORMAL = """
Given the question and previous answers, as well as the following retrieved text, please provide the answer. If you are very confident on your answer, you can provide you answer follow by [ANSWER], and end with [SOLVED]. If you need more external knowledge, you should generate the temp answer follow by [INTERMEDIARY], and end with [RETRIEVE]. Besides, you need to generate the new query based on the original query and current temp answer.

For example:
- Case 1:
Output: [ANSWER] Complete answer [SOLVED]

- Case 2:
Output: [INTERMEDIARY] Partial answer [RETRIEVE] New Query.

The followings are the question you need to solve:
- Original Query:
    {question}

Here is some retrieved relevant information along with some previous responses.
- Intermediary:
    {intermediary}

- Reference Text:
    {reference}
""".strip()

INSTRUCTION_FORCE_ANSWER = """
Given the question and previous answers, as well as the following retrieved text, please provide the answer.  you MUST provide you answer follow by [ANSWER], and end with [SOLVED]. 

For example:
Output: [ANSWER] Complete answer [SOLVED]

The followings are the question you need to solve:
- Original Query:
    {question}

Here is some retrieved relevant information along with some previous responses.
- Intermediary:
    {intermediary}

- Reference Text:
    {reference}
""".strip()

def generate_responses(
    questions: List[str], ref_list: List[str] = None, ret_txt_list: List[str] = None,
    max_length: int = 2048, instructions_override: bool = False, force_answer: bool = False
) -> List[str]:
    instruction = INSTRUCTION_FORCE_ANSWER if force_answer else INSTRUCTION_NORMAL
    batch_ref = ref_list if ref_list else [""] * len(questions)
    batch_ret = ret_txt_list if ret_txt_list else [""] * len(questions)
    prompts = []
    for q, ref, rt in zip(questions, batch_ref, batch_ret):
        messages = [{"role": "user", "content": instruction.format(question=q, intermediary=ref, reference=rt)}]
        if instructions_override or force_answer:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "[ANSWER]"
        else:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                stop_strings=["<|endoftext|>", "<|end_of_text|>"],
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        all_outputs = []
        for i, out in enumerate(outputs):
            text = tokenizer.decode(out, skip_special_tokens=False)
            prompt_len = len(tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=False))
            response = text[prompt_len:].replace("<|endoftext|>", "").replace("<|end_of_text|>", "").strip()
            if "[ANSWER]" not in response and "[INTERMEDIARY]" not in response:
                all_outputs.append("[ANSWER]" + response)
            else:
                all_outputs.append(response)
        return all_outputs
    finally:
        del inputs
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()

class QuestionState:
    def __init__(self, question: str, index: int):
        self.question = question
        self.index = index
        self.ref = ""
        self.ret_txt = ""
        self.round = 0
        self.is_completed = False
        self.final_answer = ""
        self.completed_round = -1
        self.answers = []

# Pre-compile regular expressions to improve performance
ANSWER_PATTERN = re.compile(r"\[ANSWER\](.*?)\[SOLVED\]", re.DOTALL)
INTERMEDIARY_PATTERN = re.compile(r"\[INTERMEDIARY\](.*?)\[RETRIEVE\]", re.DOTALL)
RETRIEVE_PATTERN = re.compile(r"\[RETRIEVE\](.*)", re.DOTALL)

def extract_answer_from_response(response: str) -> str:
    """Extract answer using pre-compiled regular expressions"""
    answer_match = ANSWER_PATTERN.search(response)
    if answer_match:
        return answer_match.group(1).strip()
    
    intermediary_match = INTERMEDIARY_PATTERN.search(response)
    if intermediary_match:
        return intermediary_match.group(1).strip()
    
    return response.strip()

def process_batch_round_optimized(
    batch_states: List[QuestionState], current_round: int, max_rounds: int
) -> List[QuestionState]:
    """Optimized batch processing logic, keeping model inference unchanged"""
    active_states = [state for state in batch_states if not state.is_completed]
    if not active_states:
        return batch_states

    is_last_round = (current_round == max_rounds)

    questions = [state.question for state in active_states]
    refs = [state.ref for state in active_states]
    ret_txts = [state.ret_txt for state in active_states]
    
    responses = generate_responses(questions, refs, ret_txts, force_answer=is_last_round)
    
    retrieve_queries = []
    retrieve_indices = []
    
    for i, (state, response) in enumerate(zip(active_states, responses)):
        # Extract the answer of the current round and save to the array
        current_answer = extract_answer_from_response(response)
        state.answers.append(current_answer)
        
        # Check if it is the final answer
        if response.startswith("[ANSWER]") and "[SOLVED]" in response:
            answer_match = ANSWER_PATTERN.search(response)
            if answer_match:
                state.final_answer = answer_match.group(1).strip()
                state.is_completed = True
                state.completed_round = current_round
                continue

        # Check if it contains a retrieval request
        if "[RETRIEVE]" in response:
            # Extract content between [INTERMEDIARY] and [RETRIEVE] as the new ref
            intermediary_match = INTERMEDIARY_PATTERN.search(response)
            if intermediary_match:
                state.ref = "[INTERMEDIARY]" + intermediary_match.group(1) + "[RETRIEVE]"
            else:
                state.ref = response

            # Extract content after [RETRIEVE] as the retrieval query
            retrieve_match = RETRIEVE_PATTERN.search(response)
            if retrieve_match:
                new_query = retrieve_match.group(1).strip()
                if new_query:
                    retrieve_queries.append(new_query)
                    retrieve_indices.append(i)
        else:
            # No retrieval request and not the final answer, update ref
            state.ref = response
            state.ret_txt = ""

        state.round += 1
    
    # Batch retrieval - this is the main optimization point
    if retrieve_queries:
        retrieval_results = batch_retrieve(retrieve_queries)
        for result, idx in zip(retrieval_results, retrieve_indices):
            active_states[idx].ret_txt = result

    return batch_states

def process_questions_optimized(
     input_questions: List[str], output_file: str, max_rounds: int, 
     batch_size: int = 16, processed_questions: set = None
 ) -> List[dict]:
    
    results = []
    round_completion_counts = [0] * max_rounds
    num_batches = math.ceil(len(input_questions) / batch_size)
    
    # Batch writing optimization
    write_buffer = []
    buffer_size = 50
    
    with open(output_file, 'a', encoding='utf-8') as fout:
        for batch_idx in tqdm(range(0, len(input_questions), batch_size), 
                            desc=f"Processing batches (Rank {rank})", unit="batch", total=num_batches):
            
            batch_end = min(batch_idx + batch_size, len(input_questions))
            batch_questions = input_questions[batch_idx:batch_end]
            
            batch_states = [QuestionState(q, i) for i, q in enumerate(batch_questions)]
            
            # Multi-round inference
            for round_idx in range(max_rounds):
                current_round = round_idx + 1
                batch_states = process_batch_round_optimized(batch_states, current_round, max_rounds)
                
                if all(state.is_completed for state in batch_states):
                    break
            
            # Process unfinished questions
            for state in batch_states:
                if not state.is_completed:
                    if "[INTERMEDIARY]" in state.ref and "[RETRIEVE]" in state.ref:
                        intermediary_match = INTERMEDIARY_PATTERN.search(state.ref)
                        if intermediary_match:
                            final_answer = intermediary_match.group(1).strip()
                        else:
                            final_answer = state.ref
                    else:
                        final_answer = state.ref
                    
                    if final_answer not in state.answers:
                        state.answers.append(final_answer)
                    
                    state.final_answer = final_answer
                    state.completed_round = max_rounds + 1
            
            # Count the number of completed answers per round in the current batch
            for state in batch_states:
                if 1 <= state.completed_round <= max_rounds:
                    round_completion_counts[state.completed_round - 1] += 1
            
            # Buffered writing optimization
            for state in batch_states:
                record = {
                    "question": state.question, 
                    "prediction": state.answers
                }
                results.append(record)
                write_buffer.append(json.dumps(record, ensure_ascii=False) + "\n")
                processed_questions.add(state.question)
                
                # Write to file when the buffer is full or it is the last batch
                if len(write_buffer) >= buffer_size or batch_end == len(input_questions):
                    fout.writelines(write_buffer)
                    fout.flush()
                    write_buffer.clear()
    
    # Output statistical information
    print("\n" + "="*50)
    print(f"Rank {rank} Completion statistics for each round:")
    print("="*50)
    total_completed = sum(round_completion_counts)
    for i, count in enumerate(round_completion_counts):
        round_num = i + 1
        percentage = (count / len(input_questions)) * 100 if len(input_questions) > 0 else 0
        print(f"Answers completed in round {round_num}: {count:>6} ({percentage:>5.1f}%)")
    
    unfinished_count = len(input_questions) - total_completed
    unfinished_percentage = (unfinished_count / len(input_questions)) * 100 if len(input_questions) > 0 else 0
    print(f"Unfinished after {max_rounds} rounds: {unfinished_count:>6} ({unfinished_percentage:>5.1f}%)")
    print("-" * 50)
    print(f"Rank {rank} Total questions: {len(input_questions):>6}")
    print(f"Rank {rank} Total completed: {total_completed:>6}")
    print(f"Retrieval cache entries: {len(_retrieval_cache)}")
    print("="*50)
    
    return results

def main(input_file: str, output_file: str, max_rounds: int, batch_size: int = 16):
    """Main function optimization: Includes automatic sharding and final result merging"""
    
    # 1. Prepare file paths
    # Save the original output path for final merging
    final_output_file = output_file
    output_file_root, output_file_ext = os.path.splitext(output_file)
    # Generate a rank-specific output filename, e.g., result_rank0.jsonl
    rank_output_file = f"{output_file_root}_rank{rank}{output_file_ext}"

    print(f"Rank {rank}: Loading input data...")
    with open(input_file, 'r', encoding='utf-8') as fin:
        all_input_questions = [json.loads(line).get("question", "") for line in fin]
    
    # 2. Data parallel sharding
    # Each Rank only processes its assigned partition of data
    my_input_questions = all_input_questions[rank::world_size]
    
    # 3. Load checkpoint/processed data
    processed_questions = set()
    if os.path.exists(rank_output_file):
        print(f"Rank {rank}: Detected existing shard file, loading processed questions...")
        with open(rank_output_file, 'r', encoding='utf-8') as fout:
            for line in fout:
                try:
                    data = json.loads(line)
                    processed_questions.add(data.get("question", ""))
                except:
                    continue

    # 4. Filter pending data
    original_count = len(my_input_questions)
    questions_to_process = [q for q in my_input_questions if q not in processed_questions]
    
    print(f"Rank {rank}: Original assigned count: {original_count}")
    print(f"Rank {rank}: Processed count: {original_count - len(questions_to_process)}")
    print(f"Rank {rank}: Remaining pending count: {len(questions_to_process)}")
    
    # 5. Execute processing logic
    if questions_to_process:
        # Test Elasticsearch connection (only if there are tasks)
        try:
            test_result = batch_retrieve(["test query"])
            print(f"Rank {rank}: Elasticsearch connection test: {'Success' if test_result and test_result[0] is not None else 'Failed'}")
        except Exception as e:
            print(f"Rank {rank}: Elasticsearch connection test failed: {str(e)}")
        
        print(f"Rank {rank}: Start batch processing...")
        start_time = time.time()
        
        # Call the core processing function, results are directly written to rank_output_file
        process_questions_optimized(
            questions_to_process, rank_output_file, max_rounds, 
            batch_size=batch_size, processed_questions=processed_questions
        )
        
        end_time = time.time()
        print(f"\nRank {rank}: Processing completed, time taken: {end_time - start_time:.2f} seconds")
    else:
        print(f"Rank {rank}: All assigned questions have been processed!")

    # 6. Synchronize and wait for all processes
    print(f"Rank {rank}: Waiting for other processes to finish...")
    if dist.is_initialized():
        dist.barrier()  # Block until all GPU processes reach this line

    # 7. Merge results (executed only on Rank 0)
    if rank == 0:
        print("="*50)
        print("All processes completed, Rank 0 starting to merge files...")
        print("="*50)
        
        merged_count = 0
        try:
            with open(final_output_file, 'w', encoding='utf-8') as f_out:
                for r in range(world_size):
                    part_file = f"{output_file_root}_rank{r}{output_file_ext}"
                    if os.path.exists(part_file):
                        print(f"Merging shard: {part_file}")
                        with open(part_file, 'r', encoding='utf-8') as f_in:
                            for line in f_in:
                                f_out.write(line)
                                merged_count += 1
                        
                        os.remove(part_file) 
                    else:
                        print(f"Warning: Shard file not found {part_file} (Rank might have no data or failed)")
            
            print(f"Merge completed! Final file saved at: {final_output_file}")
            print(f"Total entries: {merged_count}")
        except Exception as e:
            print(f"Error occurred during merging: {str(e)}")

if __name__ == '__main__':
    main(
        input_file=args.input_file, 
        output_file=args.output_file, 
        max_rounds=args.max_round,
        batch_size=args.batch_size
    )