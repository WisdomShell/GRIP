import json
import os
from datasets import Dataset

input_dir = "/path/to/your/data" #A,B,C,D jsonl file path
output_jsonl_dir = "/path/to/your/train_data"

files_to_merge = ['A.jsonl', 'B.jsonl', 'C.jsonl', 'D.jsonl']
sft_data = []
rl_data = []

for filename in files_to_merge:
    file_path = os.path.join(input_dir, filename)
    if os.path.exists(file_path):
        print(f"read {filename}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        
        file_sft = [json.loads(line) for line in lines[:10000]]
        file_rl  = [json.loads(line) for line in lines[-1250:]]
        
        sft_data.extend(file_sft)
        rl_data.extend(file_rl)
        
        print(f"  {filename}: SFT {len(file_sft)} 条，RL {len(file_rl)} 条")
    else:
        print(f"File not found: {filename}")

print(f"\n SFT data number: {len(sft_data)} ，RL data number: {len(rl_data)} ")

# SFT_data.jsonl
sft_jsonl_path = os.path.join(output_jsonl_dir, "SFT_data.jsonl")
with open(sft_jsonl_path, 'w', encoding='utf-8') as f:
    for item in sft_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"SFT jsonl saved: {sft_jsonl_path}")

#RL_data.jsonl
rl_jsonl_path = os.path.join(output_jsonl_dir, "RL_data.jsonl")
with open(rl_jsonl_path, 'w', encoding='utf-8') as f:
    for item in rl_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"RL jsonl saved: {rl_jsonl_path}")
