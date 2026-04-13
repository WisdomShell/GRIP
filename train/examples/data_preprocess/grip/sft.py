import os
import json
import argparse
import pandas as pd

from tqdm import tqdm
from datasets import Dataset, load_dataset
from verl.utils.hdfs_io import copy, makedirs

def make_prefix(example):
    instructionPrompt = """
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
    
    return instructionPrompt.format(
        question=example['Question'], intermediary=example['Intermediate_Answer'], reference=example['Retrieved_Context']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   default='/path/to/your/SFT_data.jsonl')
    parser.add_argument('--save_dir',    default='datasets/GRIPSFT')
    parser.add_argument('--data_source', default='GRIPSFT_1')  # Not necessary
    parser.add_argument('--train_prop',  default=0.98, type=float)
    parser.add_argument('--hdfs_dir',    default=None)
    args = parser.parse_args()

    assert args.train_prop > 0 and args.train_prop < 1, '`train_prop` should be in (0, 1)'

    df = pd.read_json(args.data_path, lines=True)
    df = df.sample(frac=1).reset_index(drop=True)
    raw_dataset = Dataset.from_pandas(df)
    print('>>> Raw Data Num:', len(raw_dataset))

    train_dataset = raw_dataset.select(range(int(len(raw_dataset) * args.train_prop)))
    test_dataset  = raw_dataset.select(range(int(len(raw_dataset) * args.train_prop), len(raw_dataset)))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example)
            data = {
                "data_source": args.data_source,
                "extra_info": {
                    "question": question,
                    "answer": example['Output'].replace("[Intermediary]", "[INTERMEDIARY]")
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    print("[INFO] Raw Length of Train Dataset:", len(train_dataset))
    print("[INFO] Raw Length of Test Dataset:", len(test_dataset))

    save_dir = args.save_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=save_dir, dst=hdfs_dir)