<div align="center">

# Retrieval as Generation: A Unified Framework with Self-Triggered Information Planning

<p>
  <strong>English</strong> | <a href="./README_zh.md">简体中文</a>
</p>

<a href="https://arxiv.org/abs/2604.11407"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" /></a>
[![Project](https://img.shields.io/badge/Project-Homepage-2ea44f?logo=githubpages&logoColor=white)](https://wisdomshell.github.io/GRIP/)
[![Task](https://img.shields.io/badge/Task-Agentic%20RAG-purple.svg)](#overview)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/collections/WisdomShell/grip)
<a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/Venue-ACL%202026-blue" /></a>
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)


**ACL 2026 Main Conference**

<a href="https://deepblue666.github.io/">Bo Li</a>, Mingda Wang, Gexiang Fang, Shikun Zhang, Wei Ye

</div>

This repository releases the core code, training pipeline, inference scripts, and evaluation utilities for **GRIP** (**G**eneration-guided **R**etrieval with **I**nformation **P**lanning), a unified Retrieval-as-Generation framework for dynamic retrieval-augmented generation.

Instead of treating retrieval as an external controller decision, GRIP internalizes retrieval behavior into token-level decoding through explicit control tokens such as `[RETRIEVE]`, `[INTERMEDIARY]`, `[ANSWER]`, and `[SOLVED]`. This design enables the model to decide **when to retrieve**, **how to reformulate follow-up queries**, and **when to stop**, all within a single autoregressive trajectory.

---

## 🌟 Overview

GRIP is built around a simple idea: retrieval control should be part of generation itself.

Under the **Retrieval as Generation** paradigm, the model emits special control tokens during decoding to regulate retrieval behavior. A typical GRIP trajectory may:

1. answer directly when internal knowledge is sufficient,
2. emit an intermediate response when information is incomplete,
3. trigger retrieval with the original or a refined query,
4. continue multi-step retrieval when needed,
5. terminate with a final answer once the question is resolved.

This repository provides practical code for:

- structured training data construction,
- supervised fine-tuning for token-controlled retrieval behavior,
- rule-based RL fine-tuning with DAPO,
- local multi-step inference,
- benchmark evaluation on QA datasets,
- Wikipedia indexing for BM25-based retrieval.

---

## ✨ Highlights

- **Unified token-level retrieval control**  
  Retrieval timing, query reformulation, and stopping are all represented as trainable decoding actions.

- **Self-Triggered Information Planning**  
  The model learns to judge information sufficiency and decide whether more evidence is needed.

- **Structured supervision for retrieval behaviors**  
  Four training types teach the model direct answering, retrieval triggering, multi-hop planning, and answer completion.

- **One-step decision optimization for multi-step retrieval**  
  GRIP learns multi-step retrieval behavior through one-step decision optimization instead of long-horizon search-policy optimization, making it simpler and more stable while preserving adaptive depth and controllable stopping.

---

## 📦 What Is Released

This repository includes the following components.

### Data construction
- `data_generation/first.sh`  
  Entry script for the first-stage data construction pipeline.

- `data_generation/make_first_steps.py`  
  Builds the initial A/B/C/D-style structured data with retrieval and answerability signals.

- `data_generation/use_gpt_for_data.py`  
  Refines specific training cases with GPT-based query rewriting and intermediary correction.

- `data_generation/merge_dataset.py`  
  Merges structured subsets into final SFT and RL training data.

- `data_generation/index.py`  
  Builds the Elasticsearch index for the Wikipedia passage corpus.

### Inference
- `inference/agent.py`  
  Main multi-step GRIP inference script.

- `inference/inference.sh`  
  Example launch script for distributed inference.

### Evaluation
- `eval/eval.py`  
  Computes EM, F1, ROUGE, and other metrics from reference and prediction files.

- `eval/utils.py`  
  Evaluation utilities.

### Training
- `train/examples/data_preprocess/grip/sft.py`  
  Converts GRIP SFT training data into parquet format.

- `train/examples/data_preprocess/grip/rl.py`  
  Converts GRIP RL training data into parquet format.

- `train/examples/sft/run_sft_llama.sh`  
  SFT training script for the LLaMA backbone.

- `train/recipe/dapo/dapo_4w_continue_rl_ep3_llama.sh`  
  RL fine-tuning script based on DAPO.

- `train/scripts/merge.sh`  
  Merges sharded checkpoints into Hugging Face format after RL training.

### Environment
- `requirements.txt`  
  Main repository dependencies.

- `train/requirements.txt`  
  Additional dependencies for the training framework.

---

## 🗂️ Repository Structure

```text
.
├── README.md
├── README_zh.md
├── requirements.txt
├── data_generation/
│   ├── first.sh
│   ├── index.py
│   ├── make_first_steps.py
│   ├── merge_dataset.py
│   └── use_gpt_for_data.py
├── eval/
│   ├── eval.py
│   └── utils.py
├── inference/
│   ├── agent.py
│   └── inference.sh
└── train/
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── setup.py
    ├── examples/
    │   ├── data_preprocess/grip/
    │   │   ├── sft.py
    │   │   └── rl.py
    │   └── sft/
    │       └── run_sft_llama.sh
    ├── recipe/
    │   └── dapo/
    │       └── dapo_4w_continue_rl_ep3_llama.sh
    ├── scripts/
    │   └── merge.sh
    └── verl/
```

> The standard workflow uses the lowercase `train/` directory shown above.

---

## ⚙️ Installation

We recommend creating two environments.

### 1. Main GRIP environment

```bash
conda create -n grip python=3.9
conda activate grip
pip install -r requirements.txt
```

### 2. Training framework environment

```bash
cd train
pip install -e .
pip install -r requirements.txt
```

Please keep the installation order consistent with the training framework requirements.

---

## 🧾 Data and Model Resources

The current release includes the following Hugging Face resources:

### Hugging Face model
- **GRIP-Llama-3-8B**: [WisdomShell/GRIP-Llama-3-8B](https://huggingface.co/WisdomShell/GRIP-Llama-3-8B)

### Hugging Face datasets
- **GRIP_SFT_Data**: [WisdomShell/GRIP_SFT_Data](https://huggingface.co/datasets/WisdomShell/GRIP_SFT_Data)
- **GRIP_RL_Data**: [WisdomShell/GRIP_RL_Data](https://huggingface.co/datasets/WisdomShell/GRIP_RL_Data)

You can also access the released collection from the repository badge:
- [WisdomShell/grip collection](https://huggingface.co/collections/WisdomShell/grip)

---

## 🔄 Pipeline

The practical workflow is:

```text
Wikipedia passages
    -> data_generation/index.py
    -> Elasticsearch index

Raw QA training data
    -> data_generation/first.sh
    -> A / B / C / D structured subsets
    -> data_generation/use_gpt_for_data.py
    -> refined subset C
    -> data_generation/merge_dataset.py
    -> SFT_data.jsonl + RL_data.jsonl
    -> train/examples/data_preprocess/grip/sft.py or rl.py
    -> parquet datasets
    -> SFT training
    -> RL training with DAPO
    -> train/scripts/merge.sh
    -> GRIP checkpoint
    -> inference/agent.py
    -> eval/eval.py
```

---

## 🚀 Quick Start

### Step 1. Build the Wikipedia index

Download the Wikipedia passage dump:

```bash
mkdir wiki_data
cd wiki_data
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -d psgs_w100.tsv.gz
cd ..
```

Set up Elasticsearch and build the index:

```bash
mkdir ret
cd ret
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz
cd elasticsearch-7.17.9
nohup bin/elasticsearch &
cd ../..
python data_generation/index.py --data_path /path/to/psgs_w100.tsv --index_name wiki
```

### Step 2. Prepare the raw QA training data

Before generating GRIP training data, combine the raw QA training sets into JSONL format:

```json
{
  "question": "Who wrote The Old Man and the Sea?",
  "answer": ["Ernest Hemingway"]
}
```

The original project uses training data from:
- Natural Questions Open
- WebQuestions
- TriviaQA

### Step 3. Construct structured GRIP training data

Update these variables in `data_generation/first.sh`:
- `MODEL_DIR`
- `INPUT_FILE`
- `BASE_OUTPUT_DIR`
- `ES_HOST`
- `ES_INDEX`

Then run:

```bash
bash data_generation/first.sh
```

This stage creates the initial structured subsets for the four GRIP behavior types.

### Step 4. Refine the GPT-based subset

Configure your OpenAI-compatible API settings in `data_generation/use_gpt_for_data.py`, especially:
- `base_url`
- `api_key`
- `INPUT_FILE`

Then run:

```bash
python data_generation/use_gpt_for_data.py
```

This stage rewrites specific training cases into refined `[INTERMEDIARY] ... [RETRIEVE] ...` patterns.

### Step 5. Merge the structured subsets

Update the paths in `data_generation/merge_dataset.py`:
- `input_dir`
- `output_jsonl_dir`

Then run:

```bash
python data_generation/merge_dataset.py
```

The script produces:
- `SFT_data.jsonl`
- `RL_data.jsonl`

---

## 🏋️ Training

### SFT data preprocessing

Before running, update `--data_path` in:

- `train/examples/data_preprocess/grip/sft.py`

Example:

```bash
python train/examples/data_preprocess/grip/sft.py \
  --data_path /path/to/SFT_data.jsonl \
  --save_dir datasets/GRIPSFT
```

This produces:
- `datasets/GRIPSFT/train.parquet`
- `datasets/GRIPSFT/test.parquet`

### RL data preprocessing

Before running, update `--data_path` and `--data_source` in:

- `train/examples/data_preprocess/grip/rl.py`

Example:

```bash
python train/examples/data_preprocess/grip/rl.py \
  --data_path /path/to/RL_data.jsonl \
  --save_dir datasets/GRIPRL \
  --data_source GRIPRL
```

This produces:
- `datasets/GRIPRL/train.parquet`
- `datasets/GRIPRL/test.parquet`

### SFT training

Use:

- `train/examples/sft/run_sft_llama.sh`

Key fields to update include:
- `NAME`
- `data.train_files`
- `data.val_files`
- `model.partial_pretrain`
- `trainer.default_local_dir`

Then run:

```bash
cd train
bash examples/sft/run_sft_llama.sh
cd ..
```

### RL fine-tuning with DAPO

Use:

- `train/recipe/dapo/dapo_4w_continue_rl_ep3_llama.sh`

Key fields to update include:
- `MODEL_PATH`
- `CKPTS_DIR`
- `TRAIN_FILE`
- `TEST_FILE`

Then run the RL script in your training environment.

### Merge model shards

After RL training, convert the saved shards into Hugging Face format:

```bash
cd train
bash scripts/merge.sh
cd ..
```

---

## 🧠 Inference

### Input format

The test file should follow the format:

```json
{
  "question": "Test query",
  "answer": ["One or more gold answers"]
}
```

### Run GRIP inference

Update the paths in `inference/inference.sh`:
- `model_path`
- `input_file`
- `output_file`

Then run:

```bash
bash inference/inference.sh
```

The prediction file will contain records like:

```json
{
  "question": "Test query",
  "prediction": ["step 1", "step 2", "final answer"]
}
```

---

## 📊 Evaluation

Evaluate predictions with:

```bash
python eval/eval.py \
  --references_path /path/to/test_dataset.jsonl \
  --predictions_path /path/to/prediction.jsonl
```

The evaluation script supports:
- EM
- F1
- ROUGE

It also handles answer fields with either: 
- `answer`
- `answer_and_def_correct_predictions`

---

## 🧩 GRIP Training Behaviors

GRIP organizes structured supervision into four training types:

- **Type-α: Direct Answer**  
  The model answers directly and terminates.

- **Type-β: Retrieval Needed**  
  The model emits a partial response and triggers retrieval.

- **Type-γ: Multi-hop Planning**  
  The model iteratively generates new intermediary states and follow-up queries.

- **Type-θ: Answer Completion**  
  The model uses retrieved evidence to synthesize and finalize the answer.

This design teaches the model retrieval control through language-native token trajectories rather than external controllers.

---

## 🛠️ Script Notes

### `data_generation/make_first_steps.py`
Main functionality:
- runs first-stage structured data generation,
- builds initial behavior-specific subsets,
- supports distributed generation,
- interacts with Elasticsearch retrieval.

### `data_generation/use_gpt_for_data.py`
Main functionality:
- refines subset C with GPT-based query generation,
- rewrites intermediary answers,
- resumes from partial progress if interrupted.

### `data_generation/merge_dataset.py`
Main functionality:
- merges A / B / C / D subsets,
- builds `SFT_data.jsonl`,
- builds `RL_data.jsonl`.

### `inference/agent.py`
Main functionality:
- runs local GRIP inference,
- supports multi-round retrieval,
- saves step-wise predictions.

### `eval/eval.py`
Main functionality:
- computes EM, F1, and ROUGE,
- matches predictions with references,
- reports summary statistics.

### `train/examples/data_preprocess/grip/sft.py`
Main functionality:
- converts SFT JSONL into parquet format.

### `train/examples/data_preprocess/grip/rl.py`
Main functionality:
- converts RL JSONL into parquet format,
- prepares reward-model fields for RL training.

### `train/examples/sft/run_sft_llama.sh`
Main functionality:
- launches GRIP supervised fine-tuning for LLaMA.

### `train/recipe/dapo/dapo_4w_continue_rl_ep3_llama.sh`
Main functionality:
- launches GRIP RL fine-tuning with DAPO.

---

## ❓ Common Issues

### 1. Elasticsearch is not running
The retrieval pipeline depends on a working Elasticsearch service and a built Wikipedia index.

### 2. Paths are still placeholders
Most scripts contain placeholder paths. Update all paths before running.

### 3. OpenAI-compatible API is not configured
`data_generation/use_gpt_for_data.py` requires valid API credentials and endpoint settings.

### 4. Training environment is incomplete
The training framework under `train/` has its own dependency setup. Install both the main repository dependencies and the training dependencies.

### 5. Inference output format is not aligned
Evaluation expects the prediction file to match the question field and prediction format used by `eval/eval.py`.

### 6. Multi-stage pipeline is run out of order
The recommended order is:
1. index Wikipedia
2. construct structured data
3. merge SFT / RL datasets
4. preprocess parquet data
5. run SFT
6. run RL
7. merge checkpoints
8. run inference and evaluation

---

## 🧭 Related RAG Projects

This repository is part of our broader research line on **controllable and adaptive Retrieval-Augmented Generation (RAG)**.

- **GRIP** [ACL 2026 Main Conference]: [Retrieval as Generation: A Unified Framework with Self-Triggered Information Planning](https://github.com/WisdomShell/GRIP)  
   A **training-based dynamic RAG** framework that internalizes retrieval control into token-level decoding.  
  

- **ETC** [AAAI 2026 Oral Paper]: [Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG](https://github.com/WisdomShell/ETC)  
   A **training-free dynamic RAG** method that improves retrieval timing by modeling entropy trends during decoding.  
  

- **SCD** [AAAI 2026 Oral Paper]: [Language Drift in Multilingual Retrieval-Augmented Generation](https://github.com/WisdomShell/SCD)  
   A **training-free multilingual RAG** method that mitigates language drift through decoding-time control.  

Together, these projects cover three complementary directions in RAG:
**training-based retrieval planning, training-free retrieval timing, and decoding-time control for multilingual generation**.

## 📖 Citation

```bibtex
@misc{li2026retrievalgenerationunifiedframework,
      title={Retrieval as Generation: A Unified Framework with Self-Triggered Information Planning}, 
      author={Bo Li and Mingda Wang and Gexiang Fang and Shikun Zhang and Wei Ye},
      year={2026},
      eprint={2604.11407},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.11407}, 
}
```

---
