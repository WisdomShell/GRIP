<div align="center">

# 基于自触发信息规划的统一检索增强框架

<p>
  <a href="[./README.md](https://github.com/WisdomShell/GRIP)">English</a> | <strong>简体中文</strong>
</p>

<a href="https://arxiv.org/abs/2604.9999"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" /></a>
[![Project](https://img.shields.io/badge/Project-Homepage-2ea44f?logo=githubpages&logoColor=white)](https://wisdomshell.github.io/GRIP/)
[![Task](https://img.shields.io/badge/Task-Agentic%20RAG-purple.svg)](#overview)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/collections/WisdomShell/grip)
<a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/Venue-ACL%202026-blue" /></a>
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)


**ACL 2026 Main Conference**

<a href="https://deepblue666.github.io/">Bo Li</a>, Mingda Wang, Gexiang Fang, Shikun Zhang, Wei Ye

</div>

本仓库公开了 **GRIP**（**G**eneration-guided **R**etrieval with **I**nformation **P**lanning）的核心代码、训练流程、推理脚本和评测工具。GRIP 是一个统一的 Retrieval-as-Generation 框架，面向动态检索增强生成任务。

与把检索看作外部控制器决策的做法不同，GRIP 通过显式控制 token 将检索行为内化到 token-level decoding 中，例如 `[RETRIEVE]`、`[INTERMEDIARY]`、`[ANSWER]` 和 `[SOLVED]`。这样模型就可以在同一个自回归轨迹中决定 **何时检索**、**如何生成后续查询**、以及 **何时停止**。

---

## 🌟 概述

GRIP 的核心思想很简单：检索控制本身应当成为生成过程的一部分。

在 **Retrieval as Generation** 范式下，模型在解码过程中发出特殊控制 token 来调节检索行为。一个典型的 GRIP 轨迹可能会：

1. 当内部知识足够时直接回答，
2. 当信息不足时先生成中间响应，
3. 通过原始查询或改写后的新查询触发检索，
4. 在需要时继续多步检索，
5. 在问题解决后输出最终答案并结束。

本仓库提供以下实用代码：
- 结构化训练数据构造，
- 面向 token-controlled retrieval behavior 的监督微调，
- 基于 DAPO 的规则型 RL 微调，
- 本地多步推理，
- QA benchmark 评测，
- 基于 BM25 的 Wikipedia 索引构建。

---

## ✨ 亮点

- **统一的 token-level retrieval control**  
  检索时机、查询改写和终止控制都被表示为可训练的解码动作。

- **Self-Triggered Information Planning**  
  模型能够学习判断当前信息是否足够，并决定是否需要更多证据。

- **针对检索行为的结构化监督**  
  四种训练类型分别教会模型直接回答、触发检索、多跳规划和答案补全。

- **用 one-step decision optimization 学习 multi-step retrieval**  
  GRIP 用 one-step decision optimization 来学习多步检索行为，而不是依赖 long-horizon search-policy optimization，因此方法更简洁、更稳定，同时仍保留自适应检索深度和可控终止能力。

---

## 📦 公开内容

本仓库包含以下组成部分。

### 数据构造
- `data_generation/first.sh`  
  第一阶段数据构造流程的入口脚本。

- `data_generation/make_first_steps.py`  
  构造带有检索与可回答性信号的初始 A/B/C/D 结构化数据。

- `data_generation/use_gpt_for_data.py`  
  使用 GPT 对特定训练样本进行中间状态修正与查询改写。

- `data_generation/merge_dataset.py`  
  将多个结构化子集整合为最终的 SFT 与 RL 训练数据。

- `data_generation/index.py`  
  为 Wikipedia passage 语料建立 Elasticsearch 索引。

### 推理
- `inference/agent.py`  
  主多步 GRIP 推理脚本。

- `inference/inference.sh`  
  分布式推理示例脚本。

### 评测
- `eval/eval.py`  
  根据参考答案与预测结果计算 EM、F1、ROUGE 等指标。

- `eval/utils.py`  
  评测辅助函数。

### 训练
- `train/examples/data_preprocess/grip/sft.py`  
  将 GRIP 的 SFT 训练数据转换为 parquet 格式。

- `train/examples/data_preprocess/grip/rl.py`  
  将 GRIP 的 RL 训练数据转换为 parquet 格式。

- `train/examples/sft/run_sft_llama.sh`  
  LLaMA backbone 的 SFT 训练脚本。

- `train/recipe/dapo/dapo_4w_continue_rl_ep3_llama.sh`  
  基于 DAPO 的 RL 微调脚本。

- `train/scripts/merge.sh`  
  将 RL 训练后的分片 checkpoint 合并为 Hugging Face 格式。

### 环境
- `requirements.txt`  
  主仓库依赖。

- `train/requirements.txt`  
  训练框架依赖。

---

## 🗂️ 仓库结构

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

> 标准流程使用的是上面展示的 lowercase `train/` 目录。

---

## ⚙️ 安装

建议分别准备两个环境。

### 1. 主 GRIP 环境

```bash
conda create -n grip python=3.9
conda activate grip
pip install -r requirements.txt
```

### 2. 训练框架环境

```bash
cd train
pip install -e .
pip install -r requirements.txt
```

请尽量按照训练框架要求保持安装顺序一致。

---

## 🧾 数据与模型资源

当前开源版本对应的 Hugging Face 资源如下：

### Hugging Face 模型
- **GRIP-Llama-3-8B**: [WisdomShell/GRIP-Llama-3-8B](https://huggingface.co/WisdomShell/GRIP-Llama-3-8B)

### Hugging Face 数据集
- **GRIP_SFT_Data**: [WisdomShell/GRIP_SFT_Data](https://huggingface.co/datasets/WisdomShell/GRIP_SFT_Data)
- **GRIP_RL_Data**: [WisdomShell/GRIP_RL_Data](https://huggingface.co/datasets/WisdomShell/GRIP_RL_Data)

你也可以通过仓库顶部的 Hugging Face badge 访问合集页面：
- [WisdomShell/grip collection](https://huggingface.co/collections/WisdomShell/grip)

---

## 🔄 方法流程

实际运行流程如下：

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

## 🚀 快速开始

### 第一步：建立 Wikipedia 索引

下载 Wikipedia passage 数据：

```bash
mkdir wiki_data
cd wiki_data
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -d psgs_w100.tsv.gz
cd ..
```

启动 Elasticsearch 并建立索引：

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

### 第二步：准备原始 QA 训练数据

在生成 GRIP 训练数据前，请先把原始 QA 数据整理为如下 JSONL 格式：

```json
{
  "question": "Who wrote The Old Man and the Sea?",
  "answer": ["Ernest Hemingway"]
}
```

原始项目使用的训练数据包括：
- Natural Questions Open
- WebQuestions
- TriviaQA

### 第三步：构造 GRIP 结构化训练数据

先在 `data_generation/first.sh` 中修改以下变量：
- `MODEL_DIR`
- `INPUT_FILE`
- `BASE_OUTPUT_DIR`
- `ES_HOST`
- `ES_INDEX`

然后运行：

```bash
bash data_generation/first.sh
```

这一步会生成对应 GRIP 四种行为类型的初始结构化子集。

### 第四步：修正 GPT 生成子集

在 `data_generation/use_gpt_for_data.py` 中配置 OpenAI-compatible API，尤其是：
- `base_url`
- `api_key`
- `INPUT_FILE`

然后运行：

```bash
python data_generation/use_gpt_for_data.py
```

这一步会把特定样本改写为更规范的 `[INTERMEDIARY] ... [RETRIEVE] ...` 形式。

### 第五步：合并结构化子集

在 `data_generation/merge_dataset.py` 中修改：
- `input_dir`
- `output_jsonl_dir`

然后运行：

```bash
python data_generation/merge_dataset.py
```

脚本会输出：
- `SFT_data.jsonl`
- `RL_data.jsonl`

---

## 🏋️ 训练

### SFT 数据预处理

运行前，请更新：

- `train/examples/data_preprocess/grip/sft.py` 中的 `--data_path`

示例：

```bash
python train/examples/data_preprocess/grip/sft.py \
  --data_path /path/to/SFT_data.jsonl \
  --save_dir datasets/GRIPSFT
```

输出结果为：
- `datasets/GRIPSFT/train.parquet`
- `datasets/GRIPSFT/test.parquet`

### RL 数据预处理

运行前，请更新：
- `train/examples/data_preprocess/grip/rl.py` 中的 `--data_path`
- `--data_source`

示例：

```bash
python train/examples/data_preprocess/grip/rl.py \
  --data_path /path/to/RL_data.jsonl \
  --save_dir datasets/GRIPRL \
  --data_source GRIPRL
```

输出结果为：
- `datasets/GRIPRL/train.parquet`
- `datasets/GRIPRL/test.parquet`

### SFT 训练

使用：
- `train/examples/sft/run_sft_llama.sh`

需要重点修改：
- `NAME`
- `data.train_files`
- `data.val_files`
- `model.partial_pretrain`
- `trainer.default_local_dir`

然后运行：

```bash
cd train
bash examples/sft/run_sft_llama.sh
cd ..
```

### 基于 DAPO 的 RL 微调

使用：
- `train/recipe/dapo/dapo_4w_continue_rl_ep3_llama.sh`

需要重点修改：
- `MODEL_PATH`
- `CKPTS_DIR`
- `TRAIN_FILE`
- `TEST_FILE`

然后在训练环境中运行对应 RL 脚本。

### 合并模型分片

RL 训练完成后，可通过以下命令把分片权重合并为 Hugging Face 格式：

```bash
cd train
bash scripts/merge.sh
cd ..
```

---

## 🧠 推理

### 输入格式

测试文件建议使用如下格式：

```json
{
  "question": "Test query",
  "answer": ["One or more gold answers"]
}
```

### 运行 GRIP 推理

在 `inference/inference.sh` 中修改：
- `model_path`
- `input_file`
- `output_file`

然后运行：

```bash
bash inference/inference.sh
```

预测结果示例格式如下：

```json
{
  "question": "Test query",
  "prediction": ["step 1", "step 2", "final answer"]
}
```

---

## 📊 评测

使用以下命令评测预测结果：

```bash
python eval/eval.py \
  --references_path /path/to/test_dataset.jsonl \
  --predictions_path /path/to/prediction.jsonl
```

评测脚本支持：
- EM
- F1
- ROUGE

同时支持两种答案字段：
- `answer`
- `answer_and_def_correct_predictions`

---

## 🧩 GRIP 的四类训练行为

GRIP 将结构化监督组织为四种训练类型：

- **Type-α: Direct Answer**  
  模型直接回答并终止。

- **Type-β: Retrieval Needed**  
  模型输出部分中间状态并触发检索。

- **Type-γ: Multi-hop Planning**  
  模型迭代地产生新的中间状态与后续查询。

- **Type-θ: Answer Completion**  
  模型利用检索证据整合并输出最终答案。

这种设计使模型通过语言原生的 token 轨迹学习检索控制，而不是依赖外部控制器。

---

## 🛠️ 脚本说明

### `data_generation/make_first_steps.py`
主要功能：
- 执行第一阶段结构化数据生成，
- 构造不同检索行为对应的初始子集，
- 支持分布式生成，
- 与 Elasticsearch 检索交互。

### `data_generation/use_gpt_for_data.py`
主要功能：
- 用 GPT 修正子集 C，
- 重写中间状态与后续查询，
- 支持从中断位置继续执行。

### `data_generation/merge_dataset.py`
主要功能：
- 合并 A / B / C / D 子集，
- 生成 `SFT_data.jsonl`，
- 生成 `RL_data.jsonl`。

### `inference/agent.py`
主要功能：
- 运行本地 GRIP 推理，
- 支持多轮检索，
- 保存逐步预测轨迹。

### `eval/eval.py`
主要功能：
- 计算 EM、F1、ROUGE，
- 对齐预测结果与参考答案，
- 输出整体统计。

### `train/examples/data_preprocess/grip/sft.py`
主要功能：
- 将 SFT JSONL 转为 parquet 格式。

### `train/examples/data_preprocess/grip/rl.py`
主要功能：
- 将 RL JSONL 转为 parquet 格式，
- 准备 RL 训练所需 reward-model 字段。

### `train/examples/sft/run_sft_llama.sh`
主要功能：
- 启动 LLaMA 版本的 GRIP 监督微调。

### `train/recipe/dapo/dapo_4w_continue_rl_ep3_llama.sh`
主要功能：
- 启动基于 DAPO 的 GRIP RL 微调。

---

## ❓ 常见问题

### 1. Elasticsearch 没有正常启动
检索流程依赖 Elasticsearch 服务和已建立好的 Wikipedia 索引。

### 2. 脚本中的路径仍是占位符
大多数脚本中的路径都需要在运行前手动修改。

### 3. OpenAI-compatible API 没有配置
`data_generation/use_gpt_for_data.py` 需要有效的 API 凭据和接口地址。

### 4. 训练环境不完整
`train/` 下的训练框架有独立依赖，需要同时安装主仓库依赖和训练依赖。

### 5. 推理输出格式不匹配
评测脚本要求预测文件中的字段与 `eval/eval.py` 期望的格式一致。

### 6. 多阶段流程执行顺序错误
推荐顺序为：
1. 建立 Wikipedia 索引
2. 构造结构化数据
3. 合并 SFT / RL 数据
4. 预处理 parquet 数据
5. 执行 SFT
6. 执行 RL
7. 合并 checkpoint
8. 推理与评测

---

## 📖 引用

```bibtex

```


---
