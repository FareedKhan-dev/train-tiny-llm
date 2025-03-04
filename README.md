<!-- omit in toc -->
# Tiny LLM from Scratch

[Live Demo](https://huggingface.co/spaces/FareedKhan/30M-LLM-Chat) | [Step-by-Step Guide](https://medium.com/@fareedkhandev/building-a-perfect-million-parameter-llm-from-scratch-in-python-3b16e26b4139) | [Model Weights](https://huggingface.co/spaces/FareedKhan/30M-LLM-Chat/tree/main/30M-SFT-LLM)

This repository contains the code and instructions for building and training a small, yet surprisingly capable, 30-million-parameter language model (LLM) from scratch in PyTorch. 

**Key Features:**

*   **Custom Tokenizer:** Train a Byte-Pair Encoding (BPE) tokenizer specifically for this project.
*   **Transformer from Scratch:** Implement a Transformer model, including RMSNorm, Rotary Positional Embeddings (RoPE), Multi-Head Attention, and FeedForward Network.
*   **Pre-training:** Train the model on a large text corpus to learn general language understanding.
*   **Supervised Fine-Tuning (SFT):** Fine-tune the pre-trained model on a conversational dataset to improve its ability to follow instructions and engage in dialogue.
*   **Streamlit Web App:** Interact with the fine-tuned model through a simple web interface.
*   **Distributed Training Support:** Includes support for distributed training using `torchrun`.
*   **Mixed Precision Training:** Leverages `torch.cuda.amp.autocast` for faster training with reduced memory usage.
  
The preprocessed data for pretraining and supervised fine-tuning has been uploaded to the Hugging Face Datasets Hub. It can be downloaded from the following links:  

- [Pretraining processed data](https://huggingface.co/datasets/FareedKhan/pretrain_data) 
- [SFT processed data](https://huggingface.co/datasets/FareedKhan/sft_data)
  
<!-- omit in toc -->
## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Installation](#2-installation)
- [3. Project Structure](#3-project-structure)
- [4. Tokenizer Training](#4-tokenizer-training)
  - [4.1. Dataset Format (C4)](#41-dataset-format-c4)
  - [4.2. Training the Tokenizer](#42-training-the-tokenizer)
  - [4.3. Tokenizer Evaluation](#43-tokenizer-evaluation)
- [5. Pre-training](#5-pre-training)
  - [5.1. Dataset Format (DeepCtrl SFT Data)](#51-dataset-format-deepctrl-sft-data)
  - [5.2. Pre-training Execution](#52-pre-training-execution)
- [6. Supervised Fine-Tuning (SFT)](#6-supervised-fine-tuning-sft)
  - [6.1. Dataset Format (GooAQ)](#61-dataset-format-gooaq)
  - [6.2. SFT Execution](#62-sft-execution)
- [7. Web App Deployment and Interaction](#7-web-app-deployment-and-interaction)
- [8. Distributed Training](#8-distributed-training)

## 1. Prerequisites

*   Python 3.8+
*   PyTorch 2.0+ (strongly recommended for Flash Attention)
*   CUDA-enabled GPU (recommended for pre-training and SFT)
*   Sufficient RAM (at least 23GB for loading the pre-training dataset, recommend using Kaggle or Colab)
*   `transformers`, `datasets`, `streamlit`, `wandb`, and other libraries (see `requirement.txt`)

## 2. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/FareedKhan-dev/train-tiny-llm.git
    cd train-tiny-llm
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate    # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirement.txt
    ```

## 3. Project Structure

```
.
├── pretrain.py              # Pre-training script
├── requirement.txt          # Project dependencies
├── train_sft.py             # Supervised fine-tuning script
├── train_tokenizer.py       # Tokenizer training script
├── web_app.py               # Streamlit web application
└── transformer/             # Directory containing core components
    ├── dataset.py           # Dataset classes (PretrainDataset, SFTDataset)
    ├── LMConfig.py          # Language model configuration class (LMConfig)
    └── model.py             # Transformer model definition (TransformerLM)
```

## 4. Tokenizer Training

### 4.1. Dataset Format (C4)

The tokenizer training data should be a JSONL file where each line is a JSON object containing a "text" field with the text content.  This is the format produced by extracting data from the `allenai/c4` dataset.

You can use this script to extract a subset of the `allenai/c4` dataset.

```python
from datasets import load_dataset
from tqdm import tqdm
import json

# Load the dataset in streaming mode to efficiently process large datasets
# without loading everything into memory.
ds = load_dataset("allenai/c4", "en", streaming=True)

# Number of rows to retrieve and save
num_rows_to_get = 2000000

# Define the output file name
output_file = "training_data.jsonl"

# Open the file in write mode and process the dataset in a streaming fashion
with open(output_file, "w", encoding="utf-8") as f:
    # Initialize progress bar to track saving progress
    with tqdm(total=num_rows_to_get, desc="Saving to JSONL", unit="entry") as pbar:
        # Iterate over the dataset stream and write each entry directly to the file
        for i, row in enumerate(iter(ds["train"])):
            if i >= num_rows_to_get:  # Stop when the desired number of rows is reached
                break
            json.dump({"text": row["text"]}, f, ensure_ascii=False)  # Convert row to JSON format
            f.write("\n")  # Write a newline to separate JSON objects in JSONL format
            pbar.update(1)  # Update progress bar after processing each row
```


**Sample `training_data.jsonl`:**

```json
{"text": "This is the first example sentence."}
{"text": "Another sentence for tokenizer training."}
{"text": "The quick brown fox jumps over the lazy dog."}
...
```
You can obtain a suitable dataset by following the instructions in the Medium article to extract a subset of the `allenai/c4` dataset, or use any text data formatted as shown above.

### 4.2. Training the Tokenizer

Use the `train_tokenizer.py` script:

```bash
python train_tokenizer.py --dataset_path training_data.jsonl --output_dir custom_tokenizer --vocab_size 6400 --evaluate
```

*   `--dataset_path`: Path to your `training_data.jsonl` file.
*   `--output_dir`:  Where to save the trained tokenizer.
*   `--vocab_size`:  Desired vocabulary size (6400).
*   `--evaluate`:  Runs a short evaluation.

This script trains a BPE tokenizer, defines special tokens (`<unk>`, `<s>`, `</s>`), and saves the tokenizer files (including a crucial `tokenizer_config.json` with the chat template).

### 4.3. Tokenizer Evaluation

The `--evaluate` flag runs the `evaluate_tokenizer` function, demonstrating:

*   Loading the tokenizer with `AutoTokenizer.from_pretrained`.
*   Formatting a conversation with `apply_chat_template`.
*   Encoding and decoding text.

## 5. Pre-training

### 5.1. Dataset Format (DeepCtrl SFT Data)

The pre-training data should be a JSONL file where each line is a JSON object with a "text" field. This "text" field should contain the concatenation of the "input" and "output" fields from the DeepCtrl SFT Data, separated by a newline character.  Crucially, the combined length of the input and output should be less than 512 characters.

**Sample `pretrain_data.jsonl`:**

```json
{"text": "What is the capital of Australia?\nCanberra"}
{"text": "Who painted the Mona Lisa?\nLeonardo da Vinci"}
{"text": "Explain the theory of relativity.\nRelativity, in simple terms, describes..."}
...
```
You can create a suitable dataset from the English portion of the [DeepCtrl SFT Data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/summary), filtering for entries where the combined length is less than 512.

Preprocessed versions of the DeepCtrl SFT Data available from the following link: [Pretraining processed data](https://huggingface.co/datasets/FareedKhan/pretrain_data)

### 5.2. Pre-training Execution

Use `pretrain.py`. Example command:

```bash
python pretrain.py \
  --data_path pretrain_data.jsonl \
  --tokenizer_path custom_tokenizer \
  --out_dir out \
  --epochs 1 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --dim 512 \
  --n_layers 8 \
  --max_seq_len 512 \
  --accumulation_steps 8 \
  --log_interval 100 \
  --save_interval 100 \
  --use_wandb \
  --ddp
```

**Key Arguments (See table below for more details):**

*   `--data_path`: Path to `pretrain_data.jsonl`.
*   `--tokenizer_path`: Path to `custom_tokenizer`.
*   `--out_dir`: Output directory.
*   `--epochs`: Number of epochs.
*   `--batch_size`: Batch size.
*   `--learning_rate`: Initial learning rate.
*   `--dim`:  Embedding dimension.
*   `--n_layers`: Number of Transformer layers.
*   `--max_seq_len`: Maximum sequence length.
*    `--device`: Specify the device (`cuda:0`, `cpu`, etc.).

This script trains the Transformer, using `PretrainDataset`, cosine annealing, mixed precision, gradient accumulation/clipping, checkpointing, and wandb logging (if enabled).

## 6. Supervised Fine-Tuning (SFT)

### 6.1. Dataset Format (GooAQ)

The SFT data should be a JSONL file where each line is a JSON object with a "conversations" field.  The "conversations" field contains a list of turns, each with "role" ("user" or "assistant") and "content" fields.  This format is obtained by converting the [GooAQ dataset](https://github.com/allenai/gooaq).

Preprocessed versions of the GooAQ dataset are available from the following link: [SFT processed data](https://huggingface.co/datasets/FareedKhan/sft_data)

**Sample `sft_data.jsonl`:**

```json
{"conversations": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris"}]}
{"conversations": [{"role": "user", "content": "Who wrote Hamlet?"}, {"role": "assistant", "content": "William Shakespeare"}]}
{"conversations": [{"role": "user", "content": "Explain photosynthesis."}, {"role": "assistant", "content": "Photosynthesis is the process..."}]}
...
```

### 6.2. SFT Execution

Use `train_sft.py`:

```bash
python train_sft.py \
  --data_path sft_data.jsonl \
  --out_dir out \
  --epochs 1 \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --dim 512 \
  --n_layers 8 \
  --max_seq_len 512\
  --log_interval 100 \
  --save_interval 100 \
  --use_wandb
  --ddp
```

**Key Differences from Pre-training:**

*   `--data_path`: Points to `sft_data.jsonl`.
*   `--learning_rate`:  Lower learning rate (e.g., `5e-5`).
*    `--out_dir`: should have your pretrained model.

This script fine-tunes the pre-trained model, using `SFTDataset` (which uses the tokenizer's chat template and generates a loss mask for the assistant's responses), loading pre-trained weights, and similar training loop features as `pretrain.py`.

## 7. Web App Deployment and Interaction

Run the Streamlit web app:

```bash
streamlit run web_app.py
```

This app provides a chat interface.  Key features:

*   **Loading:** Loads the model and tokenizer (using `@st.cache_resource`).
*   **Formatting:** Uses `apply_chat_template` to format history.
*   **Generation:** Uses the model's `generate` method (with temperature, top-p, max_new_tokens).
*   **Display:** Shows conversation history.
*   **Clearing:** Clears chat history.
* **Parameter Adjustment:** Sidebar sliders to modify generation parameters. You can type messages and receive responses, adjusting parameters like temperature and top-p in the sidebar.  The chat history is correctly formatted using the tokenizer's chat template.

## 8. Distributed Training

Use the `--ddp` flag and `torchrun`:

```bash
torchrun --nproc_per_node=<num_gpus> pretrain.py ...  # Pre-training
torchrun --nproc_per_node=<num_gpus> train_sft.py ...   # SFT
```

Replace `<num_gpus>` with the number of GPUs.  Example (2 GPUs):

```bash
torchrun --nproc_per_node=2 pretrain.py --ddp --data_path pretrain_data.jsonl --tokenizer_path custom_tokenizer --out_dir out --epochs 1
```

The scripts handle distributed process group initialization, `DistributedSampler`, `DistributedDataParallel`, and device placement.

<hr>

Contributions are welcome!