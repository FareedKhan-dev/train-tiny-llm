import os
import json
import random
import argparse
from transformers import AutoTokenizer
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    decoders
)
from typing import Generator

# Set a fixed random seed for reproducibility
random.seed(42)

def load_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """
    Generator function to read and yield text data from a JSONL file.
    
    :param file_path: Path to the JSONL file containing text data.
    :return: Yields text content from each line in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            yield data['text']

def train_custom_tokenizer(dataset_path: str, tokenizer_save_dir: str, vocab_size: int):
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer on a given dataset.
    
    :param dataset_path: Path to the dataset (JSONL file) to train the tokenizer.
    :param tokenizer_save_dir: Directory where the trained tokenizer should be saved.
    :param vocab_size: Vocabulary size for the tokenizer.
    """
    # Initialize a new BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Define special tokens
    special_tokens = ["<unk>", "<s>", "</s>"]

    # Configure tokenizer training settings
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Load dataset texts
    texts = load_texts_from_jsonl(dataset_path)

    # Train the tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    # Ensure special tokens are correctly assigned their expected indices
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # Create directory if it doesn't exist
    os.makedirs(tokenizer_save_dir, exist_ok=True)

    # Save tokenizer model and configuration
    tokenizer.save(os.path.join(tokenizer_save_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_save_dir)

    # Manually define tokenizer configuration
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
        "model_max_length": 32768,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
        "additional_special_tokens": [],
        "spaces_between_special_tokens": False,
        "sp_model_kwargs": {},
        "added_tokens_decoder": {
            "0": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "1": {"content": "<s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "2": {"content": "</s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True}
        },
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\nYou are a helpful AI assistant.</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # Save tokenizer configuration to a JSON file
    with open(os.path.join(tokenizer_save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved successfully.")

def evaluate_tokenizer(tokenizer_path: str):
    """
    Loads the trained tokenizer and evaluates it on sample English conversations.
    
    :param tokenizer_path: Path to the directory containing the trained tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    sample_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Where are you from?"},
        {"role": "assistant", "content": "I am from the cloud."}
    ]

    # Generate chat template using tokenizer
    formatted_prompt = tokenizer.apply_chat_template(
        sample_conversation,
        tokenize=False
    )
    print("Formatted chat prompt:\n", formatted_prompt)

    # Get actual vocabulary size
    vocab_size = len(tokenizer)
    print('Tokenizer vocabulary size:', vocab_size)

    # Encode the formatted prompt
    encoded_inputs = tokenizer(formatted_prompt)
    print('Encoded input length:', len(encoded_inputs['input_ids']))

    # Decode the tokens back to text
    decoded_text = tokenizer.decode(encoded_inputs['input_ids'], skip_special_tokens=False)
    print('Decoded text matches original:', decoded_text == formatted_prompt)

def main():
    """
    Parses command-line arguments and executes tokenizer training and optionally evaluation.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a custom tokenizer.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training JSONL file.")
    parser.add_argument("--output_dir", default="custom_tokenizer", help="Directory to save the trained tokenizer.")
    parser.add_argument("--vocab_size", type=int, default=6400, help="Vocabulary size for the tokenizer (default: 6400).")
    parser.add_argument("--evaluate", action='store_true', help="Flag to evaluate the tokenizer after training.")

    args = parser.parse_args()

    train_custom_tokenizer(args.dataset_path, args.output_dir, args.vocab_size)
    
    if args.evaluate:
        evaluate_tokenizer(args.output_dir)

if __name__ == '__main__':
    main()
