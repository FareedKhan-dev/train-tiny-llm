import json

from torch.utils.data import Dataset
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism (often a good idea)


class PretrainDataset(Dataset):
    """Dataset for pretraining."""

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Initializes the PretrainDataset.

        Args:
            data_path (str): Path to the JSONL data file.
            tokenizer: The tokenizer to use.
            max_length (int): Maximum sequence length.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        Loads data from a JSONL file.

        Args:
            path (str): Path to the JSONL file.

        Returns:
            list: A list of samples loaded from the file.
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # enumerate starts at 1 for line_num
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input (X), target (Y), and loss mask.
        """
        sample = self.samples[index]

        # Construct the input text, including BOS and EOS tokens.
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',  # Pad to max_length
            truncation=True,  # Truncate to max_length
            return_tensors='pt'  # Return PyTorch tensors
        )
        input_ids = encoding.input_ids.squeeze()  # Remove extra dimension
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # Create loss mask (ignore padding)

        # Create input (X) and target (Y) tensors, shifting by one position.
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # Shift loss mask as well
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT)."""

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        Initializes the SFTDataset.

        Args:
            jsonl_path (str): Path to the JSONL data file.
            tokenizer: The tokenizer to use.
            max_length (int): Maximum sequence length.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # Get special token IDs for loss mask generation.  These are now instance variables.
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def load_data(self, path):
        """
        Loads data from a JSONL file.

        Args:
            path (str): Path to the JSONL file.

        Returns:
            list: A list of samples loaded from the file.
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        Creates a chat prompt in ChatML format.

        Args:
            conversations (list): A list of conversation turns.

        Returns:
            str: The formatted chat prompt string.
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'  # Alternate user/assistant roles
            messages.append({"role": role, "content": turn['content']})
        # Use the tokenizer's apply_chat_template method for correct formatting.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Don't tokenize here, we'll do it later.
            add_generation_prompt=False  # No generation prompt needed for SFT.
        )

    def _generate_loss_mask(self, input_ids):
        """
        Generates a dynamic loss mask.  Only assistant turns are used for loss calculation.

        Args:
            input_ids (list): List of input token IDs.

        Returns:
            list: A list representing the loss mask (0 or 1 for each token).
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # Find the start of an assistant turn (indicated by bos_id).
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # Find the end of the assistant turn (indicated by eos_id).
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # Set loss mask to 1 for tokens within the assistant turn (including eos).
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids) # skip to end or break
            else:
                i += 1  # Move to the next token
        return loss_mask

    def __getitem__(self, index):
        """
        Retrieves a sample and prepares it for SFT.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input (X), target (Y), and loss mask.
        """
        sample = self.samples[index]
        # Create the chat prompt.
        prompt = self._create_chat_prompt(sample['conversations'])
        # Tokenize the prompt, truncate, and pad.
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # Generate the dynamic loss mask.
        loss_mask = self._generate_loss_mask(input_ids)

        # Create input (X) and target (Y) tensors, shifted by one position.
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # Shift loss mask

        return X, Y, loss_mask


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization (DPO)."""

    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        Initializes the DPODataset.

        Args:
            file_path (str): Path to the JSONL data file.
            tokenizer: The tokenizer to use.
            max_length (int): Maximum sequence length.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Use pad_token_id if available, otherwise 0.
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # bos and eos are now instance variables
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a sample and prepares it for DPO.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the chosen and rejected inputs, targets, and loss masks.
        """
        item = self.data[index]
        chosen = item['chosen']  # List of {role, content} dictionaries
        rejected = item['rejected']  # List of {role, content} dictionaries

        # Create prompts using apply_chat_template.
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # Tokenize, truncate, and pad both chosen and rejected prompts.
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        # Create input (x), target (y), and mask tensors for chosen and rejected.
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        """
        Generates a dynamic loss mask, similar to SFTDataset.  Only assistant turns are used.

        Args:
            input_ids (list): List of input token IDs.

        Returns:
            list: A list representing the loss mask (0 or 1 for each token).
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
             # Find the start of an assistant turn (indicated by bos_id).
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # Find the end of the assistant turn (indicated by eos_id)
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # Mark tokens within the assistant turn with 1 in the loss mask
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)  # continue
            else:
                i += 1 # Move to the next token
        return loss_mask