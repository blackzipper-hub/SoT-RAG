# SoT (Socratic Questioning) Fine-tuning with LoRA

This directory contains a simplified fine-tuning script for training language models on SoT (Socratic Questioning) data using LoRA (Low-Rank Adaptation).

## Overview

The `sft_lora.py` script is designed to fine-tune language models on SoT training data, where each example contains:
- `instruction`: The main question or topic
- `output`: A list of Socratic questions that help explore the topic

## Data Format

The training data should be in JSONL format with the following structure:

```json
{"instruction": "What are the causes and consequences of deforestation?", "output": ["How does deforestation impact local ecosystems and biodiversity?", "What policy instruments can reduce the harms of deforestation?", "What mitigation or adaptation strategies exist for deforestation?"]}
```

## Usage

### Basic Usage

```bash
python sft_lora.py \
    --train_file /path/to/your/sot_training_data.jsonl \
    --model_name_or_path microsoft/DialoGPT-medium \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5
```

### Required Arguments

- `--train_file`: Path to your SoT training data file (JSONL format)
- `--model_name_or_path`: Hugging Face model identifier or local path
- `--output_dir`: Directory to save the fine-tuned model

### Optional Arguments

- `--use_lora`: Enable LoRA fine-tuning (default: True)
- `--lora_rank`: LoRA rank (default: 64)
- `--lora_alpha`: LoRA alpha parameter (default: 16)
- `--lora_dropout`: LoRA dropout rate (default: 0.1)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--per_device_train_batch_size`: Batch size per device (default: 8)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--warmup_ratio`: Warmup ratio (default: 0)
- `--logging_steps`: Log every N steps (default: 10)
- `--checkpointing_steps`: Save checkpoint every N steps or 'epoch' (default: 'epoch')
- `--seed`: Random seed (default: 42)

### Example with Custom Parameters

```bash
python sft_lora.py \
    --train_file ../data/sot_training_data/generated_30000_full.jsonl \
    --model_name_or_path microsoft/DialoGPT-medium \
    --output_dir ./sot_model_output \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --logging_steps 20 \
    --checkpointing_steps epoch \
    --seed 42
```

## Features

- **LoRA Support**: Efficient fine-tuning using Low-Rank Adaptation
- **SoT Format**: Automatically handles SoT training data format
- **Flash Attention**: Optional flash attention support for faster training
- **Checkpointing**: Save model checkpoints during training
- **Multi-GPU**: Support for distributed training with Accelerate

## Output

The script will save:
- Fine-tuned model weights in the specified output directory
- Tokenizer files
- Training logs and metrics

## Requirements

- PyTorch
- Transformers
- Accelerate
- PEFT (for LoRA)
- Datasets

## Notes

- The script automatically detects SoT format data and applies appropriate encoding
- LoRA is enabled by default for efficient fine-tuning
- The model will be saved in Hugging Face format for easy loading
