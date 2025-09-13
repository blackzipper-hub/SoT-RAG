#!/usr/bin/env python
# coding=utf-8

import json
import torch
from transformers import AutoTokenizer
from sft_lora import encode_with_sot_format, PROMPT_DICT

def test_sot_encoding():
    """Test the SoT encoding function with sample data"""
    
    # Load a sample from the training data
    with open('/home/sunkai/software_engine/SoT_RAG/data/sot_training_data/generated_30000_full.jsonl', 'r') as f:
        sample_line = f.readline()
        sample_data = json.loads(sample_line)
    
    print("Sample data:")
    print(f"Instruction: {sample_data['instruction']}")
    print(f"Output: {sample_data['output']}")
    print()
    
    # Initialize tokenizer (using a simple one for testing)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test the encoding function
    try:
        encoded = encode_with_sot_format(sample_data, tokenizer, max_seq_length=512)
        
        print("Encoding successful!")
        print(f"Input IDs shape: {encoded['input_ids'].shape}")
        print(f"Labels shape: {encoded['labels'].shape}")
        print(f"Attention mask shape: {encoded['attention_mask'].shape}")
        
        # Decode to see the formatted text
        input_text = tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)
        print("\nFormatted input text:")
        print(input_text)
        
        # Show which parts are masked (labels = -100)
        labels = encoded['labels']
        masked_positions = (labels == -100).sum().item()
        total_positions = len(labels)
        print(f"\nMasked positions: {masked_positions}/{total_positions}")
        
        return True
        
    except Exception as e:
        print(f"Encoding failed: {e}")
        return False

if __name__ == "__main__":
    success = test_sot_encoding()
    if success:
        print("\n✅ SoT encoding test passed!")
    else:
        print("\n❌ SoT encoding test failed!")
