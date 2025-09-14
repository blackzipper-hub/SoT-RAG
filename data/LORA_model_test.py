#!/usr/bin/env python
# coding=utf-8

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import argparse

def load_model(base_model_path, lora_model_path=None):
    """加载模型和分词器"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    # 如果有LoRA适配器，则加载
    if lora_model_path:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, lora_model_path)

    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=150):
    """生成模型响应"""
    # 添加提示词
    prompt = "Please generate only the framework with exactly 5 questions, do not generate any other content.\n\n" + instruction

    # 构造完整输入
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # 编码输入
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def load_test_data_from_file(test_file):
    """从文件加载测试数据（包含完整信息）"""
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                test_data.append(data)
            except:
                print(f"Skipping invalid line: {line}")
    return test_data[:10]  # 只返回前10条

def test_with_sample_data(model, tokenizer, test_data=None):
    """使用样本数据测试模型并对比参考答案"""

    if test_data is None:
        # 默认测试样本
        test_data = [
            {
                "instruction": "What are the main causes of climate change?",
                "output": [
                    "What greenhouse gases contribute most to climate change?",
                    "How do human activities impact global temperature?",
                    "What natural factors influence climate patterns?"
                ]
            },
            {
                "instruction": "How does artificial intelligence impact society?",
                "output": [
                    "How does AI influence employment and job markets?",
                    "What ethical concerns arise with AI decision-making?",
                    "How do privacy laws need to adapt to AI technologies?"
                ]
            }
        ]

    print("\n" + "="*60)
    print("MODEL TESTING WITH REFERENCE COMPARISON")
    print("="*60)

    for i, item in enumerate(test_data, 1):
        instruction = item['instruction']
        reference_output = item['output']

        print(f"\nTest {i}: {instruction}")
        print("-" * 50)

        # 生成模型响应
        response = generate_response(model, tokenizer, instruction)
        print("Model Response:")
        print(response)
        print("-" * 30)

        # 显示参考答案
        print("Reference Output:")
        if isinstance(reference_output, list):
            for j, question in enumerate(reference_output, 1):
                print(f"{j}. {question}")
        else:
            print(reference_output)
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Test trained LoRA model with reference comparison")
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--lora_model_path", type=str,
                       help="Path to LoRA adapter (optional)")
    parser.add_argument("--test_file", type=str,
                       help="Path to test data file (jsonl format)")

    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(args.base_model_path, args.lora_model_path)

    # 准备测试数据
    if args.test_file:
        test_data = load_test_data_from_file(args.test_file)
        print(f"Loaded {len(test_data)} test samples")
    else:
        test_data = None  # 使用默认测试样本

    # 测试模型
    test_with_sample_data(model, tokenizer, test_data)

if __name__ == "__main__":
    main()
