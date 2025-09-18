import json
import os
from passage_retrieval_cuda import Retriever   
import glob
from transformers import AutoModelForCausalLM,AutoTokenizer,T5ForConditionalGeneration
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch
import re
from typing import Dict, Any,Tuple

## step2: 进行检索

def search_docs(queries,retriever,n_docs=5):
    if len(queries) < 1:
        print("检索query为空!")
        return []   
    if isinstance(queries, str):
        queries = [queries]
    docs_list = []
    for query in queries:
        docs = retriever.search_document_demo(query, n_docs=n_docs)
        sentences = [d["text"] for d in docs] if docs else []
        docs_list.append(sentences)   # ← list of list
    return docs_list




## step3: 批量检索json文件，并将结果编为新的json文件，跑数据集不用
def batch_search(file_name,retriever,output_file,domain):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    if domain == "outline":
        for item in data:
            queries = []
            for i in range(1, 7):         
                outline_key = f"outline{i}"
                query = item[outline_key] 
                queries.append(query)
            docs = search_docs(queries,retriever)
            item["outline_answer"] = docs
    if domain == "request":
        for item in data:
            query = item["request"]
            docs = search_docs(query,retriever)
            item["request_answer"] = docs
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已生成 {output_file}")


## step4：对单个问题进行tokenization,批量

def build_prompt_for_request(query,tokenizer,docs,num_docs):
    
    
    if isinstance(query, str):
        query = [query]
    if isinstance(docs, str) or (docs and not isinstance(docs[0], list)):
        docs = [docs]
    
    assert len(query) == len(docs),"queries 和 docs_batch 长度必须一致"
    
    doc_joiner = "\n\n------\n\n"
    doc_prefix= "[Document {idx}]\n"
    
    prompts = []
    
    for q,doc in zip(query,docs):
        docs_to_use = doc[:num_docs] if docs is not None else []
        
        if docs_to_use:
            doc_texts = []
            for i,d in enumerate(docs_to_use):
                doc_texts.append(doc_prefix.format(idx = i) + (d if d is not None else ""))
            docs_section = doc_joiner.join(doc_texts)
        else:
            docs_section = "(No retrieved documents.)"
        
        prompt = f"""Answer the question based on the provided documents. 
        
        Here are the retrieved documents:
        {docs_section}

        User Question:
        {q}

        Answer:"""

        prompts.append(prompt)

    return prompts

##对有outline的，先构造prompt
def build_prompt_for_outline(query,outlines,outlines_docs,num_docs,query_docs):
    if not outlines:
        raise ValueError("`outlines` must be a non-empty list of sub-questions.")
    
    def _clean_and_take(docs_list,k):
        if docs_list is None:
            return []
        cleaned = [d for d in docs_list if d is not None]
        return cleaned[:k]
    
    if isinstance(outlines_docs, list) and outlines_docs and isinstance(outlines_docs[0], list):
        # docs 为list[list[]] -- 里面是检索的list，外面是各个outline的list
        per_outline_docs = []
        for i, o in enumerate(outlines):
            if i < len(outlines_docs):
                per_outline_docs.append(_clean_and_take(outlines_docs[i], num_docs))
            else:
                per_outline_docs.append([])
        
    else:
        per_outline_docs = [[] for _ in outlines]
        print("在tokenize时docs的格式不对")
        
        
    cleaned_query_docs = _clean_and_take(query_docs, num_docs)
    if cleaned_query_docs:
        query_docs_str = "\n".join(
            [f"[Doc{i+1}] " + " ".join(str(d).split()) for i, d in enumerate(cleaned_query_docs)]
        )
    else:
        query_docs_str = "(No documents retrieved)"
    
    header = (
        "Answer the overall question based on the sub-questions below."
    )
    
    prompt_parts = [header, "", 
                    f"Overall Query: {query}",
        f"Documents for overall query:\n{query_docs_str}",
        "",
        "Sub-questions and their documents (format {outline:document}):",
        ""]
    
    for outline,doc_list in zip(outlines,per_outline_docs):
        if doc_list:
            for doc in doc_list:
                one_line_doc = " ".join(str(doc).split())
                prompt_parts.append("{" + f"{outline}:{one_line_doc}" + "}")
        else:
            prompt_parts.append("{" + f"{outline}:(No documents retrieved)" + "}")
            
    prompt = "\n".join(prompt_parts)
        
    return prompt
    
    

## step5：模型推理以及对结果标准化


## 加载模型config

MODELS_CONFIG = {
    "Llama-2-7b-chat-hf": {
        "model_path": "/home/sunkai/SoT_RAG_resources/model_ckpt/Llama-2-7b-chat-hf",
        "model_type": "causal_lm",
        "chat_template": True,
        "special_tokens": {"system": "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n", "user": "{} [/INST]", "assistant": " {} </s><s>[INST] "}
    },
    "Qwen2.5-1.5B": {
        "model_path": "/home/sunkai/SoT_RAG_resources/model_ckpt/Qwen2.5-1.5B",
        "model_type": "causal_lm",
        "chat_template": True,
        "special_tokens": {}
    },
    "deepseek-R1-Qwen": {
        "model_path": "/home/sunkai/SoT_RAG_resources/model_ckpt/deepseek-R1-Qwen",
        "model_type": "causal_lm", 
        "chat_template": True,
        "special_tokens": {}
    },
    "phi-2_hf": {
        "model_path": "/home/sunkai/SoT_RAG_resources/model_ckpt/phi-2_hf",
        "model_type": "causal_lm",
        "chat_template": False,
        "special_tokens": {}
    },
    "t5-large": {
        "model_path": "/home/sunkai/SoT_RAG_resources/model_ckpt/t5-large",
        "model_type": "seq2seq",
        "chat_template": False,
        "special_tokens": {}
    }
}

def load_model_and_tokenizer(model_name):
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"不支持的模型: {model_name}")
    config = MODELS_CONFIG[model_name]
    path = config["model_path"]
    model_type = config["model_type"]
    print(f"正在加载模型: {model_name} ({model_type})")
    
    tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code = True)
    
    #处理padtoken
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    #加载模型
    if model_type == "seq2seq":
        model = T5ForConditionalGeneration.from_pretrained(
            path, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype = torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    return model,tokenizer,config

def format_prompt_for_model(model_name,config,prompt):
    '''一些模型需要特殊prompt格式'''
    if model_name == "Llama-2-7b-chat-hf":
        # Llama-2 chat格式
        system_msg = "You are a helpful assistant. Answer questions based on the provided documents."
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{prompt} [/INST]"
        return formatted_prompt
    elif "deepseek" in model_name or "Qwen" in model_name:
        # Qwen系列模型格式
        formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return formatted_prompt
    else: return prompt
    
def get_generation_config(model_name):
    base_config = {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "repetition_penalty": 1.1
    }
    
    if "Llama" in model_name:
        base_config.update({
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
        })
    elif "Qwen" in model_name:
        base_config.update({
            "temperature": 0.3,
            "top_p": 0.8,
            "do_sample": True,
        })
    elif "deepseek" in model_name:
        base_config.update({
            "temperature": 0.3,
            "top_p": 0.95,
            "do_sample": True,
        })
    elif "phi" in model_name:
        base_config.update({
            "temperature": 0.2,
            "top_p": 0.9,
            "max_new_tokens": 200,
        })
    elif "t5" in model_name:
        base_config.update({
            "max_length": 256,  # T5使用max_length而不是max_new_tokens
            "num_beams": 4,
            "early_stopping": True,
        })
        base_config.pop("max_new_tokens")  # T5不需要max_new_tokens
        
    return base_config



def reasoning(model,tokenizer,model_name,prompts,config,save_pair_path) -> list[str]:
    '''推理并返回字符串答案'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    formatted_prompts = [format_prompt_for_model(model_name,config,prompt) for prompt in prompts]
    gen_kwargs = get_generation_config(model_name)
    
    pad_id = getattr(tokenizer,"pad_token_id",None)
    eos_id = getattr(tokenizer,"eos_token_id",None)
    
    
    if pad_id is None and eos_id is not None:
        pad_id = eos_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    
    if config["model_type"] == "seq2seq":
        #t5
        inputs = tokenizer(
            formatted_prompts,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 1024
        )
    else:
        inputs = tokenizer(
            formatted_prompts,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 2048
        )
    
    inputs = {k : v.to(device) for k,v in inputs.items()}
    
    with torch.no_grad():
        if config["model_type"] == "seq2seq":
            outputs = model.generate(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                **gen_kwargs
            )
            generated_texts = []
            for i in range(outputs.shape[0]):
                generated_text = tokenizer.decode(
                    outputs[i],
                    skip_special_tokens = True,
                    clean_up_tokenization_spaces = True
                )
                generated_texts.append(generated_text.strip())

        else:
            outputs = model.generate(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                **gen_kwargs
            )
            
            input_length = inputs["input_ids"].shape[1]
            generated_texts = []
            for i in range(outputs.shape[0]):
                generated_ids = outputs[i][input_length:]
                
                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens = True,
                    clean_up_tokenization_spaces = True
                )
                generated_texts.append(generated_text.strip())
    
    if save_pair_path:
        with open(save_pair_path, 'a', encoding='utf-8') as f:
            for q, a in zip(formatted_prompts, generated_texts):
                f.write(json.dumps({'question': q, 'answer': a}, ensure_ascii=False) + '\n')

    return generated_texts




## step6: evaluate

def evaluate_pipeline(model_name,predictions,save_output_path,dataset_name = "squad",max_samples = 2000,domain = "request",save_output = False):
    #predictions = List[answer]
    
    print(f"开始评估 - 模型: {model_name}, 数据集: {dataset_name}")
    
    # 加载evaluate库的指标
    print("加载评估指标...")
    exact_match = evaluate.load("exact_match")
    f1 = evaluate.load("f1") 
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    
    # 可修改以上指标
    
    print(f"加载数据集: {dataset_name}")
    if dataset_name == "squad":
        dataset = load_dataset("squad", split="validation")
        references = [ans["text"][0] if ans["text"] else "" 
                     for ans in dataset["answers"][:max_samples]]
    
    ## 这里可以根据数据集的字段来修改
    
    n_pred = len(predictions)
    n_ref = len(references)
    n = min(max_samples,n_pred,n_ref)
    if n_pred != n_ref:
        print(f"Warning: number of predictions ({n_pred}) != number of references ({n_ref}). Evaluating on first {n} items.")
    
    pred_use = predictions[:n]
    ref_use = references[:n]
    
    refs_for_bleu = [[r] if r != "" else [""] for r in ref_use]
    
    print("计算 Exact Match / F1 ...")
    em_res = exact_match.compute(predictions=pred_use, references=ref_use)
    f1_res = f1.compute(predictions=pred_use, references=ref_use)

    print("计算 ROUGE ...")
    try:
        rouge_res = rouge.compute(predictions=pred_use, references=ref_use)
    except Exception as e:
        rouge_res = {"error": str(e)}

    print("计算 BLEU ...")
    try:
        bleu_res = bleu.compute(predictions=pred_use, references=refs_for_bleu)
    except Exception as e:
        bleu_res = {"error": str(e)}

    print("计算 METEOR ...")
    try:
        meteor_res = meteor.compute(predictions=pred_use, references=ref_use)
    except Exception as e:
        meteor_res = {"error": str(e)}

    metrics = {
        "n_evaluated": n,
        "exact_match": em_res,
        "f1": f1_res,
        "rouge": rouge_res,
        "bleu": bleu_res,
        "meteor": meteor_res,
    }
    
    if save_output:
        out_list = []
        for i in range(n):
            out_list.append({"prediction": pred_use[i],"reference": ref_use[i]})
        with open(save_output_path,"w",encoding= "utf8") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions+references to {save_output_path}")
    
    # 打印摘要
    print("评估完成。摘要：")
    print(f" Evaluated samples: {n}")
    print(f" Exact Match: {em_res}")
    print(f" F1: {f1_res}")
    print(f" ROUGE: keys = {list(rouge_res.keys())}")
    print(f" BLEU: {bleu_res if isinstance(bleu_res, dict) else bleu_res}")

    return metrics


def LLM_as_a_judge(predictions,model_name,queries): ## query是listof request
    import openai
    from openai import OpenAI
    
    client = OpenAI()
    results = []
    
    evaluate_prompt = """
    Please judge the following answers based on their queries. Only return the results in json format and don't return
    any other things.
    
    query: {query}
    answer: {answer}
    
    Evaluation Dimensions (10 points each):
    1. Accuracy: The extent to which the answer aligns with the facts.
    2. Completeness: The degree to which the answer covers all requests of the queries.
    3. Relevance: The extent to which the answer is related to the question.
    4. Clarity: The clearness of the answer's expression.
    5. Practicality: The practical value of the answer.
    
    return format: 
    {{
        "accuracy": 
        "completeness": 
        "relevance": 
        "clarity": 
        "usefulness": 
        "overall": 
        "comments": 
    }}
    
    """
    
    for i, query , prediction in enumerate(zip(queries,predictions)):
        prompt = evaluate_prompt.format(query=query,answer=prediction)
        
        try:
            response = client.chat.completions.create(
                model = model_name,
                messages=[
                    {"role": "user" , "content": prompt}
                ],
                temperature= 0.1,
                max_tokens= 1000
            )
            evaluation_text = response.choices[0].message.content.strip()
            
            try:
                evaluation_scores = json.loads(evaluation_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a default response
                evaluation_scores = {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "clarity": 0,
                    "usefulness": 0,
                    "overall": 0,
                    "comments": "Failed to parse evaluation response"
                }
            result_entry = {
                "index" : i,
                "query": query,
                "prediction" : prediction,
                "evaluation" : evaluation_scores
            }
            
            results.append(result_entry)
            print(f"Processed query {i+1}/{len(queries)}")
        
        except Exception as e:
            print(f"Error processing query {i+1}: {str(e)}")
            # Create error entry
            error_entry = {
                "index": i,
                "query": query,
                "prediction": prediction,
                "evaluation": {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "clarity": 0,
                    "usefulness": 0,
                    "overall": 0,
                    "comments": f"Error during evaluation: {str(e)}"
                }
            }
            results.append(error_entry)
    
    output_data = {
        "model":model_name,
        "total_evaluation":len(results),
        "results": results
    }
    with open('LLM_judge_result.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to LLM_judge_result.json")
    print(f"Total evaluations: {len(results)}")
    
    return results
   


## main pipeline
