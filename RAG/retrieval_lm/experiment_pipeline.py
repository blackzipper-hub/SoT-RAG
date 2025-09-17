from experiment_func import search_docs,build_prompt_for_outline,build_prompt_for_request,reasoning,evaluate_pipeline,load_model_and_tokenizer
from typing import List, Tuple, Optional, Dict, Any
import math
from tqdm import tqdm
import glob
from passage_retrieval_cuda import Retriever
from datasets import load_dataset
import json

def run_experiment_request(
     model,
    tokenizer,
    config: Dict[str, Any],
    model_name: str,
    requests: List[str],
    retriever,
    save_output_path: str,
    dataset_name: str = "squad",
    batch_size: int = 8,
    num_docs: int = 3,
    max_prompts: Optional[int] = None,
    save_output: bool = True,
):
    if max_prompts is not None:
        requests = requests[:max_prompts]
        
    if not isinstance(requests, list) or len(requests) == 0:
        raise ValueError("requests must be a non-empty list of query strings.")
    
    print(f"Running batched retrieval for {len(requests)} queries...")
    
    docs_batch = search_docs(requests,retriever,n_docs=num_docs)
    
    if not isinstance(docs_batch, list) or len(docs_batch) != len(requests):
        # 如果检索器返回不规范
        print("Warning: retriever returned unexpected docs structure; aligning to requests length.")
    
    
    print("Building prompts...")
    all_prompts = build_prompt_for_request(requests,tokenizer,docs_batch,num_docs)
    if not isinstance(all_prompts, list) or len(all_prompts) != len(requests):

        print("Warning: build_prompt_for_request returned unexpected shape; trying to coerce to list.")
    
    print("Running inference in batches...")
    predictions = []
    
    n = len(all_prompts)
    n_batches = (n + batch_size - 1) // batch_size
    for bi in range(n_batches):
        st = bi * batch_size
        end = min((bi+1) * batch_size, n)
        batch_prompts = all_prompts[st:end]
        batch_preds = reasoning(model, tokenizer, model_name, batch_prompts, config,
    save_pair_path='q_a_pairs.jsonl')
        if not isinstance(batch_preds, list):
            raise RuntimeError("reasoning() must return a list of strings for the batch.")
        predictions.extend(batch_preds)
        print(f" Batch {bi+1}/{n_batches} done. Generated {len(batch_preds)} outputs.")
    '''
    print("Evaluating predictions...")
    metrics = evaluate_pipeline(model_name,predictions,save_output_path,dataset_name,max_samples=2000,save_output=True)
    '''
    return predictions#,metrics

def run_experiment_outlines(
     model,
    tokenizer,
    config: Dict[str, Any],
    model_name: str,
    overall_queries: List[str],
    outlines_batch: List[List[str]],
    retriever,
    save_output_path: str,
    dataset_name: str = "squad",
    batch_size: int = 4,
    num_docs: int = 3,
    max_prompts: Optional[int] = None,
    save_output: bool = True,
):
    
    assert len(overall_queries) == len(outlines_batch), "overall_queries and outlines_batch must have same length"

    if max_prompts is not None:
        overall_queries = overall_queries[:max_prompts]
        outlines_batch = outlines_batch[:max_prompts]
    if not overall_queries:
        return [], {}
        
    print(f"Running outline-based retrieval for {len(overall_queries)} queries...")
    
    overall_docs_batch = search_docs(overall_queries, retriever,n_docs=num_docs)
    
    outlines_docs_batch = []
    for outlines in outlines_batch:
        if outlines:
            outline_docs = search_docs(outlines, retriever, n_docs=num_docs)
            outlines_docs_batch.append(outline_docs)
        else:
            outlines_docs_batch.append([])
    
    print("Building prompts...")
    all_prompts = []
    for i, (query, outlines) in enumerate(zip(overall_queries, outlines_batch)):
        query_docs = overall_docs_batch[i] if i < len(overall_docs_batch) else []
        outlines_docs = outlines_docs_batch[i] if i < len(outlines_docs_batch) else []
        
        if outlines:
            prompt = build_prompt_for_outline(
                query, outlines, outlines_docs, num_docs, query_docs
            )
            all_prompts.append(prompt)
        else:
            simple_prompt = build_prompt_for_request([query], tokenizer, [query_docs], num_docs)
            all_prompts.extend(simple_prompt)
    
    print("Running inference in batches...")
    predictions = []
    
    n = len(all_prompts)
    n_batches = (n + batch_size - 1) // batch_size
    for bi in range(n_batches):
        st = bi * batch_size
        end = min((bi+1) * batch_size, n)
        batch_prompts = all_prompts[st:end]
        batch_preds = reasoning(model, tokenizer, model_name, batch_prompts, config,
    save_pair_path='q_a_pairs.jsonl')
        if not isinstance(batch_preds, list):
            raise RuntimeError("reasoning() must return a list of strings for the batch.")
        predictions.extend(batch_preds)
        print(f" Batch {bi+1}/{n_batches} done. Generated {len(batch_preds)} outputs.")
    
    '''
    print("Evaluating predictions...")
    metrics = evaluate_pipeline(
        model_name, predictions, save_output_path, 
        dataset_name, max_samples=2000, save_output=save_output
    )
    '''
    
    return predictions#, metrics

    

def main():
    #仅query
    # Configuration
    MODEL_NAME = "facebook/contriever-msmarco"
    PASSAGES_TSV = "/home/zhenghao/SoT-RAG/psgs_w100.tsv/psgs_w100.tsv"
    EMB_MULTI = (
        "/home/zhenghao/SoT-RAG/wikipedia_embeddings/wikipedia_embeddings/passages_00;"
        "/home/zhenghao/SoT-RAG/wikipedia_embeddings/wikipedia_embeddings/passages_01;"
        "/home/zhenghao/SoT-RAG/wikipedia_embeddings/wikipedia_embeddings/passages_02;"
        "/home/zhenghao/SoT-RAG/wikipedia_embeddings/wikipedia_embeddings/passages_03"
    )
    N_DOCS = 3
    SAVE_INDEX = False
    
    # Model configuration
    target_model = "Qwen2.5-1.5B"  
    batch_size = 4
    max_samples = 1000 
    
    # Output paths
    save_output_path_request = f"results_{target_model.replace('/', '_')}_request.json"
    save_output_path_outline = f"results_{target_model.replace('/', '_')}_outline.json"
    
    print("=" * 60)
    print("RAG Pipeline Experiment with SQuAD Dataset")
    print("=" * 60)
    
    try:
        print("1. Setting up retriever...")
        glob.glob = lambda x: x if isinstance(x, list) else __import__('glob').glob(x)
        
        emb_files = EMB_MULTI.split(";")
        retriever = Retriever(args=None)
        retriever.setup_retriever_demo(
            model_name_or_path=MODEL_NAME,
            passages=PASSAGES_TSV,
            passages_embeddings=emb_files,
            n_docs=N_DOCS,
            save_or_load_index=SAVE_INDEX
        )
        print("Retriever setup complete!")
        
        print(f"\n2. Loading model: {target_model}")
        model, tokenizer, config = load_model_and_tokenizer(target_model)
        print(" Model loaded successfully!")
        
        print("\n3. Loading SQuAD dataset...")
        dataset = load_dataset("squad", split="validation")
        
        # Extract questions as requests
        requests = [item["question"] for item in dataset[:max_samples]]
        print(f" Loaded {len(requests)} questions from SQuAD")
        
        # Step 4: Run request-based experiment
        print(f"\n4. Running request-based RAG experiment...")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Number of docs per query: {N_DOCS}")
        print(f"   - Max samples: {max_samples}")
        
        predictions_request= run_experiment_request(  # , metric
            model=model,
            tokenizer=tokenizer,
            config=config,
            model_name=target_model,
            requests=requests,
            retriever=retriever,
            save_output_path=save_output_path_request,
            dataset_name="squad",
            batch_size=batch_size,
            num_docs=N_DOCS,
            max_prompts=max_samples,
            save_output=True
        )
        
        print("  Request-based experiment completed!")
        
        """
        print(f"\n5. Running outline-based RAG experiment (empty outlines)...")
        
        # Create empty outlines for all requests
        empty_outlines_batch = [[] for _ in requests]
        
        predictions_outline, metrics_outline = run_experiment_outlines(
            model=model,
            tokenizer=tokenizer,
            config=config,
            model_name=target_model,
            overall_queries=requests,
            outlines_batch=empty_outlines_batch,
            retriever=retriever,
            save_output_path=save_output_path_outline,
            dataset_name="squad",
            batch_size=batch_size,
            num_docs=N_DOCS,
            max_prompts=max_samples,
            save_output=True
        )
        
        print(" Outline-based experiment completed!")
        """
        '''
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)
        
        
        def extract_metric_value(metric_dict, key):
            if isinstance(metric_dict, dict):
                return metric_dict.get(key, "N/A")
            return metric_dict if metric_dict is not None else "N/A"
        
        # Request-based results
        print("Request-Based RAG:")
        print(f"  - Total samples: {len(predictions_request)}")
        print(f"  - Results saved to: {save_output_path_request}")
        em_request = extract_metric_value(metrics_request.get("exact_match", {}), "exact_match")
        f1_request = extract_metric_value(metrics_request.get("f1", {}), "f1")
        print(f"  - Exact Match: {em_request}")
        print(f"  - F1 Score: {f1_request}")
        metrics_request_path = f"metrics_{target_model.replace('/', '_')}_request.json"
        with open(metrics_request_path, "w", encoding="utf-8") as f:
            json.dump(metrics_request, f, ensure_ascii=False, indent=2)
        print(f"  - Metrics saved to: {metrics_request_path}")
        """
        # Outline-based results
        print("\nOutline-Based RAG (Empty Outlines):")
        print(f"  - Total samples: {len(predictions_outline)}")
        print(f"  - Results saved to: {save_output_path_outline}")
        em_outline = extract_metric_value(metrics_outline.get("exact_match", {}), "exact_match")
        f1_outline = extract_metric_value(metrics_outline.get("f1", {}), "f1")
        print(f"  - Exact Match: {em_outline}")
        print(f"  - F1 Score: {f1_outline}")
        metrics_outline_path = f"metrics_{target_model.replace('/', '_')}_outline.json"
        with open(metrics_outline_path, "w", encoding="utf-8") as f:
            json.dump(metrics_outline, f, ensure_ascii=False, indent=2)
        print(f"  - Metrics saved to: {metrics_outline_path}")
        """
        '''
        # Sample predictions
        print(f"\n{'-'*40}")
        print("SAMPLE PREDICTIONS")
        print(f"{'-'*40}")
        
        for i in range(min(3, len(predictions_request))):
            print(f"\nExample {i+1}:")
            print(f"Question: {requests[i]}")
            print(f"Request-based: {predictions_request[i][:100]}...")
            """
            if i < len(predictions_outline):
                print(f"Outline-based: {predictions_outline[i][:100]}...")
            print("-" * 40)
            """
        
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()