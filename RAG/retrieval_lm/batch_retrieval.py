#!/usr/bin/env python3
# batch_outline_search.py
import json
import os
from passage_retrieval import Retriever   
import glob


MODEL_NAME   = "facebook/contriever-msmarco"   
PASSAGES_TSV = "D:/SoT-RAG/SoT-RAG-main/psgs_w100.tsv/psgs_w100.tsv"
EMB_MULTI = (
    "D:/SoT-RAG/SoT-RAG-main/wikipedia_embeddings/wikipedia_embeddings/passages_00;"
    "D:/SoT-RAG/SoT-RAG-main/wikipedia_embeddings/wikipedia_embeddings/passages_01;"
    "D:/SoT-RAG/SoT-RAG-main/wikipedia_embeddings/wikipedia_embeddings/passages_02;"
    "D:/SoT-RAG/SoT-RAG-main/wikipedia_embeddings/wikipedia_embeddings/passages_03"
)
N_DOCS       = 1          # 只要一句话
SAVE_INDEX   = False      # 是否复用已有的 faiss 索引

glob.glob = lambda x: x if isinstance(x, list) else __import__('glob').glob(x)

def main():

    emb_files = EMB_MULTI.split(";")
    retriever = Retriever(args=None)
    retriever.setup_retriever_demo(
        model_name_or_path=MODEL_NAME,
        passages=PASSAGES_TSV,
        passages_embeddings=emb_files,
        n_docs=N_DOCS,
        save_or_load_index=SAVE_INDEX
    )


    with open("./test_input.json", "r", encoding="utf-8") as f:
        data = json.load(f)


    for item in data:
        for i in range(1, 7):         
            outline_key = f"outline{i}"
            query = item[outline_key] 
            docs = retriever.search_document_demo(query, n_docs=N_DOCS)
            sentence = docs[0]["text"] if docs else ""
            item[f"ontline_result{i}"] = sentence.strip()
        question_key = "request"
        question_query = item[question_key]
        docs = retriever.search_document_demo(question_query,n_docs=N_DOCS)
        sentence = docs[0]["text"] if docs else ""
        item["request_result"] = sentence.strip()

    with open("output_full.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("已生成 output_full.json")

if __name__ == "__main__":
    main()