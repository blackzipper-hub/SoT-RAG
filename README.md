

# SoR: Skeleton-of-RAG Framework

**Authors:** First Author (with others)
**Status:** Submitted to *Knowledge-Based Systems* (Under Review)

---

## Overview

The **Skeleton-of-RAG (SoR) Framework** is designed to enhance the reasoning efficiency and retrieval effectiveness of LLM-based agents. The framework introduces a structured approach that leverages **skeleton-of-thought decomposition**, **fine-grained retrieval augmentation**, and **block-level embedding compression**, forming a complete pipeline for knowledge-intensive tasks.

---

## Module 1: Skeleton-of-Thought Query Decomposition

This module employs the **skeleton-of-thought paradigm** to decompose complex queries into structured **problem skeletons** and corresponding sub-questions.

* A **Llama3-1.5B model** is adapted with **LoRA fine-tuning** and optimized through **PPO reinforcement learning**.
* The fine-tuned model generates a **sub-question framework**, ensuring queries are reorganized into smaller, manageable tasks.
* This decomposition provides the foundation for **parallel sub-answer generation** and improves the clarity of reasoning chains.

---

## Module 2: Fine-Grained Sub-Question Retrieval

In the second module, the sub-questions produced by the skeleton framework are used to perform **retrieval-augmented generation (RAG)**.

* Each sub-question is mapped to knowledge sources with **fine-grained retrieval alignment**, rather than coarse document-level matching.
* This approach enables **more precise context retrieval** and improves the integration of external knowledge with the generation process.
* By grounding each sub-answer in high-relevance evidence, the framework enhances factual accuracy and contextual completeness.

---

## Module 3: Block-Level Embedding Compression

The final module focuses on efficiency optimization through **block-level embedding compression**.

* Retrieved content is encoded and compressed into compact **block embeddings**, reducing the input sequence length to a fraction of its original size.
* This design preserves the key semantic information while significantly **reducing inference cost**.
* The compression strategy accelerates **time-to-first-token (TTFT)**, enabling faster response generation in LLM-based agents.

---

## Highlights

* **Skeleton-driven decomposition** for structured query understanding.
* **Fine-grained retrieval augmentation** to ensure precise evidence integration.
* **Compression-based optimization** to lower computational costs and improve response latency.


