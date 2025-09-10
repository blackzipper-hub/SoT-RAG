## Download data
## Download preprocessed passage data used in DPR.

cd retrieval_lm
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
### Then, download the generated passages. We use Contriever-MSMARCO

wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
## Run retriever
### You can run passage retrieval by running the command below.


cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data YOUR_INPUT_FILE  \
    --output_dir YOUR_OUTPUT_FILE \
    --n_docs 20

## Your input file should be either a json or jsonl. Each instance must contain either question or instruction, which will be used as a query during retrieval.


## Generate embeddings for your own data
## You can generate embeddings for your own data by running the following command. (The script is adapted from the Contriever repository.) Note that generating embeddings from a large-scale corpus (>10M docs) can take time, and we recommend running it on multiple GPUs.

cd retrieval_lm
for i in {0..3}; do
  export CUDA_VISIBLE_DEVICES=${i}
  python generate_passage_embeddings.py  --model_name_or_path facebook/contriever-msmarco \
  --output_dir YOUR_OUTPUT_DIR \
  --passages YOUR_PASSAGE_DATA --shard_id ${i}  --num_shards 4 > ./log/nohup.my_embeddings.${i} 2>&1 &