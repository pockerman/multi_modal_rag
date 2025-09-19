# Document fusion & multi-step retrieval of multi-modal RAG

Toy implementation of hybrid document fusion and multi-step retrieval for multi-modal RAG with images and text modalities.
The implementation that is used is as follows:

1. Indexing: For each image produce one or more representations (embeddings, captions, metadata). Two collections are used one for the images and one for the description of the image
2. Multi-stage retrieval. 
   1. Retrieve: Get the top K images and image descriptions i.e 2 DB queries
   2. Fuse: Merge candidate sets, normalize scores, compute fused score (weighted and RRF) 
   3. Re-rank: Run cross-modal re-ranker (e.g., cross-encoder that takes query + image) over top N fused candidates; final ranking from re-rank or fused+rerank.
3. Generate: top M candidates (images + captions + metadata) into context for the generator. If generator is text-only, supply captions & metadata; if multi-modal generator, feed images directly.

There are various options for the retriever

- Visual dense retriever: CLIP (ViT-B/32 or ViT-L) embeddings stored in FAISS (HNSW/IVF+PQ for scale).
- Text retriever: BM25 (Elastic) or vector text search (Chroma/FAISS) on captions/tags.
- Optional geometric / local: ORB / SIFT + FLANN for near-duplicate or precise visual matches.
- Cross-modal re-ranker: A model that takes query (image or text) and each candidate image and returns a relevance score. Off-the-shelf: CLIP-Score, BLIP-2 cross encoder, or a small transformer fine-tuned for ranking.

### Score normalization & fusion

Different retrievers produce incompatible scores; thus before fusion we need to
normalize the results. Normalization options (per query over candidate set):

- Min-max: ```s' = (s - min)/(max-min)```
- Softmax: ```s' = exp(s)/sum(exp(s))``` (makes scores comparable as probabilities)
- Rank normalization: convert to ```1/(rank + k)``` or scale ranks to ```[0,1]```

Similarly, several options exist to fuse the results:

- Simple weighted fusion: ```fused = w_vis * vis_score_norm + w_txt * txt_score_norm```
- Reciprocal Rank Fusion (RRF) — robust, hyperparameter k (e.g., 60): ```RRF_score = sum(1 / (k + rank_i))``` for each retriever i
- Learned fusion: Train a small model (logistic regression, LightGBM) on features:
   - normalized scores, ranks, metadata features (age, popularity)
   - optionally cross-encoder score if available during training

Label with human judgments / clicks.

### Re-ranking 

Cross-encoder that consumes (query, candidate) pair and outputs a high-quality score. For images:

- Use BLIP-2 / Flamingo / a cross-attention image+text model — fine-tune for relevance.
- Or use CLIP similarity as a fast re-ranker if cross-encoder infrastructure not available.
- Batch inference on GPU for speed.



## How to use

Install the requirements and start a ChromaDB node: 

```
chroma run --path ./chromadb --host 0.0.0.0 --port 8003
```

Run the ```ingest_pipeline.py``` script to populate the database. This creates three collections:

- ```faqs_repo```
- ```defects_images```
- ```defects_texts```

Start the Ollama server

```
OLLAMA_HOST=0.0.0.0 OLLAMA_PORT=11434 ollama serve
```

## References

1. <a href="https://huyenchip.com/2023/10/10/multimodal.html">Multimodality and Large Multimodal Models (LMMs)</a>  
