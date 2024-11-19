#!/usr/bin/env bash
for embedder in tfidf bm25 dpr sbert minilm; do
    python scripts/index_articles.py --local --embedder-type $embedder --use-processed
done
