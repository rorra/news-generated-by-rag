#!/bin/env bash
sudo docker run -p 6333:6333 \
    -v $(pwd)/data:/qdrant/storage \
    qdrant/qdrant
