# import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from collections import defaultdict
import torch

# index = faiss.read_index("resources/centroids_index")
with open('resources/name2code.json') as f:
    name2code = json.load(f)

code2name = defaultdict(list)
code2index = defaultdict(list)
index2code = {}
index2name = {}
for idx, (name, code) in enumerate(name2code.items()):
    code2name[code].append(name)
    code2index[code].append(idx)
    index2code[idx] = code
    index2name[idx] = name

dst_names = np.array(list(name2code.keys()))
dst_codes = np.array(list(name2code.values()))
dst_embeddings = np.load('resources/dst_names_embeddings.npy')

dst_name2embedding = {name: emb for name, emb in zip(name2code, dst_embeddings)}

device = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "resources/sbert_model"
sbert = SentenceTransformer(model_checkpoint, device=device)