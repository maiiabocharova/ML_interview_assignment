from sentence_transformers import SentenceTransformer
import numpy as np
import json
from collections import defaultdict
import torch

dst_embeddings = np.load('resources/dst_names_embeddings.npy')

with open('resources/name2code.json') as f:
    name2code = json.load(f)

code2index = defaultdict(list)
index2code, index2name = {}, {}
for idx, (name, code) in enumerate(name2code.items()):
    code2index[code].append(idx)
    index2code[idx] = code
    index2name[idx] = name

device = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "resources/sbert_model"
sbert = SentenceTransformer(model_checkpoint, device=device)