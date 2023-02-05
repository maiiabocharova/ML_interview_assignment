from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch

dst_embeddings = np.load('resources/dst_names_embeddings.npy')

with open('resources/name2code.json') as f:
    name2code = json.load(f)

index2code, index2name = {}, {}
for idx, (name, code) in enumerate(name2code.items()):
    index2code[idx] = code
    index2name[idx] = name

device = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "resources/sbert_model"
sbert = SentenceTransformer(model_checkpoint, device=device)