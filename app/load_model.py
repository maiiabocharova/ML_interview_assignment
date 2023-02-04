from sentence_transformers import SentenceTransformer

model_checkpoint = "all-MiniLM-L6-v2"
sbert = SentenceTransformer(model_checkpoint)
sbert.save("resources/sbert_model")
