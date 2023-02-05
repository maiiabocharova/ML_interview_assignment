from fastapi import FastAPI, Form, UploadFile
import numpy as np
from sentence_transformers import SentenceTransformer, util
from resources import sbert, dst_embeddings, index2code, index2name
import json
from collections import defaultdict

app = FastAPI()


@app.post("/predict_on_file")
async def predict_on_file(file: UploadFile, top_k: str = Form("3")):
    results = defaultdict(list)
    top_k = int(top_k)
    file_bytes = await file.read()
    file_text = file_bytes.decode('utf-8')
    json_file = json.loads(file_text)

    query_names = [el['name'] for el in json_file.values()]

    query_embeddings = sbert.encode(query_names)

    similarities = util.cos_sim(query_embeddings, dst_embeddings).numpy()

    for source_idx, (key, values) in enumerate(json_file.items()):
        src_code = values['code']
        candidates_indexes = []
        if src_code:
            try:
                src_code = int(src_code.replace("-", "").ljust(8, '0'))
                start = src_code - 8 * 10 ** 6
                end = src_code + 8 * 10 ** 6
                candidates_indexes = []
                for idx, dst_code in index2code.items():
                    if start <= int(dst_code.replace("-", "")) <= end:
                        candidates_indexes.append(idx)
            except ValueError:
                pass

        if not candidates_indexes:
            candidates_indexes = np.arange(len(index2code))

        cand_similarities = similarities[source_idx, candidates_indexes]

        curr_k = min(top_k, len(candidates_indexes))
        ind = np.argpartition(
            cand_similarities,
            -curr_k
        )[-curr_k:]

        indexes = ind[np.argsort(cand_similarities[ind])][::-1][:top_k]
        cand_similarities = sorted(cand_similarities, reverse=True)[:top_k]
        cand_similarities = [str(round(el, 2)) for el in cand_similarities]

        for idx, sim in zip(np.array(candidates_indexes)[indexes], cand_similarities):
            results[key].append((
                index2name[idx],
                index2code[idx],
                sim
            ))

    return results
