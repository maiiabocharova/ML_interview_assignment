from fastapi import FastAPI, Form, UploadFile
import numpy as np
from sentence_transformers import SentenceTransformer, util
from resources import dst_name2embedding, code2name, sbert, dst_embeddings, dst_names, dst_codes
import json

app = FastAPI()


@app.post("/predict_on_file")
async def predict_on_file(file: UploadFile, top_k: str = Form("3")):
    results = {}
    top_k = int(top_k)
    file_bytes = await file.read()
    file_text = file_bytes.decode('utf-8')
    json_file = json.loads(file_text)

    query_names = [el['name'] for el in json_file.values()]

    query_embeddings = sbert.encode(query_names)
    query_name2embedding = {name: emb for name, emb in zip(query_names, query_embeddings)}

    name2embedding = dst_name2embedding | query_name2embedding

    for idx, values in json_file.items():
        query_emb = name2embedding[values['name']]
        code = values['code']
        candidates_found = False
        if code:
            try:
                code = int(code.replace("-", "").ljust(8, '0'))
            except:
                code = None

            if code:
                start = code - 8 * 10 ** 6
                end = code + 8 * 10 ** 6
                candidates_names = []
                candidates_codes = []
                for code in code2name:
                    if start <= int(code.replace("-", "")) <= end:
                        candidates_names.extend(code2name[code])
                        candidates_codes.extend(len(code2name[code]) * [code])
                candidates_names = np.array(candidates_names)
                candidates_codes = np.array(candidates_codes)

                if candidates_names.size >= 1:
                    candidates_found = True

                    cands_embeddings = np.array([name2embedding[name] for name in candidates_names])

                    similarities = util.cos_sim(query_emb, cands_embeddings)[0].numpy()

                    curr_k = min(top_k, len(candidates_names))
                    ind = np.argpartition(
                        similarities,
                        -curr_k
                    )[-curr_k:]

                    indexes = ind[np.argsort(similarities[ind])][::-1]
                    similarities = sorted(similarities, reverse=True)
                    similarities = [str(round(el, 2)) for el in similarities]
                    results[idx] = list(zip(candidates_names[indexes[:top_k]],
                                            candidates_codes[indexes[:top_k]],
                                            similarities))
        if not candidates_found:
            similarities = util.cos_sim(query_emb, dst_embeddings)[0].numpy()
            curr_k = min(top_k, len(dst_names))
            ind = np.argpartition(
                similarities,
                -curr_k
            )[-curr_k:]

            indexes = ind[np.argsort(similarities[ind])][::-1]
            similarities = sorted(similarities, reverse=True)
            similarities = [str(round(el, 2)) for el in similarities]
            results[idx] = list(zip(dst_names[indexes[:top_k]], dst_codes[indexes[:top_k]], similarities))

    return results
