import openai
import requests
from operator import itemgetter
import json
import easyocr
from PIL import Image
import io
import numpy as np

easyocr_reader = easyocr.Reader(['en'])


def process_file(api_key, pdf_doc, model):
    openai.api_key = api_key
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key,
    }
    page = pdf_doc[0]
    imgs = page.get_images()
    img_num = 0
    page_blocks = []
    for block in page.get_text("blocks")[:10]:
        # it's image
        if block[-1] == 1:
            baseImage = pdf_doc.extract_image(imgs[img_num][0])
            img = Image.open(io.BytesIO(baseImage['image']))
            text = "\n".join(
                easyocr_reader.readtext(np.array(img), detail=0, paragraph=True)
            ).strip()
            img_num += 1

        # it's text
        else:
            text = block[4].strip()
        page_blocks += [(round(block[0]),
                         round(block[1]),
                         round(block[2]),
                         round(block[3]),
                         text)]

    page_blocks.sort(key=itemgetter(1, 0))
    text = ""
    for block in page_blocks[:15]:
        if block[-1] == 0:
            text += block[-2] + "\n"
    query = text.strip()
    prompt = 'Extract Product name, Manufacturer, Part Number (only if present) ' \
                    f'from the following extract of a datasheet "{query}".' \
                    ' Return result as a json. Return ONLY json, no other words'
    model_name = "gpt-4-0613" if model == "gpt-4" else 'gpt-3.5-turbo'
    json_data = {
        'model': model_name,
        'messages': [
            {
                'role': 'user',
                'content': prompt,
            },
        ],
        'temperature': 0,
        'max_tokens': 128,
    }
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        json=json_data,
        headers=headers,
        timeout=50
    ).json()
    response = response['choices'][0]['message']['content']

    json_str = json.loads(response)
    json_str = {
        key: val.replace("Not mentioned in the extract", "") for key, val in json_str.items()
    }

    return json_str