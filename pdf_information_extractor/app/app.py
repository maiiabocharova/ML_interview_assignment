from fastapi import FastAPI, Form, UploadFile
import fitz
from process_file import process_file
from table_parsing import visualize_table
from io import BytesIO
from fastapi.responses import Response


app = FastAPI()


@app.post("/predict_key_information")
async def predict_key_information(file: UploadFile, model: str = Form("gpt-4"), api_key: str = Form("")):
    stream = await file.read()
    pdf_doc = fitz.open(stream=stream, filetype="pdf")

    return process_file(api_key=api_key, pdf_doc=pdf_doc, model=model)


@app.post("/detect_tables_structure")
async def detect_tables_structure(file: UploadFile, page: str = Form("1")):
    stream = await file.read()
    pdf_doc = fitz.open(stream=stream, filetype="pdf")
    image = visualize_table(pdf_doc, page)
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    return Response(content=image_bytes, media_type="image/png")