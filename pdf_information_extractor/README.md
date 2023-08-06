# ML interview assignment
Extraction of key information from PDF

## Module Usage
### With docker-compose
Build docker container
`docker-compose up --build`
### Without docker-compose
```
cd app
docker build -t maiia-ml .
docker run -it -p 8060:80 maiia-ml
```
By default port specified is `8060`, you can change it by specifying the port which you want to expose.

## cURL command to extract `Product Name`, `Manufacturer`, `Part Number` using OpenAI

```
curl -F "file=@path_to_pdf_file" \
     -F "api_key=your_openai_api_key" \
     -F "model=gpt-4" \
     http://127.0.0.1:8060/predict_key_information > output_file.json
```
For free trial keys limitation is ~20-30 seconds per request, so ensure that you do not surpass the limitations
`model` parameter can be either "gpt-4" or "gpt-3.5", defaults to "gpt-4"

Returns json:
```
{
    "product_name": product_name,
    "manufacturer_name": manufacturer_name,
    "part_number": part_number
}
```
## Limitations: 
to save compute, since in the provided documents needed information was present on the first page - 
only the first page is passed to the OpenAI

## cURL command to detect tables
```
curl -F "file=@path_to_pdf_file" \
     -F "page=1" \
     http://127.0.0.1:8060/detect_tables_structure > img.png
```
Numeration starts from 0.

