# ML interview assignment
Normalization of free form phrases to company's internal taxonomy

## Analysis and Logic
You can find the notebook with analysis and logic which I used to develop the module in the `notebooks` folder.

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

## cURL command to normalize names in the file

```
curl -F "file=@path_to_json_file" \
     -F "top_k=3" \
     http://127.0.0.1:8060/predict_on_file > output_fil.json
```
Where parameter `top_k` specifies how many suggestions ML module should return.

Structure of json file should be as follows:
```python
{
    0: {
        'code': code,
        'name': name
    },
    1: {
        'code': code,
        'name': name
    },
    ...
}
```
Example fie is provided `app/resources/test_file.json`


Returns json:
```
{
    0: [(predicted_name_1, predicted_code_1, distance_1), , ...(predicted_name_k, predicted_code_k, distance_k)],
    1: [(predicted_name_1, predicted_code_1, distance_1), , ...(predicted_name_k, predicted_code_k, distance_k)],
    ...
}
```
Where distance specifies the distance in vector space from the query to the name, which is compliant to the internal company's taxonomy. The greater this value is - the more model thinks 2 names are similar.

### Limitations
Module is designed to work on files with ~200-300 entries. It is not designed to work on large files which have more than 5k entries (If this would be the use case - a need for `faiss` - library for efficient similarity search would arise)
