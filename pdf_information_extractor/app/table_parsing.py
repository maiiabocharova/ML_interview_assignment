from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
import torch
from PIL import Image, ImageDraw

table_detector = TableTransformerForObjectDetection.from_pretrained(
    "resources/table_detector"
)

table_structure_detector = TableTransformerForObjectDetection.from_pretrained(
    "resources/table_structure_detector"
)

feature_extractor = DetrFeatureExtractor()


def get_rows(img):
    encoding = feature_extractor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = table_structure_detector(**encoding)
    target_sizes = [img.size[::-1]]
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=0.95, target_sizes=target_sizes
    )[0]
    return results['boxes'].tolist()


def visualize_table(pdf_doc, page_num):
    page = pdf_doc[int(page_num)]
    pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(image, "RGBA")

    # detect tables
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = table_detector(**encoding)
    width, height = image.size
    predictions = feature_extractor.post_process_object_detection(
        outputs, threshold=0.7, target_sizes=[(height, width)]
    )[0]

    for pred in predictions['boxes']:
        draw.rectangle(pred.tolist(), outline='blue', width=2)

    # detect rows
    for pred in predictions['boxes']:
        table_coords = []
        adjusted_corrds = []
        for i, el in enumerate(pred.tolist()):
            if i == 0:
                table_coords.append(max(0, el - 10))
                adjusted_corrds.append(max(0, el - 10))
            elif i == 1:
                table_coords.append(max(0, el - 15))
                adjusted_corrds.append(max(0, el - 15))
            elif i == 2:
                table_coords.append(min(image.size[0], el + 10))
            else:
                table_coords.append(min(image.size[1], el + 15))
        table_image = image.crop(table_coords)
        rows_coords = get_rows(table_image)
        for row in rows_coords:
            row_coords = [el + adjusted_corrds[i % 2] for i, el in enumerate(row)]
            draw.rectangle(row_coords, outline='red', width=1)
    return image