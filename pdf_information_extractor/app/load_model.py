from transformers import TableTransformerForObjectDetection
import easyocr

table_detector = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)
table_detector.save_pretrained("resources/table_detector")


table_structure_detector = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
)
table_structure_detector.save_pretrained("resources/table_structure_detector")

easyocr_reader = easyocr.Reader(['en'])