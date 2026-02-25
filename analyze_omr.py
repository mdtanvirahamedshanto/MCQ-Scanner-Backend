import fitz  # PyMuPDF
import cv2
import numpy as np
from app.utils.omr_engine import process_omr_image, _preprocess_image, _find_corner_markers

def convert_pdf_to_image(pdf_path, output_path):
    print(f"Opening {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc[0]
    # Render at 300 DPI for high quality
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    pix.save(output_path)
    print(f"Saved image to {output_path}")

def analyze_image(img_path):
    print("Reading image...")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image")
        return
    
    blurred = _preprocess_image(img)
    markers = _find_corner_markers(blurred)
    print(f"Markers found: {markers is not None}")
    if markers is not None:
        print("Marker coordinates:")
        for m in markers:
            print(m)
            cv2.circle(img, (int(m[0]), int(m[1])), 20, (0, 0, 255), -1)
        cv2.imwrite("test_omr_debug.jpg", img)
    else:
        print("Could not detect the 4 corner markers.")
    
    # Let's see what process_omr_image does
    res = process_omr_image(img_path, num_questions=100)
    print("Success:", res.success)
    print("Error:", res.error_message)
    print("Roll:", res.roll_number)
    print("Set:", res.set_code)
    print("Answers:", res.answers)
    
if __name__ == "__main__":
    pdf_path = "OMR_board_1771997571402.pdf"
    img_path = "test_omr.jpg"
    convert_pdf_to_image(pdf_path, img_path)
    analyze_image(img_path)
