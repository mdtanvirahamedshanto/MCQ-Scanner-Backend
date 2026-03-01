import cv2
import easyocr

def test_ocr(img_path):
    print(f"\n--- Analyzing {img_path} ---")
    img = cv2.imread(img_path)
    
    # We only care about the top half to speed up OCR
    h, w = img.shape[:2]
    top_half = img[0:int(h * 0.5), :]
    
    # Initialize reader (try English and Bengali if possible, otherwise English is fine for numbers/english text)
    # The sheet might have "Roll" or some bengali text
    try:
        reader = easyocr.Reader(['bn', 'en'])
        results = reader.readtext(top_half)
    except Exception as e:
        print(f"OCR Error: {e}")
        return

    print("OCR Results:")
    found = False
    for (bbox, text, prob) in results:
        # Check if it looks like Roll or Set Code
        text_lower = text.lower()
        if 'roll' in text_lower or 'set' in text_lower or 'code' in text_lower or 'রোল' in text_lower or 'সেট' in text_lower:
            print(f"  Found '{text}' at {bbox} with prob {prob:.2f}")
            found = True
            
    if not found:
        print("  No relevant text found.")
        # Just print top 5 texts found to see what's there
        for (bbox, text, prob) in results[:5]:
            print(f"  Sample: '{text}' at {bbox}")

for img in ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']:
    test_ocr(img)
