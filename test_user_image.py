import sys
from app.utils.omr_engine import process_omr_image

img_path = "uploads/d00b7e1e-ecf0-4438-9c41-2a56c0196b2a.png"
print(f"Testing image: {img_path}")
try:
    result = process_omr_image(img_path, num_questions=20)
    print("Success:", result.success)
    print("Error:", result.error_message)
    print("Answers:", result.answers)
    print("Roll:", result.roll_number)
    print("Set:", result.set_code)
except Exception as e:
    import traceback
    traceback.print_exc()
