import cv2
from app.utils.omr_engine import process_omr_image

if __name__ == "__main__":
    img_path = "simulated_marked.jpg"
    res = process_omr_image(img_path, num_questions=100)
    print("Success:", res.success)
    print("Error:", res.error_message)
    print("Roll:", res.roll_number)
    print("Set:", res.set_code)
    print("Answers:", res.answers)
