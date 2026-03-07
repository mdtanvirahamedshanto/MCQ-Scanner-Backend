import cv2
from app.utils.omr_engine import process_omr_image

def test():
    image_path = "mcq.png"
    result = process_omr_image(image_path, num_questions=20, template_type="20q_mcq_png")
    
    print("Success:", result.success)
    if not result.success:
        print("Error:", result.error_message)
    else:
        print("Answers:")
        for idx, ans in enumerate(result.answers):
            print(f"Q{idx+1}: {ans}")
            
if __name__ == "__main__":
    test()
