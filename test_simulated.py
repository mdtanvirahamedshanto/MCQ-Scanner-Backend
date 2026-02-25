import cv2
from app.utils.omr_engine import process_omr_image

if __name__ == "__main__":
    img_path = "test_omr.jpg"
    res = process_omr_image(img_path, num_questions=100)
    print("Success:", res.success)
    print("Error:", res.error_message)
    print("Roll:", res.roll_number)
    print("Set:", res.set_code)
    print("Answers:", res.answers)
    
    from app.utils.omr_engine import _detect_marked_option_by_density
    if getattr(_detect_marked_option_by_density, "debug_img", None) is not None:
        cv2.imwrite("debug_rois.jpg", _detect_marked_option_by_density.debug_img)
        print("Saved debug_rois.jpg")
