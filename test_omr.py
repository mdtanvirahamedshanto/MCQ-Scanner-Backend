import sys
from app.utils.omr_engine import process_omr_image

def run_test():
    try:
        res = process_omr_image("uploads/53a751cc-a4a4-40a8-9a85-0a8d0564dcdd.png", num_questions=60)
        print("Success:", res.success)
        print("Error:", res.error_message)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
