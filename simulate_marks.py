import cv2
import numpy as np
from app.utils.omr_engine import process_omr_image, _preprocess_image, _find_corner_markers, _warp_perspective, SHEET_WIDTH, SHEET_HEIGHT

# Load blank
img = cv2.imread("test_omr.jpg")
blurred = _preprocess_image(img)
markers = _find_corner_markers(blurred)
warped = _warp_perspective(img, markers)

roll_x, roll_y = 378, 211
roll_region_w, roll_region_h = 674, 938
set_region_x, set_region_y = 1480, 211
set_region_w, set_region_h = 334, 935

# Fill Roll Number: let's try to fill digit 0=1, 1=2, 2=3, 3=4, 4=5, 5=5
num_digits = 6
cell_w = roll_region_w // num_digits
cell_h = roll_region_h // 10
test_roll = [1, 2, 3, 4, 5, 5]
for col, val in enumerate(test_roll):
    cx = roll_x + col * cell_w + cell_w // 2
    cy = roll_y + val * cell_h + cell_h // 2
    cv2.circle(warped, (cx, cy), 15, (0, 0, 0), -1)

# Fill Set Code: choose index 1 ("খ")
num_options_set = 4
cell_w_set = set_region_w // num_options_set
cell_h_set = roll_region_h
cx_set = set_region_x + 1 * cell_w_set + cell_w_set // 2
cy_set = roll_y + cell_h_set // 2
cv2.circle(warped, (cx_set, cy_set), 15, (0, 0, 0), -1)

# Fill Grid:
grid_top = 1524
grid_height = 1705
grid_left = 11
column_step = 622

total_questions = 100
q_per_col = 25

col_width = 593
q_block_h = grid_height // q_per_col
bubble_grid_w = col_width - int(30 * (2480 / 670.0))
opt_w_adj = bubble_grid_w // 4
opt_h = q_block_h

# Answer pattern: 0, 1, 2, 3 repeating
for q in range(total_questions):
    col_idx = q // q_per_col
    row_idx = q % q_per_col
    x_start = grid_left + col_idx * column_step
    bubble_grid_x = x_start + int(30 * (2480 / 670.0))
    
    ans = q % 4
    cx_ans = bubble_grid_x + ans * opt_w_adj + opt_w_adj // 2
    cy_ans = grid_top + row_idx * q_block_h + opt_h // 2
    cv2.circle(warped, (cx_ans, cy_ans), 15, (0, 0, 0), -1)

# save simulated marked image
cv2.imwrite("simulated_marked.jpg", warped)
print("Saved simulated_marked.jpg")
