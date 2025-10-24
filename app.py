"""
OMR Sheet Answer Detector
Detects both filled and unfilled circles and determines which option is marked
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

def detect_all_circles(image_path):
    """
    Detect both filled and unfilled circles from OMR sheet
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return None, None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    print(f"\nImage loaded: {width}x{height} pixels")
    
    # Answer grid boundaries
    grid_y_min = 900
    grid_y_max = 2400
    grid_x_min = 40
    grid_x_max = 1450
    
    print(f"Scanning area: Y({grid_y_min}-{grid_y_max}), X({grid_x_min}-{grid_x_max})")
    
    # Detect all circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=18,
        param1=50,
        param2=28,
        minRadius=7,
        maxRadius=26
    )
    
    if circles is None:
        print("❌ No circles detected!")
        return None, None, None
    
    circles = np.uint16(np.around(circles[0]))
    
    # Separate filled and unfilled circles
    all_circles = []
    filled_circles = []
    
    for circle in circles:
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        # Must be in answer grid
        if not (grid_y_min <= y <= grid_y_max and grid_x_min <= x <= grid_x_max):
            continue
        
        # Check if filled (dark)
        roi_y1 = max(0, y - r)
        roi_y2 = min(height, y + r)
        roi_x1 = max(0, x - r)
        roi_x2 = min(width, x + r)
        
        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size > 0:
            mean_intensity = np.mean(roi)
            
            circle_data = {
                'x': x,
                'y': y,
                'r': r,
                'intensity': mean_intensity,
                'filled': mean_intensity < 135
            }
            
            all_circles.append(circle_data)
            
            if circle_data['filled']:
                filled_circles.append(circle_data)
    
    if not all_circles:
        print("❌ No circles found in grid!")
        return None, None, None
    
    return img_rgb, all_circles, filled_circles, (grid_x_min, grid_y_min, grid_x_max, grid_y_max)

def group_circles_by_columns(circles):
    """
    Group circles by columns (X position)
    Each column contains multiple questions, each question has 4 options
    """
    if not circles:
        return []
    
    # Sort by X position to find columns
    circles_sorted = sorted(circles, key=lambda c: c['x'])
    
    # Find major X position gaps to identify columns
    x_positions = [c['x'] for c in circles_sorted]
    x_gaps = []
    for i in range(1, len(x_positions)):
        gap = x_positions[i] - x_positions[i-1]
        if gap > 150:  # Large gap = column boundary
            x_gaps.append((x_positions[i-1] + x_positions[i]) / 2)
    
    # Determine column boundaries
    column_boundaries = [0] + x_gaps + [2000]
    
    # Group circles into columns
    columns = []
    for i in range(len(column_boundaries) - 1):
        left = column_boundaries[i]
        right = column_boundaries[i + 1]
        
        column_circles = [c for c in circles if left <= c['x'] < right]
        if column_circles:
            # Sort by Y (top to bottom)
            column_circles.sort(key=lambda c: c['y'])
            columns.append(column_circles)
    
    return columns

def determine_answers(all_circles):
    """
    Determine which option is filled for each question
    Structure: 4 columns, each column has multiple questions (top to bottom)
    Each question has 4 options (left to right)
    """
    columns = group_circles_by_columns(all_circles)
    
    answers = {}
    questions_per_column = 0
    
    for col_idx, column in enumerate(columns):
        
        # Group circles in this column by Y position (each group = 1 question with 4 options)
        column_sorted = sorted(column, key=lambda c: c['y'])
        
        question_groups = []
        current_group = [column_sorted[0]]
        
        for i in range(1, len(column_sorted)):
            # If Y difference is small (<30px), same question
            if abs(column_sorted[i]['y'] - current_group[-1]['y']) < 30:
                current_group.append(column_sorted[i])
            else:
                # New question
                current_group.sort(key=lambda c: c['x'])  # Sort options left to right
                question_groups.append(current_group)
                current_group = [column_sorted[i]]
        
        # Add last group
        current_group.sort(key=lambda c: c['x'])
        question_groups.append(current_group)
        
        if col_idx == 0:
            questions_per_column = len(question_groups)
        
        # Process each question in this column
        for q_idx, question_options in enumerate(question_groups):
            question_num = col_idx * questions_per_column + q_idx + 1
            
            # Find which option is filled
            for opt_idx, circle in enumerate(question_options):
                option_num = opt_idx + 1
                
                if circle['filled']:
                    answers[question_num] = option_num
                    break
    
    return answers, columns

def create_visualization(img_rgb, all_circles, answers_dict, grid_bounds):
    """
    Create debug visualization showing all circles and marked answers
    """
    debug_img = img_rgb.copy()
    
    # Draw grid boundary
    gx1, gy1, gx2, gy2 = grid_bounds
    cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), (255, 255, 0), 2)
    
    # Draw all circles
    for circle in all_circles:
        x, y, r = circle['x'], circle['y'], circle['r']
        
        if circle['filled']:
            # Filled circle - green with thick border
            cv2.circle(debug_img, (x, y), r+3, (0, 255, 0), 3)
        else:
            # Unfilled circle - gray thin border
            cv2.circle(debug_img, (x, y), r+1, (150, 150, 150), 1)
    
    # Add question numbers (will be done per question group)
    # Group by Y position to show question numbers
    all_sorted = sorted(all_circles, key=lambda c: c['y'])
    question_rows = []
    current_row = [all_sorted[0]]
    
    for i in range(1, len(all_sorted)):
        if abs(all_sorted[i]['y'] - current_row[-1]['y']) < 30:
            current_row.append(all_sorted[i])
        else:
            question_rows.append(current_row)
            current_row = [all_sorted[i]]
    question_rows.append(current_row)
    
    # Title
    title = f"Detected: {len(answers_dict)} answers"
    
    # Save
    output_file = 'omr_detection_result.png'
    plt.figure(figsize=(15, 28))
    plt.imshow(debug_img)
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_file}")
    return output_file

def save_results(answers):
    """
    Save answers in JSON format
    """
    # Convert to string keys for JSON
    result = {str(q): opt for q, opt in sorted(answers.items())}
    
    # Save to JSON file
    json_file = 'detected_answers.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved: {json_file} ({len(result)} answers)")

def main():
    """Main function"""
    # Default image
    image_path = 'nexesai.test_omr-sheet_15_answer.png'
    
    # Check for command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    print(f"\nProcessing: {image_path}\n")
    
    # Detect circles
    result = detect_all_circles(image_path)
    
    if result is None or result[0] is None:
        print("\n❌ Detection failed!")
        return 1
    
    img_rgb, all_circles, filled_circles, grid_bounds = result
    
    # Determine answers
    answers, rows = determine_answers(all_circles)
    
    # Create visualization
    create_visualization(img_rgb, all_circles, answers, grid_bounds)
    
    # Save results
    save_results(answers)
    
    # Count total questions
    columns = group_circles_by_columns(all_circles)
    total_questions = 0
    for col in columns:
        col_sorted = sorted(col, key=lambda c: c['y'])
        q_groups = []
        curr = [col_sorted[0]]
        for i in range(1, len(col_sorted)):
            if abs(col_sorted[i]['y'] - curr[-1]['y']) < 30:
                curr.append(col_sorted[i])
            else:
                q_groups.append(curr)
                curr = [col_sorted[i]]
        q_groups.append(curr)
        total_questions += len(q_groups)
    
    return 0

if __name__ == "__main__":
    exit(main())
