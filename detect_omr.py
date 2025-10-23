"""
OMR Sheet Answer Detector
Automatically detects filled circles from OMR sheet image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def detect_omr_answers(image_path):
    """
    Detect answers from OMR sheet image
    """
    print("="*80)
    print("OMR Sheet Answer Detection")
    print("="*80)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    print(f"\nImage loaded: {width}x{height} pixels")
    
    # Answer grid boundaries (adjusted: 3 rows up, 1 row down)
    # Each row is approximately 50px, so: up = -150px, down = +50px
    grid_y_min = 900   # উপরে 3 লাইন বেশি (1050 - 150)
    grid_y_max = 2400  # নিচে 1 লাইন বেশি (2350 + 50)
    grid_x_min = 40
    grid_x_max = 1450
    
    print(f"Scanning answer grid area...")
    
    # Detect circles using HoughCircles
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
        return None
    
    circles = np.uint16(np.around(circles[0]))
    print(f"✓ Found {len(circles)} circles in image")
    
    # Filter filled circles in answer area
    filled_circles = []
    
    for circle in circles:
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        # Must be in answer grid
        if not (grid_y_min <= y <= grid_y_max and grid_x_min <= x <= grid_x_max):
            continue
        
        # Check if filled (dark)
        roi_size = r
        roi_y1 = max(0, y - roi_size)
        roi_y2 = min(height, y + roi_size)
        roi_x1 = max(0, x - roi_size)
        roi_x2 = min(width, x + roi_size)
        
        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size > 0:
            mean_intensity = np.mean(roi)
            
            # Only dark/filled circles
            if mean_intensity < 135:
                filled_circles.append({
                    'x': x,
                    'y': y,
                    'r': r,
                    'intensity': mean_intensity
                })
    
    print(f"✓ Detected {len(filled_circles)} filled circles")
    
    if len(filled_circles) == 0:
        print("❌ No filled circles found!")
        return None
    
    # Sort by Y position
    filled_circles.sort(key=lambda c: c['y'])
    
    # Get grid parameters
    all_y = [c['y'] for c in filled_circles]
    all_x = [c['x'] for c in filled_circles]
    
    y_min, y_max = min(all_y), max(all_y)
    x_min, x_max = min(all_x), max(all_x)
    
    grid_height = y_max - y_min
    grid_width = x_max - x_min
    
    # Expected: 25 rows per column, row height
    row_height = grid_height / 24  # 25 rows = 24 gaps
    column_width = grid_width / 3  # 4 columns = 3 gaps
    
    print(f"✓ Grid analysis complete")
    print(f"  - Y range: {y_min} to {y_max}")
    print(f"  - X range: {x_min} to {x_max}")
    print(f"  - Row height: {row_height:.1f}px")
    print(f"  - Column width: {column_width:.1f}px")
    
    # Map circles to questions
    print(f"\nMapping circles to questions...")
    
    answers = {}
    debug_info = []
    
    for circle in filled_circles:
        x, y = circle['x'], circle['y']
        
        # Determine row (0-24)
        row_index = int(round((y - y_min) / row_height))
        row_index = max(0, min(24, row_index))
        
        # Determine column (0-3)
        col_index = int(round((x - x_min) / column_width))
        col_index = max(0, min(3, col_index))
        
        # Question number
        question_num = col_index * 25 + row_index + 1
        
        if not (1 <= question_num <= 100):
            continue
        
        # Determine option (1-4) based on X position within column
        col_x_start = x_min + col_index * column_width
        col_x_end = x_min + (col_index + 1) * column_width if col_index < 3 else x_max
        col_width_actual = col_x_end - col_x_start
        
        x_in_col = x - col_x_start
        option_width = col_width_actual / 4 if col_width_actual > 0 else 100
        option = int(x_in_col / option_width) + 1 if option_width > 0 else 1
        option = max(1, min(4, option))
        
        # Keep darkest circle for each question
        if question_num not in answers or circle['intensity'] < answers[question_num]['intensity']:
            answers[question_num] = {
                'option': option,
                'intensity': circle['intensity'],
                'x': x,
                'y': y,
                'r': circle['r']
            }
            debug_info.append({
                'q': question_num,
                'opt': option,
                'x': x,
                'y': y,
                'col': col_index,
                'row': row_index
            })
    
    print(f"✓ Mapped {len(answers)} answers to questions")
    
    # Create debug visualization
    print(f"\nCreating debug visualization...")
    
    debug_img = img_rgb.copy()
    
    # Draw grid area
    cv2.rectangle(debug_img, (grid_x_min, grid_y_min), (grid_x_max, grid_y_max), 
                  (255, 255, 0), 2)
    
    # Draw all filled circles (gray)
    for circle in filled_circles:
        cv2.circle(debug_img, (circle['x'], circle['y']), 
                   circle['r'], (200, 200, 200), 1)
    
    # Highlight detected answers (green)
    for q_num in sorted(answers.keys()):
        ans = answers[q_num]
        x, y, r = ans['x'], ans['y'], ans['r']
        option = ans['option']
        
        # Draw circle
        cv2.circle(debug_img, (x, y), r+3, (0, 255, 0), 2)
        
        # Draw question number
        text = f"Q{q_num}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        text_x = x - text_size[0] // 2
        text_y = y - r - 5
        
        # Background for text
        cv2.rectangle(debug_img, 
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + 2),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(debug_img, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Add title
    title = f"OMR Detection: {len(answers)}/100 answers detected"
    cv2.putText(debug_img, title, (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Save debug image
    debug_filename = 'omr_detection_result.png'
    plt.figure(figsize=(15, 28))
    plt.imshow(debug_img)
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(debug_filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Debug image saved: {debug_filename}")
    
    return answers

def save_results(answers, output_file='detected_answers.txt'):
    """
    Save detected answers to file
    """
    if not answers:
        print("No answers to save!")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("OMR Sheet - Detected Answers\n")
        f.write("="*70 + "\n\n")
        
        for q in sorted(answers.keys()):
            f.write(f"প্রশ্ন {q:3d} - অপশন {answers[q]['option']}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write(f"Total: {len(answers)}/100\n")
        
        # Missing questions
        missing = [q for q in range(1, 101) if q not in answers]
        if missing:
            f.write(f"\nMissing ({len(missing)}): {', '.join(map(str, missing))}\n")
    
    print(f"✓ Results saved: {output_file}")
    
    # Also save CSV
    csv_file = 'detected_answers.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Question,Answer\n")
        for q in range(1, 101):
            ans = answers[q]['option'] if q in answers else ''
            f.write(f"{q},{ans}\n")
    
    print(f"✓ CSV saved: {csv_file}")

def print_results(answers):
    """
    Print results to console
    """
    if not answers:
        print("\n❌ No answers detected!")
        return
    
    print("\n" + "="*80)
    print("DETECTED ANSWERS")
    print("="*80 + "\n")
    
    # Print in 4 columns
    for col in range(4):
        print(f"\nColumn {col+1} (Questions {col*25+1}-{(col+1)*25}):")
        print("-"*60)
        
        for row in range(25):
            q = col * 25 + row + 1
            if q in answers:
                print(f"  প্রশ্ন {q:3d} → অপশন {answers[q]['option']}")
            else:
                print(f"  প্রশ্ন {q:3d} → [NOT DETECTED]")
    
    print("\n" + "="*80)
    print(f"Total: {len(answers)}/100 answers detected")
    
    missing = [q for q in range(1, 101) if q not in answers]
    if missing:
        print(f"Missing: {len(missing)} questions")
        if len(missing) <= 20:
            print(f"Missing questions: {', '.join(map(str, missing))}")
    
    print("="*80)

def main():
    """
    Main function
    """
    # Default image path
    image_path = 'nexesai.test_omr-sheet_16_answer.png'
    
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    print(f"\nProcessing: {image_path}\n")
    
    # Detect answers
    answers = detect_omr_answers(image_path)
    
    if answers:
        # Save results
        save_results(answers)
        
        # Print results
        print_results(answers)
        
        print("\n✅ Detection complete!")
        print("\nGenerated files:")
        print("  📄 detected_answers.txt  - Full results in text format")
        print("  📊 detected_answers.csv  - CSV format for Excel")
        print("  🖼️  omr_detection_result.png - Debug visualization")
    else:
        print("\n❌ Detection failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

