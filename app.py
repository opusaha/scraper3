"""
OMR Sheet Answer Detector API
Detects both filled and unfilled circles and determines which option is marked
Flask API version for web-based processing
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import io
import os
import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        print("No circles detected!")
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
        print("No circles found in grid!")
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

def create_visualization(img_rgb, all_circles, answers_dict, grid_bounds, wrong_answers=None, session_id=None):
    """
    Create debug visualization showing all circles and marked answers
    Mark wrong answers in red if provided
    """
    debug_img = img_rgb.copy()
    
    # Draw grid boundary
    gx1, gy1, gx2, gy2 = grid_bounds
    cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), (255, 255, 0), 2)
    
    # Create wrong question set and correct answer map for quick lookup
    wrong_questions = set()
    correct_answers_map = {}
    if wrong_answers:
        wrong_questions = {item['question'] for item in wrong_answers}
        correct_answers_map = {item['question']: item['correct'] for item in wrong_answers}
    
    # Group circles by question to identify which question each circle belongs to
    columns = group_circles_by_columns(all_circles)
    circle_to_question = {}
    circle_to_option = {}  # Maps circle to its option number (1-4)
    questions_per_column = 0
    
    for col_idx, column in enumerate(columns):
        column_sorted = sorted(column, key=lambda c: c['y'])
        question_groups = []
        current_group = [column_sorted[0]]
        
        for i in range(1, len(column_sorted)):
            if abs(column_sorted[i]['y'] - current_group[-1]['y']) < 30:
                current_group.append(column_sorted[i])
            else:
                current_group.sort(key=lambda c: c['x'])
                question_groups.append(current_group)
                current_group = [column_sorted[i]]
        
        current_group.sort(key=lambda c: c['x'])
        question_groups.append(current_group)
        
        if col_idx == 0:
            questions_per_column = len(question_groups)
        
        for q_idx, question_options in enumerate(question_groups):
            question_num = col_idx * questions_per_column + q_idx + 1
            for opt_idx, circle in enumerate(question_options):
                circle_key = (circle['x'], circle['y'])
                circle_to_question[circle_key] = question_num
                circle_to_option[circle_key] = opt_idx + 1  # Option number 1-4
    
    # Draw all circles
    for circle in all_circles:
        x, y, r = circle['x'], circle['y'], circle['r']
        circle_key = (x, y)
        q_num = circle_to_question.get(circle_key)
        opt_num = circle_to_option.get(circle_key)
        is_wrong_question = q_num in wrong_questions
        is_correct_option = is_wrong_question and opt_num == correct_answers_map.get(q_num)
        
        if circle['filled']:
            # Filled circle - red if wrong answer, green if correct
            color = (255, 0, 0) if is_wrong_question else (0, 255, 0)
            cv2.circle(debug_img, (x, y), r+3, color, 3)
        else:
            # Unfilled circle
            if is_correct_option:
                # Fill the correct answer circle in green for wrong questions
                cv2.circle(debug_img, (x, y), r, (0, 255, 0), -1)  # -1 fills the circle
                cv2.circle(debug_img, (x, y), r+2, (0, 255, 0), 2)  # Green border
            else:
                # Gray thin border for other unfilled circles
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
    if wrong_answers is not None:
        total = len(answers_dict)
        wrong_count = len(wrong_answers)
        right_count = total - wrong_count
        accuracy = (right_count / total * 100) if total > 0 else 0
        title = f"Detected: {len(answers_dict)} answers | Right: {right_count} | Wrong: {wrong_count} | Accuracy: {accuracy:.1f}%"
    else:
        title = f"Detected: {len(answers_dict)} answers"
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_id:
        output_file = f"{app.config['RESULT_FOLDER']}/omr_result_{session_id}_{timestamp}.png"
    else:
        output_file = f"{app.config['RESULT_FOLDER']}/omr_result_{timestamp}.png"
    
    plt.figure(figsize=(15, 28))
    plt.imshow(debug_img)
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file

def save_results(answers, session_id=None):
    """
    Save answers in JSON format
    """
    # Convert to string keys for JSON
    result = {str(q): opt for q, opt in sorted(answers.items())}
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_id:
        json_file = f"{app.config['RESULT_FOLDER']}/detected_answers_{session_id}_{timestamp}.json"
    else:
        json_file = f"{app.config['RESULT_FOLDER']}/detected_answers_{timestamp}.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return json_file, result

def compare_with_correct_answers(detected_answers, correct_answer_file='correct_answer.json'):
    """
    Compare detected answers with correct answers
    ক=1, খ=2, গ=3, ঘ=4
    """
    # Mapping
    bengali_to_num = {'ক': 1, 'খ': 2, 'গ': 3, 'ঘ': 4}
    num_to_bengali = {1: 'ক', 2: 'খ', 3: 'গ', 4: 'ঘ'}
    
    try:
        with open(correct_answer_file, 'r', encoding='utf-8') as f:
            correct_answers = json.load(f)
    except FileNotFoundError:
        print(f"\nCorrect answer file not found: {correct_answer_file}")
        return
    
    # Convert correct answers from Bengali to numbers
    correct_nums = {}
    for q, ans in correct_answers.items():
        if ans in bengali_to_num:
            correct_nums[int(q)] = bengali_to_num[ans]
    
    # Compare
    total = len(correct_nums)
    right = 0
    wrong = 0
    wrong_list = []
    
    print("\n" + "="*70)
    print("ANSWER COMPARISON REPORT")
    print("="*70)
    
    for q_num in sorted(correct_nums.keys()):
        detected = detected_answers.get(q_num, None)
        correct = correct_nums[q_num]
        
        if detected == correct:
            right += 1
        else:
            wrong += 1
            wrong_list.append({
                'question': q_num,
                'detected': detected,
                'correct': correct,
                'detected_bengali': num_to_bengali.get(detected, '?') if detected else '?',
                'correct_bengali': num_to_bengali[correct]
            })
    
    # Summary
    print(f"\nRight (সঠিক উত্তর): {right}/{total}")
    print(f"Wrong (ভুল উত্তর): {wrong}/{total}")
    accuracy = (right / total * 100) if total > 0 else 0
    print(f"Accuracy (নির্ভুলতা): {accuracy:.2f}%")
    
    # Show wrong answers
    if wrong_list:
        print(f"\n{'='*70}")
        print("Wrong Answers (ভুল উত্তরগুলো):")
        print("="*70)
        print(f"{'প্রশ্ন':<8} {'সনাক্তকৃত':<15} {'সঠিক উত্তর':<15}")
        print(f"{'Question':<8} {'Detected':<15} {'Correct':<15}")
        print("-"*70)
        
        for item in wrong_list:
            q = item['question']
            det = f"{item['detected']} ({item['detected_bengali']})" if item['detected'] else "Not detected"
            cor = f"{item['correct']} ({item['correct_bengali']})"
            print(f"{q:<8} {det:<15} {cor:<15}")
    
    print("="*70 + "\n")
    
    return right, wrong, wrong_list

# API Routes

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'OMR Sheet Answer Detector API',
        'version': '1.0',
        'endpoints': {
            'POST /upload': 'Upload OMR sheet image for processing',
            'GET /health': 'Health check endpoint',
            'GET /results/<session_id>': 'Get processing results by session ID'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })

@app.route('/upload', methods=['POST'])
def upload_and_process():
    """Upload OMR sheet image and process it"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'supported_types': list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        uploaded_filename = f"{session_id}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
        file.save(file_path)
        
        # Process the image
        result = detect_all_circles(file_path)
        
        if result is None or result[0] is None:
            # Clean up uploaded file
            os.remove(file_path)
            return jsonify({'error': 'Failed to detect circles in image'}), 400
        
        img_rgb, all_circles, filled_circles, grid_bounds = result
        
        # Determine answers
        answers, columns = determine_answers(all_circles)
        
        # Save results
        json_file, answers_data = save_results(answers, session_id)
        
        # Compare with correct answers if available
        comparison_result = None
        try:
            comparison_result = compare_with_correct_answers(answers, 'correct_answer.json')
        except Exception as e:
            # Comparison failed, but continue without it
            pass
        
        # Create visualization
        wrong_list = comparison_result[2] if comparison_result else None
        visualization_file = create_visualization(img_rgb, all_circles, answers, grid_bounds, wrong_list, session_id)
        
        # Prepare response
        response_data = {
            'session_id': session_id,
            'status': 'success',
            'total_questions': len(answers),
            'detected_answers': answers_data,
            'files': {
                'visualization': visualization_file,
                'json_results': json_file
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add comparison results if available
        if comparison_result:
            right, wrong, wrong_list = comparison_result
            accuracy = (right / (right + wrong) * 100) if (right + wrong) > 0 else 0
            
            response_data['comparison'] = {
                'total': right + wrong,
                'correct': right,
                'incorrect': wrong,
                'accuracy': round(accuracy, 2),
                'wrong_answers': wrong_list
            }
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify(response_data)
        
    except Exception as e:
        # Clean up uploaded file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500

@app.route('/results/<session_id>', methods=['GET'])
def get_results(session_id):
    """Get results by session ID"""
    try:
        # Look for result files with this session ID
        result_files = []
        for filename in os.listdir(app.config['RESULT_FOLDER']):
            if session_id in filename:
                result_files.append(filename)
        
        if not result_files:
            return jsonify({'error': 'No results found for this session ID'}), 404
        
        # Find JSON result file
        json_file = None
        visualization_file = None
        
        for filename in result_files:
            if filename.endswith('.json'):
                json_file = filename
            elif filename.endswith('.png'):
                visualization_file = filename
        
        response_data = {
            'session_id': session_id,
            'files': {}
        }
        
        # Load JSON results if available
        if json_file:
            json_path = os.path.join(app.config['RESULT_FOLDER'], json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            response_data['detected_answers'] = answers_data
            response_data['files']['json_results'] = json_file
        
        # Add visualization file if available
        if visualization_file:
            response_data['files']['visualization'] = visualization_file
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve results',
            'message': str(e)
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download result files"""
    try:
        file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to download file',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8080)
