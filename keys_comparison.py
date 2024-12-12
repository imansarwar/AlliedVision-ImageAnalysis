# import os
# import cv2
# import numpy as np
# import pytesseract
# from flask import Flask, render_template, request, jsonify
# from ultralytics import YOLO

# # Initialize Flask app
# app = Flask(__name__)
# app.config['STATIC_FOLDER'] = r'C:\Users\ImanOwais\source\repos\PYTHON\PythonAPIAspDotNet\static\KeyboardImages'
# os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# # Set up Tesseract OCR executable path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Initialize YOLOv8 model
# model = YOLO('yolov8n.pt')

# def detect_keyboard(image):
#     """Detects the keyboard in the image and returns the bounding box of the keyboard."""
#     results = model(image)
#     for result in results:
#         for box in result.boxes:
#             label = model.names[int(box.cls[0])]
#             if label == 'keyboard':  # Assuming 'keyboard' is the label for the keyboard
#                 x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
#                 return (x_min, y_min, x_max, y_max)
#     return None

# def detect_keys(image):
#     """Detects individual keys and extracts text using Tesseract OCR."""
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     key_data = []
#     for contour in contours:
#         x_min, y_min, w, h = cv2.boundingRect(contour)
#         if w > 15 and h > 15 and w < 150 and h < 150:  # Filter out small or large areas that are not likely to be keys
#             key_image = image[y_min:y_min+h, x_min:x_min+w]
#             key_text = pytesseract.image_to_string(key_image, config='--psm 10').strip()  # Single character mode
#             if key_text:  # Only add non-empty text boxes
#                 key_data.append({
#                     'bbox': (x_min, y_min, x_min + w, y_min + h),
#                     'text': key_text
#                 })
#     return key_data

# def compute_iou(bbox1, bbox2):
#     """Computes Intersection over Union (IoU) between two bounding boxes."""
#     x1_min, y1_min, x1_max, y1_max = bbox1
#     x2_min, y2_min, x2_max, y2_max = bbox2

#     x_left = max(x1_min, x2_min)
#     y_top = max(y1_min, y2_min)
#     x_right = min(x1_max, x2_max)
#     y_bottom = min(y1_max, y2_max)

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     intersection_area = (x_right - x_left) * (y_bottom - y_top)
#     bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
#     bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
#     union_area = bbox1_area + bbox2_area - intersection_area

#     return intersection_area / union_area

# def compare_keys(keys1, keys2):
#     """Compares keys from two images based on position and text."""
#     missing_keys = []
#     for key1 in keys1:
#         matched = False
#         for key2 in keys2:
#             iou = compute_iou(key1['bbox'], key2['bbox'])
#             if iou > 0.5 and key1['text'] == key2['text']:  # Match if IoU > 0.5 and text matches
#                 matched = True
#                 break
#         if not matched:
#             missing_keys.append(key1)
#     return missing_keys

# def check_key_order(keys1, keys2):
#     """Checks the order of keys in the test image (keys2) compared to the reference image (keys1)."""
#     reference_order = [key['text'] for key in keys1]
#     test_order = [key['text'] for key in keys2]

#     for i in range(len(reference_order) - 1):
#         if reference_order[i] in test_order:
#             ref_index = test_order.index(reference_order[i])
#             if ref_index + 1 < len(test_order) and test_order[ref_index + 1] != reference_order[i + 1]:
#                 print(f"Order error: After {reference_order[i]}, it should be {reference_order[i + 1]} but found {test_order[ref_index + 1]}.")
#         else:
#             print(f"Key {reference_order[i]} is missing in the test image.")


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/compare', methods=['POST'])
# def compare():
#     if 'image1' not in request.files or 'image2' not in request.files:
#         return jsonify({'error': 'Both image1 and image2 are required.'}), 400

#     image1 = request.files['image1']
#     image2 = request.files['image2']

#     # Save the uploaded images to the static folder
#     image1_path = os.path.join(app.config['STATIC_FOLDER'], image1.filename)
#     image2_path = os.path.join(app.config['STATIC_FOLDER'], image2.filename)
    
#     image1.save(image1_path)
#     image2.save(image2_path)

#     # Read the images with OpenCV
#     image1_cv = cv2.imread(image1_path)
#     image2_cv = cv2.imread(image2_path)

#     # Detect the keyboards in both images
#     keyboard_bbox1 = detect_keyboard(image1_cv)
#     keyboard_bbox2 = detect_keyboard(image2_cv)

#     if not keyboard_bbox1 and not keyboard_bbox2:
#      return jsonify({'error': 'Keyboard not detected in both images.'}), 400
#     elif not keyboard_bbox1:
#      return jsonify({'error': 'Keyboard not detected in the reference image (image1).'}), 400
#     elif not keyboard_bbox2:
#      return jsonify({'error': 'Keyboard not detected in the test image (image2).'}), 400


#     # Crop the keyboard regions from both images
#     x_min1, y_min1, x_max1, y_max1 = keyboard_bbox1
#     x_min2, y_min2, x_max2, y_max2 = keyboard_bbox2
#     cropped_image1 = image1_cv[y_min1:y_max1, x_min1:x_max1]
#     cropped_image2 = image2_cv[y_min2:y_max2, x_min2:x_max2]

#     # Detect keys in the cropped images
#     keys1 = detect_keys(cropped_image1)
#     keys2 = detect_keys(cropped_image2)

#     # Compare keys between the two images
#     missing_keys = compare_keys(keys1, keys2)

#     # Check the order of keys in the test image compared to the reference
#     check_key_order(keys1, keys2)

#     # Highlight missing keys in image2
#     for key in missing_keys:
#         x_min, y_min, x_max, y_max = key['bbox']
#         cv2.rectangle(image2_cv, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red rectangle

#     # Save the modified image2 with red bounding boxes
#     modified_image2_path = os.path.join(app.config['STATIC_FOLDER'], 'modified_image2.png')
#     cv2.imwrite(modified_image2_path, image2_cv)

#     # Output missing keys to the console
#     for key in missing_keys:
#         print(f"Missing key: {key['text']} at {key['bbox']}")

#     return jsonify({
#         'missing_keys': missing_keys,
#         'modified_image2_path': f"/static/KeyboardImages/modified_image2.png"  # Use relative path
#     })

# if __name__ == "__main__":
#     app.run(debug=True)



# import pytesseract
# import cv2
# from PIL import Image

# # Path to tesseract executable (you might need to specify the path if it's not in your PATH)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update path accordingly

# # Function to process image and extract key labels row by row
# def extract_keyboard_keys(image_path):
#     # Step 1: Read the image using OpenCV
#     img = cv2.imread(image_path)
    
#     # Step 2: Convert the image to grayscale (improves OCR accuracy)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Step 3: Apply thresholding to make the text more distinguishable
#     _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    
#     # Step 4: Use pytesseract to do OCR on the processed image
#     extracted_text = pytesseract.image_to_string(thresh_img, config='--psm 6')  # psm 6 is for block of text
    
#     # Step 5: Post-process the extracted text
#     # Split by lines and filter out empty lines
#     rows = [line.strip() for line in extracted_text.split("\n") if line.strip()]
    
#     # Step 6: Print the results in a format similar to your desired output
#     print("Extracted Key Labels:")
#     for i, row in enumerate(rows):
#         print(f"Row{i+1}: {row}")

# # Example usage
# image_path = r"C:\Users\ImanOwais\source\repos\PYTHON\PythonAPIAspDotNet\static\KeyboardImages\keyboardblack1.PNG"  # Path to your image
# extract_keyboard_keys(image_path)

import cv2
import pytesseract
import numpy as np

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess image for key detection."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return image, thresh

def detect_keys(thresh):
    """Detect keys using contours."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    key_bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 200 and 50 < h < 200:  # Filter based on size
            key_bounding_boxes.append((x, y, w, h))

    # Sort bounding boxes row-wise and left-to-right within rows
    key_bounding_boxes = sorted(key_bounding_boxes, key=lambda box: (box[1], box[0]))
    return key_bounding_boxes

def group_keys_into_rows(key_bounding_boxes, row_tolerance=30):
    """Group keys into rows based on their Y-coordinates."""
    rows = []
    current_row = []
    current_y = None

    for box in key_bounding_boxes:
        x, y, w, h = box
        if current_y is None or abs(y - current_y) < row_tolerance:
            current_row.append(box)
            current_y = y
        else:
            rows.append(current_row)
            current_row = [box]
            current_y = y

    if current_row:
        rows.append(current_row)

    # Sort each row by X-coordinate
    for row in rows:
        row.sort(key=lambda box: box[0])

    return rows

def extract_text_from_keys(image, rows):
    """Extract text from keys using OCR."""
    row_texts = []
    for row in rows:
        row_text = []
        for (x, y, w, h) in row:
            key_roi = image[y : y + h, x : x + w]
            key_text = pytesseract.image_to_string(
                key_roi, config="--oem 3 --psm 6 -l eng+swedish"
            ).strip()
            row_text.append(key_text if key_text else "[?]")  # Handle empty results
        row_texts.append(" ".join(row_text))
    return row_texts

def main(image_path):
    """Main function to process the keyboard image."""
    # Step 1: Preprocess the image
    original_image, thresh = preprocess_image(image_path)

    # Step 2: Detect keys
    key_bounding_boxes = detect_keys(thresh)

    # Step 3: Group keys into rows
    rows = group_keys_into_rows(key_bounding_boxes)

    # Step 4: Extract text from keys
    row_texts = extract_text_from_keys(original_image, rows)

    # Step 5: Display results row by row
    for i, row in enumerate(row_texts, start=1):
        print(f"Row {i}: {row}")

    # Debug: Draw bounding boxes and detected text
    for row in rows:
        for (x, y, w, h) in row:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            key_roi = original_image[y : y + h, x : x + w]
            key_text = pytesseract.image_to_string(
                key_roi, config="--oem 3 --psm 6 -l eng+swedish"
            ).strip()
            cv2.putText(
                original_image,
                key_text if key_text else "[?]",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

    # Show the image with bounding boxes
    cv2.imshow("Detected Keys", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the program
image_path = r"C:\Users\ImanOwais\source\repos\PYTHON\PythonAPIAspDotNet\static\KeyboardImages\4.jpg"
main(image_path)
