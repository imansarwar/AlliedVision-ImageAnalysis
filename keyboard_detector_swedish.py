# import sys
# import threading
# import time
# import cv2
# from flask import Flask, jsonify, send_file, Response, request, render_template, send_from_directory
# from queue import Queue
# from vmbpy import *
# from ultralytics import YOLO


# # Flask app initialization
# app = Flask(__name__)

# opencv_display_format = PixelFormat.Rgb8
# camera = None
# handler = None
# display_queue = Queue(50)  # Queue to hold frames for display
# last_frame = None

# # Define paths for keyboard images
# KEYBOARD_IMAGES = {
#     "swedish": "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\keyboard2.png",
#     "standard": "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\StandardKeyboardAV.jpg",
# }

# selected_keyboard = None  # Track which keyboard is selected

# @app.route('/detect-keyboard', methods=['GET'])
# def detect_keyboard():
#     try:
#         # Define paths
#         image_dir = "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\"
#         input_filename = "captured_keyboard.jpg"
#         output_filename = "detected_keyboard.jpg"
#         input_path = f"{image_dir}{input_filename}"
#         output_path = f"{image_dir}{output_filename}"

#         # Load the YOLOv8 model
#         model = YOLO('yolov8n.pt')  # Ensure yolov8n.pt is downloaded in the current directory

#         # Read the captured image
#         image = cv2.imread(input_path)
#         if image is None:
#             return jsonify({"error": "Captured image not found"}), 404

#         # Run the YOLO model on the image
#         results = model(image)

#         # Process detections
#         detected = False
#         for result in results:
#             for box in result.boxes:
#                 label = model.names[int(box.cls)]  # Get the class name
#                 conf = box.conf.item()  # Confidence score
#                 if label == "keyboard" and conf > 0.5:  # Adjust confidence threshold if needed
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     # Draw the bounding box
#                     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(image, f"Keyboard {conf:.2f}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     detected = True

#         if detected:
#             print("Keyboard detected and bounding box drawn.")
#         else:
#             print("No keyboard detected in the image.")

            
#         # Save the annotated image
#         cv2.imwrite(output_path, image)

#         # Return success response with the path to the detected image
#         return jsonify({"message": "Keyboard detection completed", "detected_image": output_filename}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Function to get camera
# def get_camera(camera_id=None):
#     """Get the camera instance."""
#     with VmbSystem.get_instance() as vmb:
#         if camera_id:
#             try:
#                 return vmb.get_camera_by_id(camera_id)
#             except VmbCameraError:
#                 sys.exit(f"Failed to access Camera '{camera_id}'. Abort.")
#         else:
#             cameras = vmb.get_all_cameras()
#             if not cameras:
#                 sys.exit("No Cameras accessible. Abort.")
#             return cameras[0]

# # Function to set up camera
# def setup_camera(cam):
#     """Set up the camera's basic settings."""
#     with cam:
#         try:
#             cam.ExposureAuto.set("Continuous")
#         except (AttributeError, VmbFeatureError):
#             pass

#         try:
#             cam.BalanceWhiteAuto.set("Continuous")
#         except (AttributeError, VmbFeatureError):
#             pass

#         try:
#             stream = cam.get_streams()[0]
#             print("Stream initialized with stream ID:", stream.get_id())
#             stream.GVSPAdjustPacketSize.run()
#             while not stream.GVSPAdjustPacketSize.is_done():
#                 pass
#         except (AttributeError, VmbFeatureError):
#             pass

# # Function to set pixel format
# def setup_pixel_format(cam):
#     """Set up the pixel format for the camera."""
#     with cam:
#         cam_formats = cam.get_pixel_formats()
#         if opencv_display_format in cam_formats:
#             cam.set_pixel_format(opencv_display_format)
#         else:
#             sys.exit("Camera does not support an OpenCV compatible format. Abort.")

# # Frame handler class
# class Handler:
#     """Handles frames from the camera and pushes them into a queue."""
#     def __call__(self, cam, stream, frame):
#         try:
#             print(f"Frame received: ID={frame.get_id()}, Status={frame.get_status()}")
#             if frame.get_status() == FrameStatus.Complete:
#                 if frame.get_pixel_format() == opencv_display_format:
#                     display = frame
#                 else:
#                     display = frame.convert_pixel_format(opencv_display_format)

#                 if not display_queue.full():
#                     display_queue.put(display.as_opencv_image(), timeout=1)
#                     print("Frame queued successfully.")
#                 else:
#                     print("Queue is full. Frame dropped.")
#             else:
#                 print(f"Frame skipped: Status={frame.get_status()}")

#             cam.queue_frame(frame)
#         except Exception as e:
#             print(f"Error in Handler: {e}")

# def start_camera():
#     global camera, last_frame
#     with VmbSystem.get_instance():
#         camera = get_camera(None)  # Use the first camera
#         with camera:
#             setup_camera(camera)
#             setup_pixel_format(camera)

#             while True:
#                 try:
#                     # Attempt to capture a frame with a reasonable timeout
#                     frame = camera.get_frame(timeout_ms=10000)  # Timeout in case the camera is slow
#                     if frame.get_status() == FrameStatus.Complete:
#                         # Convert the frame to the desired pixel format
#                         img = frame.convert_pixel_format(PixelFormat.Bgr8).as_opencv_image()

#                         # Convert the image to grayscale
#                         grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                         last_frame = grayscale_img  # Save the last frame

#                         # Put the grayscale frame into the display queue
#                         if not display_queue.full():
#                             display_queue.put(grayscale_img, timeout=1)
#                             print("Grayscale frame manually queued.")
#                         else:
#                             print("Queue is full. Frame dropped.")
#                     else:
#                         print(f"Frame capture failed with status: {frame.get_status()}")
                    
#                     retry_delay = 0.5  # Reset delay after success

#                     # Requeue the frame for future use
#                     camera.queue_frame(frame)
                    

#                 except VmbTimeout:
#                     print("Frame capture timed out. Retrying...")  # Log timeout events
#                     time.sleep(retry_delay)
#                     retry_delay = min(retry_delay * 2, 5)  # Double delay, max out at 5 seconds
#                 except Exception as e:
#                     print(f"Error during frame capture: {e}")

#                 # Debug print to check the queue status
#                 print(f"Queue size: {display_queue.qsize()}")

#                 # Sleep briefly to avoid overwhelming the system
#                 time.sleep(0.2)  # Increased delay to manage slower frame capture


# # Route to serve the HTML page
# @app.route('/')
# def index():
#     """Serve the index page."""
#     return render_template('index.html')

# # Route for streaming
# @app.route('/stream', methods=['GET'])
# def stream():
#     """Stream the camera feed."""
#     def generate():
#         while True:
#             try:
#                 if display_queue.empty():
#                     time.sleep(0.1)  # Avoid busy-waiting
#                     continue

#                 frame = display_queue.get(timeout=1)  # Get frame with timeout to handle slow queues

#                 # Check if the frame is grayscale or color
#                 if len(frame.shape) == 2:  # Grayscale image
#                     ret, jpeg = cv2.imencode('.jpg', frame)
#                 else:  # Color image
#                     ret, jpeg = cv2.imencode('.jpg', frame)

#                 # if display_queue.empty() and last_frame is not None:
#                 #   yield (b'--frame\r\n'
#                 #                      b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n\r\n')
#                 if ret:
#                     yield (b'--frame\r\n'
#                         b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

#             except Exception as e:
#                 print(f"Error during streaming: {e}")
#                 break

#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Route to select keyboard
# @app.route('/select-keyboard', methods=['POST'])
# def select_keyboard():
#     """Select a keyboard layout."""
#     global selected_keyboard
#     data = request.json
#     keyboard_type = data.get("keyboard_type")
#     if keyboard_type not in KEYBOARD_IMAGES:
#         return jsonify({"error": "Invalid keyboard type"}), 400

#     selected_keyboard = keyboard_type  # Store just the keyboard type

#     # Determine image path and keyboard name based on selected keyboard
#     if keyboard_type == 'swedish':
#         selected_keyboard_name = 'Swedish Keyboard'
#         image_path = '/static/KeyboardImages/keyboard2.png'  # Relative path
#     else:
#         selected_keyboard_name = 'Standard Keyboard'
#         image_path = '/static/KeyboardImages/StandardKeyboardAV.jpg'  # Relative path

#     return jsonify({
#         "message": f"Keyboard '{keyboard_type}' selected",
#         "image_path": image_path,
#         "keyboard_name": selected_keyboard_name
#     }), 200


# # Route to capture image and compare
# @app.route('/capture-and-compare', methods=['GET'])
# def capture_and_compare():
#     global selected_keyboard, last_frame

#     try:
#         # Check if a keyboard is selected
#         if not selected_keyboard:
#             return jsonify({"error": "No keyboard selected"}), 400

#         # Retrieve the current or last available frame
#         if display_queue.empty():
#             if last_frame is None:
#                 return jsonify({"error": "No frame available, and no last frame to use"}), 500
#             frame = last_frame  # Use the last captured frame
#             print("Using the last available frame.")
#         else:
#             frame = display_queue.get(True)
#             last_frame = frame  # Update the last captured frame

#         # Define the directory and file name for the captured frame
#         keyboard_image_dir = "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\"
#         captured_filename = "captured_keyboard.jpg"
#         full_path = f"{keyboard_image_dir}{captured_filename}"

#         # Save the captured frame
#         cv2.imwrite(full_path, frame)

#         return jsonify({"message": "Frame captured successfully", "filename": captured_filename}), 200

#     except Exception as e:
#         # Log the error and return a JSON response
#         print(f"Error in capture-and-compare: {e}")
#         return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

    
# @app.route('/static/keyboard-images/<filename>')
# def serve_keyboard_image(filename):
#     """Serve static files from the KeyboardImages directory."""
#     keyboard_image_dir = "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\"
#     try:
#         return send_from_directory(keyboard_image_dir, filename)
#     except FileNotFoundError:
#         return jsonify({"error": "File not found"}), 404


# # Start Flask app
# if __name__ == '__main__':
#     threading.Thread(target=start_camera, daemon=True).start()
#     app.run(host='0.0.0.0', port=5001)



import sys
import threading
import time
import cv2
from flask import Flask, jsonify, send_file, Response, request, render_template, send_from_directory
from queue import Queue
from vmbpy import *
from ultralytics import YOLO


# Flask app initialization
app = Flask(__name__)

opencv_display_format = PixelFormat.Rgb8
camera = None
handler = None
display_queue = Queue(50)  # Queue to hold frames for display
last_frame = None

# Define paths for keyboard images
KEYBOARD_IMAGES = {
    "swedish": "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\keyboard2.png",
    "standard": "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\StandardKeyboardAV.jpg",
}

selected_keyboard = None  # Track which keyboard is selected

# Route to detect keyboard and compare with selected keyboard
@app.route('/detect-keyboard', methods=['GET'])
def detect_keyboard():
    try:
        # Define paths
        image_dir = "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\"
        input_filename = "captured_keyboard.jpg"
        output_filename = "detected_keyboard.jpg"
        input_path = f"{image_dir}{input_filename}"
        output_path = f"{image_dir}{output_filename}"

        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')  # Ensure yolov8n.pt is downloaded in the current directory

        # Read the captured image
        image = cv2.imread(input_path)
        if image is None:
            return jsonify({"error": "Captured image not found"}), 404

        # Run the YOLO model on the image
        results = model(image)

        # Initialize detection status
        detected = False
        detected_label = None  # Store detected label (keyboard type)
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]  # Get the class name
                conf = box.conf.item()  # Confidence score
                if label == "keyboard" and conf > 0.5:  # Adjust confidence threshold if needed
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"Keyboard {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected = True
                    detected_label = label  # Store the detected keyboard label

        if detected:
            print("Keyboard detected and bounding box drawn.")
        else:
            print("No keyboard detected in the image.")

        # Compare the detected keyboard with the selected keyboard
        comparison_result = "No match"
        if detected_label:
            if selected_keyboard and selected_keyboard.lower() in detected_label.lower():
                comparison_result = "Match"
            else:
                comparison_result = "No match"

        # Save the annotated image
        cv2.imwrite(output_path, image)

        # Return success response with the path to the detected image and comparison result
        return jsonify({
            "message": "Keyboard detection completed",
            "detected_image": output_filename,
            "comparison_result": comparison_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to get camera
def get_camera(camera_id=None):
    """Get the camera instance."""
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                sys.exit(f"Failed to access Camera '{camera_id}'. Abort.")
        else:
            cameras = vmb.get_all_cameras()
            if not cameras:
                sys.exit("No Cameras accessible. Abort.")
            return cameras[0]

# Function to set up camera
def setup_camera(cam):
    """Set up the camera's basic settings."""
    with cam:
        try:
            cam.ExposureAuto.set("Continuous")
        except (AttributeError, VmbFeatureError):
            pass

        try:
            cam.BalanceWhiteAuto.set("Continuous")
        except (AttributeError, VmbFeatureError):
            pass

        try:
            stream = cam.get_streams()[0]
            print("Stream initialized with stream ID:", stream.get_id())
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VmbFeatureError):
            pass

# Function to set pixel format
def setup_pixel_format(cam):
    """Set up the pixel format for the camera."""
    with cam:
        cam_formats = cam.get_pixel_formats()
        if opencv_display_format in cam_formats:
            cam.set_pixel_format(opencv_display_format)
        else:
            sys.exit("Camera does not support an OpenCV compatible format. Abort.")

# Frame handler class
class Handler:
    """Handles frames from the camera and pushes them into a queue."""
    def __call__(self, cam, stream, frame):
        try:
            print(f"Frame received: ID={frame.get_id()}, Status={frame.get_status()}")
            if frame.get_status() == FrameStatus.Complete:
                if frame.get_pixel_format() == opencv_display_format:
                    display = frame
                else:
                    display = frame.convert_pixel_format(opencv_display_format)

                if not display_queue.full():
                    display_queue.put(display.as_opencv_image(), timeout=1)
                    print("Frame queued successfully.")
                else:
                    print("Queue is full. Frame dropped.")
            else:
                print(f"Frame skipped: Status={frame.get_status()}")

            cam.queue_frame(frame)
        except Exception as e:
            print(f"Error in Handler: {e}")

def start_camera():
    global camera, last_frame
    with VmbSystem.get_instance():
        camera = get_camera(None)  # Use the first camera
        with camera:
            setup_camera(camera)
            setup_pixel_format(camera)

            while True:
                try:
                    # Attempt to capture a frame with a reasonable timeout
                    frame = camera.get_frame(timeout_ms=10000)  # Timeout in case the camera is slow
                    if frame.get_status() == FrameStatus.Complete:
                        # Convert the frame to the desired pixel format
                        img = frame.convert_pixel_format(PixelFormat.Bgr8).as_opencv_image()

                        # Convert the image to grayscale
                        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        last_frame = grayscale_img  # Save the last frame

                        # Put the grayscale frame into the display queue
                        if not display_queue.full():
                            display_queue.put(grayscale_img, timeout=1)
                            print("Grayscale frame manually queued.")
                        else:
                            print("Queue is full. Frame dropped.")
                    else:
                        print(f"Frame capture failed with status: {frame.get_status()}")
                    
                    retry_delay = 0.5  # Reset delay after success

                    # Requeue the frame for future use
                    camera.queue_frame(frame)
                    

                except VmbTimeout:
                    print("Frame capture timed out. Retrying...")  # Log timeout events
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 5)  # Double delay, max out at 5 seconds
                except Exception as e:
                    print(f"Error during frame capture: {e}")

                # Debug print to check the queue status
                print(f"Queue size: {display_queue.qsize()}")

                # Sleep briefly to avoid overwhelming the system
                time.sleep(0.2)  # Increased delay to manage slower frame capture


# Route to serve the HTML page
@app.route('/')
def index():
    """Serve the index page."""
    return render_template('index.html')

# Route for streaming
@app.route('/stream', methods=['GET'])
def stream():
    """Stream the camera feed."""
    def generate():
        while True:
            try:
                if display_queue.empty():
                    time.sleep(0.1)  # Avoid busy-waiting
                    continue

                frame = display_queue.get(timeout=1)  # Get frame with timeout to handle slow queues

                # Check if the frame is grayscale or color
                if len(frame.shape) == 2:  # Grayscale image
                    ret, jpeg = cv2.imencode('.jpg', frame)
                else:  # Color image
                    ret, jpeg = cv2.imencode('.jpg', frame)

                # if display_queue.empty() and last_frame is not None:
                #   yield (b'--frame\r\n'
                #                      b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n\r\n')
                if ret:
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            except Exception as e:
                print(f"Error during streaming: {e}")
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to select keyboard
@app.route('/select-keyboard', methods=['POST'])
def select_keyboard():
    """Select a keyboard layout."""
    global selected_keyboard
    data = request.json
    keyboard_type = data.get("keyboard_type")
    if keyboard_type not in KEYBOARD_IMAGES:
        return jsonify({"error": "Invalid keyboard type"}), 400

    selected_keyboard = keyboard_type  # Store just the keyboard type

    # Determine image path and keyboard name based on selected keyboard
    if keyboard_type == 'swedish':
        selected_keyboard_name = 'Swedish Keyboard'
        image_path = '/static/KeyboardImages/keyboard2.png'  # Relative path
    else:
        selected_keyboard_name = 'Standard Keyboard'
        image_path = '/static/KeyboardImages/StandardKeyboardAV.jpg'  # Relative path

    return jsonify({
        "message": f"Keyboard '{keyboard_type}' selected",
        "image_path": image_path,
        "keyboard_name": selected_keyboard_name
    }), 200


# Route to capture image and compare
@app.route('/capture-and-compare', methods=['GET'])
def capture_and_compare():
    global selected_keyboard, last_frame

    try:
        # Check if a keyboard is selected
        if not selected_keyboard:
            return jsonify({"error": "No keyboard selected"}), 400

        # Retrieve the current or last available frame
        if display_queue.empty():
            if last_frame is None:
                return jsonify({"error": "No frame available, and no last frame to use"}), 500
            frame = last_frame  # Use the last captured frame
            print("Using the last available frame.")
        else:
            frame = display_queue.get(True)
            last_frame = frame  # Update the last captured frame

        # Define the directory and file name for the captured frame
        keyboard_image_dir = "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\"
        captured_filename = "captured_keyboard.jpg"
        full_path = f"{keyboard_image_dir}{captured_filename}"

        # Save the captured frame
        cv2.imwrite(full_path, frame)

        return jsonify({"message": "Frame captured successfully", "filename": captured_filename}), 200

    except Exception as e:
        # Log the error and return a JSON response
        print(f"Error in capture-and-compare: {e}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

    
@app.route('/static/keyboard-images/<filename>')
def serve_keyboard_image(filename):
    """Serve static files from the KeyboardImages directory."""
    keyboard_image_dir = "C:\\Users\\ImanOwais\\source\\repos\\PYTHON\\PythonAPIAspDotNet\\static\\KeyboardImages\\"
    try:
        return send_from_directory(keyboard_image_dir, filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


# Start Flask app
if __name__ == '__main__':
    threading.Thread(target=start_camera, daemon=True).start()
    app.run(host='0.0.0.0', port=5001)







