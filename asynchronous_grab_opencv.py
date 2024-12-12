# import sys
# from typing import Optional
# from queue import Queue

# from vmbpy import *


# # All frames will either be recorded in this format, or transformed to it before being displayed
# opencv_display_format = PixelFormat.Bgr8


# def print_preamble():
#     print('///////////////////////////////////////////////////')
#     print('/// VmbPy Asynchronous Grab with OpenCV Example ///')
#     print('///////////////////////////////////////////////////\n')


# def print_usage():
#     print('Usage:')
#     print('    python asynchronous_grab_opencv.py [camera_id]')
#     print('    python asynchronous_grab_opencv.py [/h] [-h]')
#     print()
#     print('Parameters:')
#     print('    camera_id   ID of the camera to use (using first camera if not specified)')
#     print()


# def abort(reason: str, return_code: int = 1, usage: bool = False):
#     print(reason + '\n')

#     if usage:
#         print_usage()

#     sys.exit(return_code)


# def parse_args() -> Optional[str]:
#     args = sys.argv[1:]
#     argc = len(args)

#     for arg in args:
#         if arg in ('/h', '-h'):
#             print_usage()
#             sys.exit(0)

#     if argc > 1:
#         abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

#     return None if argc == 0 else args[0]


# def get_camera(camera_id: Optional[str]) -> Camera:
#     with VmbSystem.get_instance() as vmb:
#         if camera_id:
#             try:
#                 return vmb.get_camera_by_id(camera_id)

#             except VmbCameraError:
#                 abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

#         else:
#             cams = vmb.get_all_cameras()
#             if not cams:
#                 abort('No Cameras accessible. Abort.')

#             return cams[0]


# def setup_camera(cam: Camera):
#     with cam:
#         # Enable auto exposure time setting if camera supports it
#         try:
#             cam.ExposureAuto.set('Continuous')

#         except (AttributeError, VmbFeatureError):
#             pass

#         # Enable white balancing if camera supports it
#         try:
#             cam.BalanceWhiteAuto.set('Continuous')

#         except (AttributeError, VmbFeatureError):
#             pass

#         # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
#         try:
#             stream = cam.get_streams()[0]
#             stream.GVSPAdjustPacketSize.run()
#             while not stream.GVSPAdjustPacketSize.is_done():
#                 pass

#         except (AttributeError, VmbFeatureError):
#             pass


# def setup_pixel_format(cam: Camera):
#     # Query available pixel formats. Prefer color formats over monochrome formats
#     cam_formats = cam.get_pixel_formats()
#     cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
#     convertible_color_formats = tuple(f for f in cam_color_formats
#                                       if opencv_display_format in f.get_convertible_formats())

#     cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
#     convertible_mono_formats = tuple(f for f in cam_mono_formats
#                                      if opencv_display_format in f.get_convertible_formats())

#     # if OpenCV compatible color format is supported directly, use that
#     if opencv_display_format in cam_formats:
#         cam.set_pixel_format(opencv_display_format)

#     # else if existing color format can be converted to OpenCV format do that
#     elif convertible_color_formats:
#         cam.set_pixel_format(convertible_color_formats[0])

#     # fall back to a mono format that can be converted
#     elif convertible_mono_formats:
#         cam.set_pixel_format(convertible_mono_formats[0])

#     else:
#         abort('Camera does not support an OpenCV compatible format. Abort.')

# class Handler:
#     def __init__(self):
#         self.display_queue = Queue(10)

#     def get_image(self):
#         return self.display_queue.get(True)

#     def __call__(self, cam: Camera, stream: Stream, frame: Frame):
#         if frame.get_status() == FrameStatus.Complete:
#             print('{} acquired {}'.format(cam, frame), flush=True)

#             # Convert frame if it is not already the correct format
#             if frame.get_pixel_format() == opencv_display_format:
#                 display = frame
#             else:
#                 # This creates a copy of the frame. The original `frame` object can be requeued
#                 # safely while `display` is used
#                 display = frame.convert_pixel_format(opencv_display_format)

#             self.display_queue.put(display.as_opencv_image(), True)

#         cam.queue_frame(frame)

# def main():
#     print_preamble()
#     cam_id = parse_args()

#     with VmbSystem.get_instance():
#         with get_camera(cam_id) as cam:
#             # setup general camera settings and the pixel format in which frames are recorded
#             setup_camera(cam)
#             setup_pixel_format(cam)
#             handler = Handler()

#             try:
#                 # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
#                 cam.start_streaming(handler=handler, buffer_count=10)

#                 msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
#                 import cv2
#                 ENTER_KEY_CODE = 13
#                 while True:
#                     key = cv2.waitKey(1)
#                     if key == ENTER_KEY_CODE:
#                         cv2.destroyWindow(msg.format(cam.get_name()))
#                         break

#                     display = handler.get_image()
#                     cv2.imshow(msg.format(cam.get_name()), display)

#             finally:
#                 cam.stop_streaming()


# if __name__ == '__main__':
#     main()


import sys
import threading
import time
import cv2
from flask import Flask, jsonify, send_file, Response
from queue import Queue
from vmbpy import *

# Flask app initialization
app = Flask(__name__)

opencv_display_format = PixelFormat.Bgr8
camera = None
handler = None
display_queue = Queue(10)  # Queue to hold frames for display


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


def setup_pixel_format(cam):
    """Set up the pixel format for the camera."""
    with cam:  # Ensure the camera is within the valid context
        print("Pixel format set:", camera.get_pixel_format())
        cam_formats = cam.get_pixel_formats()
        color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
        convertible_color_formats = [
            f for f in color_formats if opencv_display_format in f.get_convertible_formats()
        ]
        mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
        convertible_mono_formats = [
            f for f in mono_formats if opencv_display_format in f.get_convertible_formats()
        ]

        if opencv_display_format in cam_formats:
            cam.set_pixel_format(opencv_display_format)
        elif convertible_color_formats:
            cam.set_pixel_format(convertible_color_formats[0])
        elif convertible_mono_formats:
            cam.set_pixel_format(convertible_mono_formats[0])
        else:
            sys.exit("Camera does not support an OpenCV compatible format. Abort.")



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
    """Start the camera streaming."""
    global camera
    with VmbSystem.get_instance():
        camera = get_camera(None)  # Use the first camera
        with camera:
            setup_camera(camera)
            setup_pixel_format(camera)
            while True:  # Manually capture frames
                try:
                    frame = camera.get_frame()
                    if frame.get_status() == FrameStatus.Complete:
                        img = frame.convert_pixel_format(PixelFormat.Bgr8).as_opencv_image()
                        if not display_queue.full():
                            display_queue.put(img, timeout=1)
                            print("Frame manually queued.")
                    camera.queue_frame(frame)
                except Exception as e:
                    print(f"Error during manual capture: {e}")



@app.route('/start-stream', methods=['GET'])
def start_stream():
    """Start the camera stream."""
    try:
        print("Starting camera...")
        start_camera()
        print("Camera started")
        return jsonify({"message": "Camera stream started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/capture-image', methods=['GET'])
def capture_image():
    """Capture a single image from the camera."""
    print("Waiting for frames...")
    if display_queue.empty():
        print("No frames in queue!")
        return jsonify({"error": "No frame available"}), 500

    frame = display_queue.get(True)
    filename = 'keyboard_image.jpg'
    cv2.imwrite(filename, frame)  # Save the captured frame

    return send_file(filename, mimetype='image/jpeg')


@app.route('/stream', methods=['GET'])
def stream():
    """Stream the camera feed."""
    def generate():
        while True:
            if display_queue.empty():
                print("Queue is empty")
                time.sleep(0.1)
                continue
            frame = display_queue.get(True)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                print("Sending frame...")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop-stream', methods=['GET'])
def stop_stream():
    """Stop the camera stream."""
    global camera
    if camera:
        camera.stop_streaming()
        camera = None
        return jsonify({"message": "Camera stream stopped"}), 200
    else:
        return jsonify({"error": "No active camera stream"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


