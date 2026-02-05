import cv2

def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    return cap

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file")
    return cap

def read_frame(cap):
    return cap.read()

def release_camera(cap):
    cap.release()
