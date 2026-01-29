import cv2

def open_camera(index: int = 0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open video device")
    return cap

def read_frame(cap):
    return cap.read()

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
