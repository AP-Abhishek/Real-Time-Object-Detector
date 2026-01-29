from src.camera import run_camera
from src.detector import load_model

def main():
    load_model()
    print("Model loaded successfully.")

if __name__ == "__main__":
    main()
