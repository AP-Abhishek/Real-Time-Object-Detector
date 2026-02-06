#!/usr/bin/env python
import sys
import subprocess
from pathlib import Path

def check_python_version():
    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11+ required")
        return False
    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_imports():
    imports = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("scipy", "SciPy"),
        ("ultralytics", "Ultralytics"),
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print(f"OK: {name}")
        except ImportError:
            print(f"ERROR: {name} not installed")
            all_ok = False
    
    return all_ok

def check_config():
    if not Path("config.yaml").exists():
        print("ERROR: config.yaml not found")
        return False
    print("OK: config.yaml found")
    return True

def check_model():
    if not Path("models/yolov8n.pt").exists():
        print("WARNING: Model not found (will auto-download on first run)")
        return True
    print("OK: Model found")
    return True

def check_validation():
    try:
        from src.validation import validate_config, ValidationError
        from src.config_loader import load_config
        cfg = load_config()
        print("OK: Configuration valid")
        return True
    except Exception as e:
        print(f"ERROR: Config validation failed: {e}")
        return False

def main():
    print("=== Setup Verification ===\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_imports),
        ("Config File", check_config),
        ("Model File", check_model),
        ("Configuration Validation", check_validation),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, False))
    
    print("\n=== Summary ===")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✓ Setup verified successfully!")
        return 0
    else:
        print("\n✗ Setup verification failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
