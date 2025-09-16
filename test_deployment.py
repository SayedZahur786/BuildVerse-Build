#!/usr/bin/env python3
"""
Test script to validate CareCanvas app deployment readiness.
Run this before deploying to check for common issues.
"""

import os
import sys
import tempfile
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import streamlit as st
        import torch
        import ultralytics
        import pandas as pd
        import numpy as np
        import cv2
        from PIL import Image
        import openpyxl
        import sklearn
        print("✅ All required packages import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test that the YOLO model loads correctly."""
    print("Testing model loading...")
    try:
        from predict import model
        print(f"✅ Model loaded successfully on device: {model.device}")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_app_modules():
    """Test that app modules load correctly."""
    print("Testing app modules...")
    try:
        from predict import predict
        from ingredients import ingredient_database
        from app import get_path, local_css
        print(f"✅ App modules loaded, {len(ingredient_database)} ingredients in database")
        return True
    except Exception as e:
        print(f"❌ App module error: {e}")
        return False

def test_file_operations():
    """Test file operations and temp directory creation."""
    print("Testing file operations...")
    try:
        # Test temp directory creation
        os.makedirs("temp_images", exist_ok=True)
        
        # Test temp file creation and cleanup
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        
        os.remove(tmp_path)
        print("✅ File operations working correctly")
        return True
    except Exception as e:
        print(f"❌ File operation error: {e}")
        return False

def test_prediction():
    """Test prediction functionality if test image exists."""
    print("Testing prediction functionality...")
    try:
        from predict import predict
        test_image = "images/1.jpg"
        
        if os.path.exists(test_image):
            predictions, save_path = predict(test_image)
            print(f"✅ Prediction test completed: {len(predictions)} predictions")
            
            # Cleanup if analysis image was created
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
                print("✅ Temp file cleanup successful")
        else:
            print("⚠️  Test image not found, skipping prediction test")
        return True
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Run all deployment readiness tests."""
    print("🧪 Running CareCanvas deployment readiness tests...\n")
    
    tests = [
        test_imports,
        test_model_loading,
        test_app_modules,
        test_file_operations,
        test_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! App is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())