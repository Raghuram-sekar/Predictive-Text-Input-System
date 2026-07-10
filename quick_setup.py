#!/usr/bin/env python3
"""
Quick Setup Script for Professional N-gram Demo
==============================================
Installs all required packages and downloads NLTK data.
Run this once before the demo.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🚀 Setting up Professional N-gram Predictive Text System")
    print("=" * 55)
    
    # Required packages
    required_packages = [
        "nltk",
        "numpy", 
        "scipy"
    ]
    
    print("📦 Installing required packages...")
    all_successful = True
    
    for package in required_packages:
        print(f"  Installing {package}...")
        if not install_package(package):
            all_successful = False
    
    if all_successful:
        print("\n✅ All packages installed successfully!")
        print("\n🎯 Setup complete! You can now run:")
        print("   python professional_demo.py")
        print("\nThis will give you a working predictive text system")
        print("with professional-grade accuracy for your demonstration!")
    else:
        print("\n❌ Some packages failed to install.")
        print("Please install them manually or check your internet connection.")

if __name__ == "__main__":
    main()