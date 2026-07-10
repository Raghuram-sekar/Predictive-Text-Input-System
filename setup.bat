@echo off
echo Setting up Predictive Text System...

echo Step 1: Creating virtual environment...
python -m venv venv

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Installing required packages...
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install nltk>=3.7
pip install tqdm>=4.64.0

echo Step 4: Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('brown', quiet=True); nltk.download('gutenberg', quiet=True)"

echo Setup complete! 
echo.
echo To run the project:
echo 1. venv\Scripts\activate.bat
echo 2. python comprehensive_demo.py
echo.
pause
