# How to Run Predictive Text System - Command Guide

## Prerequisites
1. Open PowerShell or Command Prompt
2. Navigate to your project folder

## Step 1: Navigate to Project Directory
```powershell
cd "C:\Users\Raghuram S\Amrita\FAI\Project\PTS"
```

## Step 2: Activate Virtual Environment
```powershell
& "C:/Users/Raghuram S/Amrita/FAI/Project/.venv/Scripts/python.exe"
```
Or if you have the virtual environment activated:
```powershell
..\..\.venv\Scripts\Activate.ps1
```

## Step 3: Run Different Components

### A) Main Comprehensive Demo (Shows overfitting vs legitimate improvement)
```powershell
python comprehensive_demo.py
```
**What it does:** Complete demonstration comparing fake vs real accuracy

### B) Interactive Typing Demo (Real-time predictions)
```powershell
python working_interactive_demo.py
```
**What it does:** Type sentences and see next word predictions in real-time

### C) Comprehensive Evaluation (Academic testing)
```powershell
python comprehensive_evaluation.py
```
**What it does:** Rigorous evaluation with cross-validation and overfitting checks

### D) Final Report (Summary of results)
```powershell
python final_report.py
```
**What it does:** Generates academic summary of all results

### E) Test Guide (Known patterns testing)
```powershell
python test_guide.py
```
**What it does:** Test specific patterns that the model knows well

## Quick Start (Recommended Order)

### For Academic Presentation:
1. ```python comprehensive_demo.py``` - Shows your methodology
2. ```python final_report.py``` - Gets your final results
3. ```python working_interactive_demo.py``` - Interactive demonstration

### For Development/Testing:
1. ```python comprehensive_evaluation.py``` - Verify no overfitting
2. ```python test_guide.py``` - Test known patterns
3. ```python working_interactive_demo.py``` - Manual testing

## Troubleshooting

### If you get "python not found":
```powershell
& "C:/Users/Raghuram S/Amrita/FAI/Project/.venv/Scripts/python.exe" comprehensive_demo.py
```

### If you get import errors:
Make sure you're in the right directory:
```powershell
pwd
```
Should show: `C:\Users\Raghuram S\Amrita\FAI\Project\PTS`

### If modules are missing:
```powershell
pip install -r requirements.txt
```

## File Descriptions

| File | Purpose | When to Use |
|------|---------|-------------|
| `comprehensive_demo.py` | Complete overfitting vs legitimate demo | Academic presentation |
| `working_interactive_demo.py` | Real-time typing interface | Live demonstration |
| `comprehensive_evaluation.py` | Rigorous testing framework | Verification/validation |
| `final_report.py` | Results summary | Final academic report |
| `test_guide.py` | Known pattern testing | Quick functionality check |

## Example Session

```powershell
# Navigate to project
cd "C:\Users\Raghuram S\Amrita\FAI\Project\PTS"

# Run main demo
python comprehensive_demo.py

# Try interactive demo
python working_interactive_demo.py

# Get final results
python final_report.py
```

## Tips
- Always run from the PTS directory
- Use `comprehensive_demo.py` for the best academic presentation
- Use `working_interactive_demo.py` for live demonstrations
- Check `final_report.py` for your final accuracy numbers (67.7%)
