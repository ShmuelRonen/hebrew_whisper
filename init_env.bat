@echo off

rem Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
)

rem Activate the virtual environment
call venv\Scripts\activate

rem Perform git pull to update the repository
echo Updating the repository...
git pull
echo Repository updated.

rem Install requirements
echo Installing requirements...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
echo Requirements installed.

rem Run the Python script
python app.py

pause