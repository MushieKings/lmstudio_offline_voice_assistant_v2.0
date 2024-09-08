@echo off

IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)

:: Deactivate the virtual environment if it’s already activated
call .\venv\Scripts\deactivate.bat

:: Activate the virtual environment
call .\venv\Scripts\activate.bat

:: Upgrade pip
python.exe -m pip install --upgrade pip


:: Install the requirements
pip3 install -r requirements.txt

pause