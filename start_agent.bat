@echo off

rem Define the name of your virtual environment
set ENV_NAME=myenv

rem Check if the virtual environment folder exists
if not exist %ENV_NAME% (
    rem Create a new virtual environment
    python -m venv %ENV_NAME%
)

rem Activate the virtual environment
call %ENV_NAME%\Scripts\activate

rem Install the required packages from requirements.txt
python -m pip install -r requirements.txt

rem Run your Streamlit application
python -m streamlit run agent_llama_ui.py

rem Deactivate the virtual environment
deactivate
