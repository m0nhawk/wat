@echo off

setlocal EnableDelayedExpansion

set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\src;%cd%\ui

IF NOT EXIST "ui\ui_main.py" (
    start /I /B pyuic5 ui\main.ui -o ui\ui_main.py
)
python main.py

endlocal
