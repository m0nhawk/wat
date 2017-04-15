@echo off

setlocal

set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\src

if not exist "ui_main.py" (
    pyuic5 main.ui -o ui_main.py

    set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\src
    python wat.py
) else (
    python wat.py
)

endlocal
