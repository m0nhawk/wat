@echo off

setlocal
set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\src

python wat.py

endlocal
