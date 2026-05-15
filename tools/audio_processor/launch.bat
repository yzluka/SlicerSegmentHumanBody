@echo off
rem -----------------------------------------------------------------------
rem EDIT BEFORE USE: update both paths below to match your Python installation.
rem   PYTHONPATH — the site-packages directory for the Python env that has
rem                faster-whisper installed.
rem   pythonw.exe — the pythonw.exe (or python.exe) for that same env.
rem -----------------------------------------------------------------------
set PYTHONPATH=C:\Users\82755\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages
"C:\Users\82755\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\pythonw.exe" "%~dp0app.py" %*
