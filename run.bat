@ECHO OFF
SET thisdir=%~dp0
REM git -C %thisdir% pull https://github.com/landowark/Etienne.git master:master
REM %thisdir%\venv\Scripts\pip install -r requirements.txt
%thisdir%\venv\Scripts\python %thisdir%\main.py