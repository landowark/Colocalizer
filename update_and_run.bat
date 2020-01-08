@ECHO OFF
SET thisdir=%~dp0
git -C %thisdir% pull https://github.com/landowark/Etienne.git master:master
%thisdir%\venv\Scripts\pip install -r requirements.txt
%thisdir%\venv\Scripts\python %thisdir%\main.py