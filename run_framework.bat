@echo off
chcp 65001 > nul
cd /d "%~dp0framework"
python main.py
pause