@echo off
chcp 65001 > nul
cd /d "%~dp0pipeline"
python main.py
pause