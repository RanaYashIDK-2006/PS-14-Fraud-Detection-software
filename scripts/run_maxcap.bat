@echo off
set PYTHONIOENCODING=utf-8
cd /d C:\Users\Rana Yash Singh\OneDrive\Documents\Project
echo Starting max-capacity test at %date% %time% > freebuff_mc.log
.venv\Scripts\python.exe -u scripts\max_capacity_test.py >> freebuff_mc.log 2>&1
echo Finished at %date% %time% >> freebuff_mc.log
