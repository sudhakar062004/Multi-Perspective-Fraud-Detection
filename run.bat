@echo off
title Multi-Perspective Fraud Detection System
echo ======================================================================
echo  A Multi-Perspective Fraud Detection Method for E-Commerce Transactions
echo ======================================================================
echo.
echo [1/3] Checking dependencies...
python -c "import django, pandas, sklearn, matplotlib" 2>NUL
if errorlevel 1 (
    echo Installing missing requirements...
    pip install -r requirements.txt
)
echo.
echo [2/3] Applying database migrations...
python manage.py migrate --noinput
echo.
echo [3/3] Starting Django Web Application...
echo.
echo Application URL: http://127.0.0.1:8000/
echo Login Credentials:
echo   Username: admin
echo   Password: admin
echo.
start http://127.0.0.1:8000/
python manage.py runserver 127.0.0.1:8000
pause