@echo off
setlocal EnableExtensions
title PPE SAFETY - AI MONITORING
cd /d "%~dp0"
echo ==============================================
echo        PPE SAFETY - AI MONITORING
echo ==============================================
echo Starting system...
where python >nul 2>nul || (echo ERROR: Python is not installed.& pause & exit /b 1)
where node >nul 2>nul || (echo ERROR: Node.js is not installed.& pause & exit /b 1)
where npm.cmd >nul 2>nul || (echo ERROR: npm is not installed.& pause & exit /b 1)
echo [1/4] Starting existing backend...
netstat -ano | findstr ":8501 .*LISTENING" >nul
if errorlevel 1 start "PPE Safety Backend" /min cmd /c "cd /d ""%~dp0"" && python -m streamlit run app_hse.py --server.port 8501 --server.headless true"
set /a PPE_BACKEND_TRIES=0
:wait_backend
set /a PPE_BACKEND_TRIES+=1
curl.exe -fsS --max-time 8 http://127.0.0.1:8501/_stcore/health >nul 2>nul
if not errorlevel 1 goto backend_ready
if %PPE_BACKEND_TRIES% GEQ 120 goto backend_timeout
ping 127.0.0.1 -n 2 >nul
goto wait_backend
:backend_ready
echo [OK] Backend ready on port 8501
echo [2/4] Starting frontend...
if not exist "%~dp0frontend\node_modules" (
  echo Installing frontend dependencies for first use...
  call npm.cmd --prefix "%~dp0frontend" install
  if errorlevel 1 (echo ERROR: Dependency installation failed.& pause & exit /b 1)
)
netstat -ano | findstr ":6600 .*LISTENING" >nul
if not errorlevel 1 goto frontend_port_busy
start "PPE Safety Frontend" /min cmd /c "cd /d ""%~dp0frontend"" && npm.cmd run dev -- --port 6600 --strictPort"
echo [OK] Frontend start requested
echo [3/4] Checking AI system...
set /a PPE_TRIES=0
:wait_frontend
set /a PPE_TRIES+=1
curl.exe -fsS --max-time 5 http://localhost:6600 >nul 2>nul
if not errorlevel 1 goto ready
if %PPE_TRIES% GEQ 30 goto timeout
ping 127.0.0.1 -n 2 >nul
goto wait_frontend
:ready
echo [OK] AI monitoring interface ready
echo [4/4] Opening application...
start "" "http://localhost:6600"
echo [OK] PPE Safety ready
echo ==============================================
ping 127.0.0.1 -n 5 >nul
exit /b 0
:timeout
echo ERROR: Frontend did not become ready. Check the frontend window.
pause
exit /b 1
:frontend_port_busy
echo ERROR: Port 6600 is already occupied. Close the application using this port and try again.
pause
exit /b 1
:backend_timeout
echo ERROR: The existing backend did not become available on port 8501.
pause
exit /b 1
