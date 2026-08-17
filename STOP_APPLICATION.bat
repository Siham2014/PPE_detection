@echo off
setlocal EnableExtensions
title Stop PPE SAFETY
echo Stopping PPE Safety services...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":6600 .*LISTENING"') do taskkill /PID %%P /T /F >nul 2>nul
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501 .*LISTENING"') do taskkill /PID %%P /T /F >nul 2>nul
echo PPE Safety has been stopped.
ping 127.0.0.1 -n 4 >nul
