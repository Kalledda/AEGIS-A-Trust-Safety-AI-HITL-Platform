@echo off
title AEGIS Platform Launcher
echo =========================================
echo 🛡️ AEGIS: Trust & Safety AI-HITL Platform
echo =========================================
echo.
echo Starting Docker Container...
echo (Waiting 5 seconds for the server to boot...)
echo.

:: Start Docker in its own window so it doesn't block the script
start "AEGIS Server" docker run --rm -p 8501:8501 aegis-platform

:: Wait 5 seconds
timeout /t 5 /nobreak > NUL

:: Open the web browser
start http://localhost:8501