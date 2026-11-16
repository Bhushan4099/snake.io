@echo off
rem Snake AI Lab - Windows one-command entry point. Examples:
rem   run.bat -Quick     setup + tests + quick pipeline into results_quick\
rem   run.bat            setup + tests + full pipeline into results\
rem   run.bat -UI        setup + Pygame menu
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
