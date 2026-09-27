@echo off
cd /d "%~dp0"
if not exist ".env" (
  echo.
  echo FarmTwin needs your Sarvam API key for voice features.
  echo Create .env now? This will save the key only on this computer.
  choice /C YN /N /M "Create .env [Y/N]: "
  if errorlevel 2 goto start
  set /p SARVAM_KEY=Paste your Sarvam API key: 
  >.env echo SARVAM_API_KEY=%SARVAM_KEY%
  >>.env echo SARVAM_BASE_URL=https://api.sarvam.ai
  >>.env echo SARVAM_CHAT_MODEL=sarvam-105b
  >>.env echo SARVAM_STT_MODEL=saaras:v4
  >>.env echo SARVAM_TTS_MODEL=bulbul:v3
  echo .env created.
)
:start
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
pause
