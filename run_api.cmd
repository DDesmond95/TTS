@echo off
echo Starting Qwen3 TTS API...
call conda activate tts
set PYTHONPATH=%PYTHONPATH%;%~dp0src
python -m qwen3_api.main
pause
