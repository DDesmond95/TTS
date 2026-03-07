@echo off
echo ========================================================
echo        OmniVoice Studio Example Runner
echo ========================================================
echo.

if "%1"=="" goto help

python test_suite.py --task %1
goto end

:help
echo Usage: run_examples.cmd [task]
echo.
echo Tasks:
echo   download     - Download all necessary models
echo   omnivoice    - Test integrated platform engine
echo   qwen3        - Test Qwen3-TTS directly
echo   meanvc       - Test MeanVC (info only)
echo   tcsinger     - Test TCSinger2 (info only)
echo   sculptor     - Test VoiceSculptor (info only)
echo   all          - Run all tests
echo.
echo Example: run_examples.cmd omnivoice
goto end

:end
echo.
echo ========================================================
pause
