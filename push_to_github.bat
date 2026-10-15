@echo off
title Push Stroke Risk Prediction to GitHub
echo ========================================================
echo   Pushing Stroke Risk Prediction to GitHub Repo:
echo   https://github.com/vu241fa04185-sys/Stroke-Risk-Prediction-.git
echo ========================================================
echo.
cd /d "C:\Users\satya\.gemini\antigravity\scratch\stroke_prediction_system"

echo Setting remote URL...
git remote set-url origin https://github.com/vu241fa04185-sys/Stroke-Risk-Prediction-.git

echo.
echo Pushing branch 'main' to GitHub...
echo (If prompted, sign in as 'vu241fa04185-sys' in the browser window)
echo.
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   SUCCESS! Your project has been pushed to GitHub!
    echo   https://github.com/vu241fa04185-sys/Stroke-Risk-Prediction-
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   NOTE: If you got a 403 Permission Denied error:
    echo   1. Go to: https://github.com/vu241fa04185-sys/Stroke-Risk-Prediction-/settings/access
    echo   2. Add 'SATYAMJAk' as a collaborator with Write access.
    echo   3. Then run this script again!
    echo ========================================================
)

echo.
pause
