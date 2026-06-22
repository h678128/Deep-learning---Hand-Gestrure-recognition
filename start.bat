@echo off
echo === Handbevegelse - musekontroll ===
echo.

:: Last ned modell hvis den mangler
if not exist "modell\landmark_heatmap11_best.pt" (
    echo Laster ned modell...
    if not exist "modell" mkdir modell
    curl -L -o "modell\landmark_heatmap11_best.pt" "https://github.com/ViktorKnu/Deep-learning---Hand-gesture--recognition/releases/download/v1.0/landmark_heatmap11_best.pt"
    if errorlevel 1 (
        echo FEIL: Kunne ikke laste ned modellen.
        pause
        exit /b 1
    )
    echo Modell lastet ned.
    echo.
)

:: Installer avhengigheter
echo Installerer avhengigheter...
pip install torch torchvision opencv-python numpy mediapipe pyautogui --quiet
echo.

:: Start appen
echo Starter håndbevegelse-appen...
echo Trykk Q i kameravinduet for aa avslutte.
echo.
python src/webcam_live.py --checkpoint modell/landmark_heatmap11_best.pt
pause
