@echo off
echo Installerer avhengigheter...
pip install torch torchvision opencv-python numpy mediapipe pyautogui --quiet
echo.
echo Starter håndbevegelse-appen...
python src/webcam_live.py --checkpoint modell/landmark_heatmap11_best.pt
pause
