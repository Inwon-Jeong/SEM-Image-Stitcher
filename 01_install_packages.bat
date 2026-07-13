@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ========================================
echo Python package installation
echo ========================================
echo.
echo Existing OpenCV packages will be removed first.
echo.

python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless

echo.
echo Installing tested versions...
echo.

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

echo.
echo Installed versions:
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"

echo.
pause
