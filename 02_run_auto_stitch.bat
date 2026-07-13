@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist "input_images" mkdir "input_images"
if not exist "output" mkdir "output"

echo ================================================
echo SEM automatic mosaic stitching
echo SIFT + RANSAC Homography + minimum-error seam
echo ================================================
echo.
echo Put all overlapping SEM images in input_images.
echo File names may be arbitrary.
echo The program determines image connections from image content.
echo.

python "sem_stitch_sift_ransac_seam.py" ^
  --input-dir "input_images" ^
  --output-dir "output" ^
  --ratio-test 0.78 ^
  --ransac-threshold 4.0 ^
  --max-features 15000 ^
  --min-good-matches 10 ^
  --min-inliers 8 ^
  --min-inlier-ratio 0.30 ^
  --min-scale 0.75 ^
  --max-scale 1.33 ^
  --max-rotation 20.0 ^
  --edge-weight 1.20 ^
  --smoothness-penalty 2.50 ^
  --max-seam-step 5 ^
  --cost-blur-sigma 1.20 ^
  --feather-half-width 2.0

echo.
if errorlevel 1 (
  echo Stitching failed. Check the error message above.
) else (
  echo Stitching completed successfully.
)
echo.
pause
