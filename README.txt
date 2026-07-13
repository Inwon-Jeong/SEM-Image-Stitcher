Automatic Multi-Image SEM Stitching Program
=============================================

Processing Workflow
-------------------
1. Automatically detect all supported image files in the input_images folder
2. Detect and match SIFT features for all image pairs
3. Remove uncertain correspondences using Lowe's ratio test
4. Remove outliers with RANSAC and calculate pairwise homographies
5. Construct an image connectivity network using a maximum spanning tree based on confidence scores
6. Compose global homographies in the coordinate system of the automatically selected central image
7. Warp all images onto a common canvas using Lanczos interpolation
8. Calculate a minimum-error seam in the overlap between each newly added image and the current composite
9. Use both intensity differences and edge costs to avoid strong structural boundaries
10. Apply feather blending only within a narrow region around each seam
11. Save only three TIFF files: a seam-preview image, the full composite, and the maximum-valid-rectangle crop

Supported Features
------------------
- Automatic image-count detection: two or more images
- Support for two-dimensional mosaics with multiple rows and columns
- Arbitrary file names: image connectivity is determined from image content rather than file names
- Supported formats: TIF/TIFF, PNG, JPG/JPEG, and BMP
- Support for 8-bit and 16-bit source images
- Support for grayscale or color images (grayscale and color images cannot be mixed in the same run)

First-Time Setup and Use
------------------------
1. Extract the ZIP archive.
2. Run 01_install_packages.bat once.
3. Place the overlapping original SEM images in the input_images folder.
4. Run 02_run_auto_stitch.bat.
5. Check the results in the output folder.

Output Files
------------
output/SEM_stitched_seams_preview.tif
- Preview image showing the actual minimum-error seam locations as colored lines

output/SEM_stitched_full.tif
- Full common-canvas composite before removal of black or empty margins

output/SEM_stitched_cropped.tif
- Recommended final image, cropped to the largest rectangular region without empty margins
