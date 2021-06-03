# Vietnamese Receipt Reader 

## Pipeline
1. Use Canny Edge Detector and then detect contours to extract receipt from the image.
2. Use Pixel Agreation Network (PAN) to detect text regions from extracted receipt, then crop these regions.
3. Use VietOCR to extract texts from regions.
4. Retrieve information
