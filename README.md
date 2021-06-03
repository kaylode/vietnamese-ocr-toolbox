# Vietnamese Receipt Reader 

## Pipeline
1. Use Canny Edge Detector and then detect contours to extract receipt from the image.
2. Use Pixel Agreation Network (PAN) to detect text regions from extracted receipt, then crop these regions.
3. Use VietOCR to extract texts from regions.
4. Retrieve information

## Datasets
- [MCOCR-2020](https://drive.google.com/file/d/1cyEGMVcEkquduJp3ewGq9Q4SyliX0bfB/view?usp=sharing)
- [SROIE19](https://drive.google.com/drive/folders/1jdFA0yg8uw15scux8O73qs6c5fr1cUff?usp=sharing)


## References
- https://github.com/WenmuZhou/PAN.pytorch
- https://github.com/andrewdcampbell/OpenCV-Document-Scanner
- https://github.com/pbcquoc/vietocr
