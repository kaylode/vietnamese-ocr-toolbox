# Arguments

IMAGE=''
INPUT="./data/sroie19/images/val/${IMAGE}.jpg"
OUTPUT_DIR="./results/$IMAGE"
PREPROCESS_RES="$OUTPUT_DIR/preprocessed.jpg"
DETECTION_RES="$OUTPUT_DIR/crops"
OCR_RES="$OUTPUT_DIR/ocr.txt"

PAN_WEIGHT='/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/detection-checkpoints/PANNet_best.pth'

OCR_WEIGHT="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/ocr-checkpoints/transformerocr.pth"
OCR_CONFIG="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/ocr-checkpoints/config.yml"


cd preprocess
python scan.py --image $IMAGE --output $PREPROCESS_RES
cd..

cd detection
python predict.py -i $PREPROCESS_RES -o $DETECTION_RES -w $PAN_WEIGHT
cd..

cd ocr
python predict.py -i $DETECTION_RES -o $OCR_RES -w $OCR_WEIGHT -c $OCR_CONFIG

