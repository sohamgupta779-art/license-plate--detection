import cv2
import easyocr
from ultralytics import YOLO

def run_inference(weights, image_path):
    model = YOLO(weights)
    reader = easyocr.Reader(['en'])
    img = cv2.imread(image_path)
    results = model(image_path)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            text = reader.readtext(crop, detail=0)
            plate = text[0] if text else "UNKNOWN"
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, plate, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    out_path = "demo/output.jpg"
    cv2.imwrite(out_path, img)
    print(f"✅ Saved result to {out_path}")

if __name__ == "__main__":
    run_inference("best.pt", "demo/sample_input.jpg")
