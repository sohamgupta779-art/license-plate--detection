"""Automatic License Detection
Original file is located at
    https://colab.research.google.com/drive/1JIlVeyx5AAW7E7ua48r5ZUBJHBbKY6Uo
"""
from google.colab import files
uploaded = files.upload()

!pip install ultralytics==8.0.120

import torch
import zipfile, os


dataset_zip = "License Plate Detection.v1i.yolov5pytorch.zip"

with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall("/content/data/")

print("Extracted files:", os.listdir("/content/data/"))




from ultralytics import YOLO


model = YOLO("yolov8n.pt")


model.train(
    data="/content/data/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs",
    name="license_plate",
    verbose=True
)

from ultralytics import YOLO


model = YOLO("/content/runs/license_plate9/weights/best.pt")


results = model.predict(
    source="/content/data/test/images",
    imgsz=640,
    conf=0.25,
    save=True
)

import matplotlib.pyplot as plt
import cv2
import glob


predicted_images = glob.glob("/content/runs/detect/predict/*.jpg")


for img_path in predicted_images[:3]:
    img = cv2.imread(img_path)[:,:,::-1]  # BGR → RGB
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.title(img_path.split("/")[-1])
    plt.axis("off")
    plt.show()

!pip install easyocr

import cv2
import easyocr
from ultralytics import YOLO


model = YOLO("/content/runs/license_plate9/weights/best.pt")


reader = easyocr.Reader(['en'])


img_path = "/content/data/test/images/"
results = model(img_path)


for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        img = cv2.imread(r.path)
        crop = img[y1:y2, x1:x2]  # crop plate region


        text = reader.readtext(crop, detail=0)
        print(f"Detected Plate: {text}")


        from matplotlib import pyplot as plt
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title(f"OCR: {text}")
        plt.axis("off")
        plt.show()

import cv2
import easyocr
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt


model = YOLO("/content/runs/license_plate9/weights/best.pt")


reader = easyocr.Reader(['en'])


test_images = glob.glob("/content/data/test/images/*.jpg")

results_dict = {}

for img_path in test_images:
    img = cv2.imread(img_path)
    detections = model(img_path)

    plates = []
    for r in detections:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            # OCR on crop
            text = reader.readtext(crop, detail=0)
            if text:
                cleaned_text = "".join([c for c in text[0] if c.isalnum()])  # filter non-alphanumeric
                plates.append(cleaned_text)

    results_dict[img_path] = plates


    if plates:
        plt.figure(figsize=(8,5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{img_path.split('/')[-1]} | OCR: {plates}")
        plt.axis("off")
        plt.show()


for k,v in results_dict.items():
    print(f"{k.split('/')[-1]} --> {v}")

import pandas as pd


rows = []
for img_path, plates in results_dict.items():
    if plates:
        for plate in plates:
            rows.append({"image": img_path.split("/")[-1], "plate_text": plate})
    else:
        rows.append({"image": img_path.split("/")[-1], "plate_text": "NO_PLATE_DETECTED"})

df = pd.DataFrame(rows)


csv_path = "/content/ocr_results.csv"
df.to_csv(csv_path, index=False)

print(f"OCR results saved to {csv_path}")
df.head()

from google.colab import files
files.download("/content/ocr_results.csv")

import cv2
import glob


image_paths = sorted(glob.glob("/content/data/test/images/*.jpg"))


frame = cv2.imread(image_paths[0])
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("demo_input.mp4", fourcc, 3, (width, height))  # 3 FPS

for img_path in image_paths:
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print(" Demo video created: demo_input.mp4")

import easyocr
from ultralytics import YOLO


model = YOLO("/content/runs/license_plate9/weights/best.pt")
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture("demo_input.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("demo_output.mp4", fourcc, 3, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            text = reader.readtext(crop, detail=0)
            if text:
                plate_text = "".join([c for c in text[0] if c.isalnum()])
            else:
                plate_text = ""

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, plate_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()
print(" Processed demo video saved as demo_output.mp4")

from google.colab import files
files.download("demo_output.mp4")

import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files

# Load your trained model once
model = YOLO("/content/runs/license_plate9/weights/best.pt")

# Initialize OCR once
reader = easyocr.Reader(['en'])

def detect_and_ocr():
    # Upload new image
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]

    # Run YOLO detection
    results = model(img_path)
    img = cv2.imread(img_path)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            # OCR on crop
            text = reader.readtext(crop, detail=0)
            plate_text = "".join([c for c in text[0] if c.isalnum()]) if text else "UNKNOWN"

            # Draw bounding box + OCR result
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, plate_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show image
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

detect_and_ocr()

model = YOLO("/content/runs/license_plate9/weights/best.pt")
reader = easyocr.Reader(['en'])

def batch_detect_and_ocr():

    uploaded = files.upload()
    results_dict = {}

    for img_path in uploaded.keys():
        img = cv2.imread(img_path)
        detections = model(img_path)
        plates = []

        for r in detections:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]


                text = reader.readtext(crop, detail=0)
                if text:
                    plate_text = "".join([c for c in text[0] if c.isalnum()])
                    plates.append(plate_text)

                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, plate_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        results_dict[img_path] = plates if plates else ["NO_PLATE_DETECTED"]


        plt.figure(figsize=(10,6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{img_path} | OCR: {results_dict[img_path]}")
        plt.axis("off")
        plt.show()

    return results_dict

results = batch_detect_and_ocr()
for img, plates in results.items():
    print(f"{img} --> {plates}")

model = YOLO("/content/runs/license_plate9/weights/best.pt")
reader = easyocr.Reader(['en'])

def batch_detect_and_ocr_to_csv(csv_path="/content/batch_ocr_results.csv"):

    uploaded = files.upload()
    results_dict = {}
    rows = []

    for img_path in uploaded.keys():
        img = cv2.imread(img_path)
        detections = model(img_path)
        plates = []

        for r in detections:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]


                text = reader.readtext(crop, detail=0)
                if text:
                    plate_text = "".join([c for c in text[0] if c.isalnum()])
                    plates.append(plate_text)


                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, plate_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        results_dict[img_path] = plates if plates else ["NO_PLATE_DETECTED"]


        for plate in results_dict[img_path]:
            rows.append({"image": img_path, "plate_text": plate})


        plt.figure(figsize=(10,6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{img_path} | OCR: {results_dict[img_path]}")
        plt.axis("off")
        plt.show()


    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"Batch OCR results saved to {csv_path}")
    return results_dict, csv_path

results, csv_file = batch_detect_and_ocr_to_csv()
for img, plates in results.items():
    print(f"{img} --> {plates}")

from google.colab import files
files.download(csv_file)






