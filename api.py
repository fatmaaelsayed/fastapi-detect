from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np
import uuid
import traceback

from insightface.app import FaceAnalysis  # استيراد InsightFace
from main import Model  # تأكد أن المسار صحيح

app = FastAPI()

UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تهيئة نموذج الكشف عن الوجه من InsightFace مرة واحدة
face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])  # أو ['CUDAExecutionProvider'] لو عندك GPU
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# تهيئة النموذج مرة واحدة
model = Model(num_classes=2)
model_path = "df_model.pt"  # عدل المسار حسب مكان النموذج
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# تحويل للصور
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image_from_path(image_path):
    img = Image.open(image_path).convert("RGB")
    return transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(img).unsqueeze(0).unsqueeze(1).to(device)  # شكل (B=1, Seq=1, C, H, W)

def extract_faces_from_video(video_path, num_frames=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)
    count = 0
    extracted = 0
    
    while cap.isOpened() and extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            # كشف الوجوه باستخدام InsightFace
            faces = face_analyzer.get(frame)
            if faces:
                # نأخذ الوجه الأول فقط (مثل الكود القديم)
                face = faces[0]
                bbox = face.bbox.astype(int)  # إحداثيات الوجه: [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                face_img = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_tensor = image_transform(face_rgb)
                frames.append(face_tensor)
                extracted += 1
        
        count += 1
    cap.release()
    
    if not frames:
        raise Exception("No faces detected in video.")
    
    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)  # Batch=1, Seq=num_frames, C, H, W
    return frames_tensor

label_map = {0: "real", 1: "fake"}

@app.post("/predict/")
async def predict_file(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # احفظ الملف
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # صورة
            img_tensor = preprocess_image_from_path(file_path)
            with torch.no_grad():
                _, output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
            prediction_label = label_map[pred.item()]
            confidence = conf.item() * 100
            result = {"type": "image", "prediction": prediction_label, "confidence": round(confidence, 2)}

        elif filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            # فيديو
            frames_tensor = extract_faces_from_video(file_path)
            with torch.no_grad():
                _, output = model(frames_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
            prediction_label = label_map[pred.item()]
            confidence = conf.item() * 100
            result = {"type": "video", "prediction": prediction_label, "confidence": round(confidence, 2)}

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return result
