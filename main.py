import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import time
import uuid
import os
import logging
import traceback

from insightface.app import FaceAnalysis  # استيراد مكتبة InsightFace

logger = logging.getLogger(__name__)

# تهيئة InsightFace لاكتشاف الوجوه
app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=-1)  # ctx_id=-1 يعني استخدام CPU (لو عايز GPU: 0)

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        weights = ResNeXt50_32X4D_Weights.DEFAULT
        model = resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

def detect_and_crop_faces(frame):
    faces = app.get(frame)
    cropped_faces = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_img = frame[y1:y2, x1:x2]
        cropped_faces.append(face_img)
    return cropped_faces

def extract_frames(video_path, num_frames=16, frames_folder='static/frames'):
    frames = []
    frame_paths = []
    unique_id = str(uuid.uuid4()).split('-')[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise Exception("Video file appears to be empty")
        
    interval = total_frames // num_frames
    count = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0 and frame_count < num_frames:
            cropped_faces = detect_and_crop_faces(frame)
            if len(cropped_faces) == 0:
                count += 1
                continue
                
            try:
                face_frame = cropped_faces[0]
                frame_path = os.path.join(frames_folder, f'frame_{unique_id}_{frame_count}.jpg')
                cv2.imwrite(frame_path, face_frame)
                frame_paths.append(os.path.basename(frame_path))
                frames.append(face_frame)
                frame_count += 1
                logger.info(f"Extracted frame {frame_count}: {os.path.basename(frame_path)}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                continue
                
        count += 1
        if frame_count >= num_frames:
            break
            
    cap.release()
    
    if len(frames) == 0:
        raise Exception("No faces detected in the video")
        
    return frames, frame_paths

def predict(model, img, path='./'):
    try:
        with torch.no_grad():
            fmap, logits = model(img.to())
            logits = F.softmax(logits, dim=1)
            _, prediction = torch.max(logits, 1)
            confidence = logits[:, int(prediction.item())].item() * 100
            label_map = {0: "real", 1: "fake"}
            predicted_label = label_map[int(prediction.item())]
            logger.info(f'Prediction: {predicted_label} with confidence {confidence:.2f}%')
            return [predicted_label, confidence]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        raise

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            cropped_faces = detect_and_crop_faces(frame)
            try:
                face_frame = cropped_faces[0]
                frame = face_frame
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def detectFakeVideo(videoPath, model_path='df_model.pt'):
    start_time = time.time()
    try:
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        path_to_videos = [videoPath]
        video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
        model = Model(2)
        
        if not os.path.exists(model_path):
            raise Exception("Model file not found")
            
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        prediction = predict(model, video_dataset[0])
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        return prediction, processing_time
    except Exception as e:
        logger.error(f"Error in detectFakeVideo: {str(e)}")
        traceback.print_exc()
        raise

