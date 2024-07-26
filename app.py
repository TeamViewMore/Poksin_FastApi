import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from uuid import uuid4
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from sqlalchemy.orm import Session
import json

# .env 파일 로드
load_dotenv()

# 데이터베이스 설정
DATABASE_URL = os.getenv("DATABASE_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# 환경 변수 확인
if not DATABASE_URL or not BUCKET_NAME:
    raise RuntimeError("DATABASE_URL and BUCKET_NAME must be set in the environment variables.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class EvidenceEntity(Base):
    __tablename__ = "evidenceentity"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = Column(String(255))
    done = Column(Boolean)
    fileUrls = Column(Text)
    title = Column(String(255))
    category_id = Column(Integer)
    user_id = Column(Integer)

class ViolenceSegment(Base):
    __tablename__ = "violence_segment"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey('evidenceentity.id'))
    s3_url = Column(String(255))
    duration = Column(Float)
    evidence = relationship("EvidenceEntity", back_populates="segments")

EvidenceEntity.segments = relationship("ViolenceSegment", order_by=ViolenceSegment.id, back_populates="evidence")

Base.metadata.create_all(bind=engine)  # 새 테이블 생성


# 모델 경로 설정
model_path = os.path.abspath('./modelnew.h5')

# 사전 학습된 모델 로드
try:
    model = keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

app = FastAPI()

# AWS S3 설정
s3 = boto3.client('s3')
DIRECTORY = 'video/'

def upload_to_s3(file_path, bucket, s3_file_name):
    try:
        s3.upload_file(file_path, bucket, s3_file_name)
        s3_url = f"https://{bucket}.s3.amazonaws.com/{s3_file_name}"
        return s3_url
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="File not found.")
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Credentials not available.")
    except PartialCredentialsError:
        raise HTTPException(status_code=500, detail="Incomplete credentials provided.")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"ClientError: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def download_from_s3(file_name, download_path):
    try:
        if not BUCKET_NAME:
            raise ValueError("BUCKET_NAME is not set")
        s3.download_file(BUCKET_NAME, f"{DIRECTORY}{file_name}", download_path)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '403':
            raise HTTPException(status_code=403, detail=f"Access denied to S3 bucket. Check your permissions: {e}")
        elif error_code == '404':
            raise HTTPException(status_code=404, detail=f"File not found in S3 bucket: {e}")
        else:
            raise HTTPException(status_code=500, detail=f"ClientError: {e}")
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Credentials not available.")
    except PartialCredentialsError:
        raise HTTPException(status_code=500, detail="Incomplete credentials provided.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file from S3: {str(e)}")

# 비디오 읽기 함수 (1초당 5프레임만 추출)
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_rate = max(int(fps / 5), 1)  # 1초에 5프레임을 추출
    frame_count = 0
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
        if frame_count % sample_rate == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return original_frames, frames, fps

# 프레임 전처리 함수
def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (128, 128)) / 255.0
        processed_frames.append(frame)
    processed_frames = np.array(processed_frames)
    return processed_frames

# 폭력 검출 함수
def detect_violence(frames, model, merge_gap=5):
    violence_segments = []
    current_segment = None
    num_frames = len(frames)
    
    for i in range(num_frames):
        frame = np.expand_dims(frames[i], axis=0)  # 모델 입력 형태에 맞추기 위해 차원 추가
        prediction = model.predict(frame)
        if prediction[0][0] > 0.5:  # 모델 출력이 확률이라고 가정
            if current_segment is None:
                current_segment = [i, i]
            else:
                current_segment[1] = i
        else:
            if current_segment is not None:
                violence_segments.append(current_segment)
                current_segment = None
    if current_segment is not None:
        violence_segments.append(current_segment)
    
    merge_gap_frames = merge_gap * 5  # 초당 5프레임만 사용
    merged_segments = []
    i = 0
    while i < len(violence_segments):
        start, end = violence_segments[i]
        while i + 1 < len(violence_segments) and violence_segments[i + 1][0] - end < merge_gap_frames:
            end = violence_segments[i + 1][1]
            i += 1
        merged_segments.append([start, end])
        i += 1
    
    return merged_segments

# 비디오 저장 함수 (폭력 검출된 구간별로 저장)
def save_violence_segments(original_frames, violence_segments, output_path_template, fps):
    height, width, layers = original_frames[0].shape
    saved_segments = []
    for idx, (start, end) in enumerate(violence_segments):
        output_path = output_path_template.format(idx + 1)
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # 원본 프레임 인덱스로 변환
        start_frame = start * 5
        end_frame = (end + 1) * 5 - 1
        
        # 지속 시간이 1초 미만인 경우 이전 1초와 이후 1초를 추가
        if (end_frame - start_frame + 1) / fps < 1.0:
            start_frame = max(start_frame - int(fps), 0)
            end_frame = min(end_frame + int(fps), len(original_frames) - 1)
        
        for i in range(start_frame, end_frame + 1):
            if i < len(original_frames):
                frame = original_frames[i]
                cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 2)
                video.write(frame)
        video.release()
        saved_segments.append(output_path)
    
    return saved_segments

# 요청 데이터 모델
class EvidenceRequest(BaseModel):
    evidence_id: int
    file_name: str

# 백그라운드 작업 함수
def process_video(evidence_id, file_name, model, result_file_path):
    temp_dir = os.path.join(os.getcwd(), 'temp')
    video_path = os.path.join(temp_dir, f'downloaded_video_{evidence_id}.mp4')
    download_from_s3(file_name, video_path)
    
    original_frames, sampled_frames, fps = read_video(video_path)
    
    if fps == 0:
        fps = 30
    
    processed_frames = preprocess_frames(sampled_frames)
    violence_segments = detect_violence(processed_frames, model, merge_gap=5)
    
    output_path_template = os.path.join(temp_dir, 'violence_segment_{}.mp4')
    saved_segments = save_violence_segments(original_frames, violence_segments, output_path_template, fps)
    
    segments_info = []
    db = SessionLocal()
    evidence = db.query(EvidenceEntity).filter(EvidenceEntity.id == evidence_id).first()
    print(evidence)
    for idx, (start, end) in enumerate(violence_segments):
        start_frame = start * 5
        end_frame = (end + 1) * 5 - 1
        
        # 지속 시간이 1초 미만인 경우 이전 1초와 이후 1초를 추가
        if (end_frame - start_frame + 1) / fps < 1.0:
            start_frame = max(start_frame - int(fps), 0)
            end_frame = min(end_frame + int(fps), len(original_frames) - 1)
        
        duration = (end_frame - start_frame + 1) / fps  # 원본 프레임 기준 지속 시간 계산
        
        s3_file_name = f"{DIRECTORY}{uuid4()}.mp4"
        segment_s3_url = upload_to_s3(saved_segments[idx], BUCKET_NAME, s3_file_name)
        
        segment_info = {
            "segment_index": idx + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration": duration,
            "s3_url": segment_s3_url
        }
        
        violence_segment = ViolenceSegment(
            evidence_id=evidence.id,
            s3_url=segment_info["s3_url"],
            duration=segment_info["duration"]
        )
        db.add(violence_segment)
        segments_info.append(segment_info)
    
    evidence.done = True
    
    db.commit()
    db.close()
    
    with open(result_file_path, 'w') as f:
        json.dump({"segments": segments_info}, f)
    
    # 로컬 파일 삭제
    for file_path in saved_segments:
        os.remove(file_path)
    os.remove(video_path)

@app.post("/detect-violence/")
async def detect_violence_in_video(background_tasks: BackgroundTasks, request: EvidenceRequest):
    try:
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        result_file_path = os.path.join(temp_dir, f'result_{request.evidence_id}.json')
        
        background_tasks.add_task(process_video, request.evidence_id, request.file_name, model, result_file_path)
        
        return JSONResponse(content={"message": "비디오 처리 중입니다.", "evidence_id": request.evidence_id, "file_name": request.file_name})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{evidence_id}")
async def get_result(evidence_id: int):
    try:
        temp_dir = os.path.join(os.getcwd(), 'temp')
        result_file_path = os.path.join(temp_dir, f'result_{evidence_id}.json')
        
        if not os.path.exists(result_file_path):
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
        
        with open(result_file_path, 'r') as f:
            result = json.load(f)
        
        # 결과 파일 읽은 후 삭제
        os.remove(result_file_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
