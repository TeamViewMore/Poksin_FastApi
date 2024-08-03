
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from uuid import uuid4
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from models import EvidenceEntity, ViolenceSegment

# Load .env file
load_dotenv()

# Logging configuration
log_file_path = 'fastapi.log'
handler = RotatingFileHandler(log_file_path, maxBytes=1000000, backupCount=3)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

if not DATABASE_URL or not BUCKET_NAME:
    raise RuntimeError("DATABASE_URL and BUCKET_NAME must be set in the environment variables.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = scoped_session(SessionLocal)

# Load pre-trained model
model_path = os.path.abspath('./modelnew.h5')

try:
    model = keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# AWS S3 setup
s3 = boto3.client('s3')
DIRECTORY = 'video/'

def upload_to_s3(file_path, bucket, s3_file_name):
    try:
        logger.info(f"Uploading {file_path} to S3 bucket {bucket} as {s3_file_name}")
        s3.upload_file(file_path, bucket, s3_file_name)
        s3_url = f"https://{bucket}.s3.amazonaws.com/{s3_file_name}"
        logger.info(f"Successfully uploaded to {s3_url}")
        return s3_url
    except FileNotFoundError:
        logger.error("File not found.")
        raise HTTPException(status_code=500, detail="File not found.")
    except NoCredentialsError:
        logger.error("Credentials not available.")
        raise HTTPException(status_code=500, detail="Credentials not available.")
    except PartialCredentialsError:
        logger.error("Incomplete credentials provided.")
        raise HTTPException(status_code=500, detail="Incomplete credentials provided.")
    except ClientError as e:
        logger.error(f"ClientError: {e}")
        raise HTTPException(status_code=500, detail=f"ClientError: {e}")
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

def download_from_s3(file_name, download_path):
    try:
        if not BUCKET_NAME:
            raise ValueError("BUCKET_NAME is not set")
        s3.download_file(BUCKET_NAME, f"{DIRECTORY}{file_name}", download_path)
        logger.info(f"Downloaded {file_name} from S3 to {download_path}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '403':
            logger.error(f"Access denied to S3 bucket. Check your permissions: {e}")
            raise HTTPException(status_code=403, detail=f"Access denied to S3 bucket. Check your permissions: {e}")
        elif error_code == '404':
            logger.error(f"File not found in S3 bucket: {e}")
            raise HTTPException(status_code=404, detail=f"File not found in S3 bucket: {e}")
        else:
            logger.error(f"ClientError: {e}")
            raise HTTPException(status_code=500, detail=f"ClientError: {e}")
    except NoCredentialsError:
        logger.error("Credentials not available.")
        raise HTTPException(status_code=500, detail="Credentials not available.")
    except PartialCredentialsError:
        logger.error("Incomplete credentials provided.")
        raise HTTPException(status_code=500, detail="Incomplete credentials provided.")
    except Exception as e:
        logger.error(f"Error downloading file from S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading file from S3: {str(e)}")

# Video processing functions
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_rate = max(int(fps / 5), 1)
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

def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (128, 128)) / 255.0
        processed_frames.append(frame)
    processed_frames = np.array(processed_frames)
    return processed_frames

def detect_violence(frames, model, merge_gap=5):
    violence_segments = []
    current_segment = None
    num_frames = len(frames)

    for i in range(num_frames):
        frame = np.expand_dims(frames[i], axis=0)
        prediction = model.predict(frame)
        if prediction[0][0] > 0.5:
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

    merge_gap_frames = merge_gap * 5
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

def save_violence_segments(original_frames, violence_segments, output_path_template, fps):
    height, width, layers = original_frames[0].shape
    saved_segments = []
    for idx, (start, end) in enumerate(violence_segments):
        output_path = output_path_template.format(idx + 1)
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        start_frame = start * 5
        end_frame = (end + 1) * 5 - 1

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

class EvidenceRequest(BaseModel):
    evidence_id: int
    file_name: str

def process_video(evidence_id, file_name, model, result_file_path, db_session):
    logger.info(f"Start processing video: evidence_id={evidence_id}, file_name={file_name}")
    temp_dir = os.path.join(os.getcwd(), 'temp')
    video_path = os.path.join(temp_dir, f'downloaded_video_{evidence_id}.mp4')

    try:
        download_from_s3(file_name, video_path)
        logger.info(f"Downloaded video from S3: {video_path}")

        original_frames, sampled_frames, fps = read_video(video_path)
        logger.info(f"Read video frames: total_frames={len(original_frames)}, sampled_frames={len(sampled_frames)}, fps={fps}")

        if fps == 0:
            fps = 30

        processed_frames = preprocess_frames(sampled_frames)
        logger.info(f"Preprocessed frames: {processed_frames.shape}")

        violence_segments = detect_violence(processed_frames, model, merge_gap=5)
        logger.info(f"Detected violence segments: {violence_segments}")

        output_path_template = os.path.join(temp_dir, 'violence_segment_{}.mp4')
        saved_segments = save_violence_segments(original_frames, violence_segments, output_path_template, fps)
        logger.info(f"Saved violence segments: {saved_segments}")

        segments_info = []
        evidence = db_session.query(EvidenceEntity).filter(EvidenceEntity.id == evidence_id).first()
        if not evidence:
            logger.error(f"EvidenceEntity with id {evidence_id} not found.")
            raise HTTPException(status_code=404, detail="EvidenceEntity not found.")
        logger.info(f"Queried evidence entity: {evidence}")

        for idx, (start, end) in enumerate(violence_segments):
            start_frame = start * 5
            end_frame = (end + 1) * 5 - 1

            if (end_frame - start_frame + 1) / fps < 1.0:
                start_frame = max(start_frame - int(fps), 0)
                end_frame = min(end_frame + int(fps), len(original_frames) - 1)

            duration = (end_frame - start_frame + 1) / fps

            s3_file_name = f"{DIRECTORY}{uuid4()}.mp4"
            segment_s3_url = upload_to_s3(saved_segments[idx], BUCKET_NAME, s3_file_name)
            logger.info(f"Uploaded segment to S3: {segment_s3_url}")

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
            db_session.add(violence_segment)
            segments_info.append(segment_info)

        try:
            evidence.done = 1
            db_session.commit()
            logger.info(f"Successfully updated evidence.done for evidence_id={evidence_id}")

            updated_evidence = db_session.query(EvidenceEntity).filter(EvidenceEntity.id == evidence_id).first()
            logger.info(f"Updated evidence.done value: {updated_evidence.done}")
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error updating evidence.done: {str(e)}")
            raise HTTPException(status_code=500, detail="Error updating evidence.done")

        with open(result_file_path, 'w') as f:
            json.dump({"segments": segments_info}, f)
        logger.info(f"Saved results to file: {result_file_path}")

        for file_path in saved_segments:
            os.remove(file_path)
        logger.info(f"Deleted saved segments: {saved_segments}")

        os.remove(video_path)
        logger.info(f"Deleted original video: {video_path}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise e

@app.post("/detect-violence/")
async def detect_violence_in_video(background_tasks: BackgroundTasks, request: EvidenceRequest, db: Session = Depends(get_db)):
    try:
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        result_file_path = os.path.join(temp_dir, f'result_{request.evidence_id}.json')

        background_tasks.add_task(process_video, request.evidence_id, request.file_name, model, result_file_path, db)

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

        os.remove(result_file_path)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
