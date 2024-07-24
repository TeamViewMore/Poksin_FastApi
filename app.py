import cv2
import numpy as np
import tensorflow as tf
import keras
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import os
import json
from uuid import uuid4
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

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
BUCKET_NAME = 'poksin'
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# 백그라운드 작업 함수
def process_video(video_path, model, result_file_path):
    original_frames, sampled_frames, fps = read_video(video_path)
    
    if fps == 0:
        fps = 30
    
    processed_frames = preprocess_frames(sampled_frames)
    violence_segments = detect_violence(processed_frames, model, merge_gap=5)
    
    temp_dir = os.path.join(os.getcwd(), 'temp')
    output_path_template = os.path.join(temp_dir, 'violence_segment_{}.mp4')
    saved_segments = save_violence_segments(original_frames, violence_segments, output_path_template, fps)
    
    segments_info = []
    for idx, (start, end) in enumerate(violence_segments):
        start_frame = start * 5
        end_frame = (end + 1) * 5 - 1
        
        # 지속 시간이 1초 미만인 경우 이전 1초와 이후 1초를 추가
        if (end_frame - start_frame + 1) / fps < 1.0:
            start_frame = max(start_frame - int(fps), 0)
            end_frame = min(end_frame + int(fps), len(original_frames) - 1)
        
        duration = (end_frame - start_frame + 1) / fps  # 원본 프레임 기준 지속 시간 계산
        
        s3_file_name = f"{DIRECTORY}{uuid4()}.mp4"
        s3_url = upload_to_s3(saved_segments[idx], BUCKET_NAME, s3_file_name)
        
        segments_info.append({
            "segment_index": idx + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration": duration,
            "s3_url": s3_url
        })
    
    with open(result_file_path, 'w') as f:
        json.dump({"segments": segments_info}, f)
    
    # 로컬 파일 삭제
    for file_path in saved_segments:
        os.remove(file_path)

@app.post("/detect-violence/")
async def detect_violence_in_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        video_id = str(uuid4())
        video_path = os.path.join(temp_dir, f'uploaded_video_{video_id}.mp4')
        result_file_path = os.path.join(temp_dir, f'result_{video_id}.json')
        
        with open(video_path, 'wb') as f:
            f.write(await file.read())
        
        # 업로드한 원본 비디오를 S3에 저장
        s3_file_name = f"{DIRECTORY}{uuid4()}.mp4"
        s3_url = upload_to_s3(video_path, BUCKET_NAME, s3_file_name)
        
        background_tasks.add_task(process_video, video_path, model, result_file_path)
        
        # 업로드 후 원본 비디오 파일 삭제
        os.remove(video_path)
        
        return JSONResponse(content={"message": "업로드가 완료되었습니다. 비디오 처리 중입니다.", "video_id": video_id, "s3_url": s3_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{video_id}")
async def get_result(video_id: str):
    try:
        temp_dir = os.path.join(os.getcwd(), 'temp')
        result_file_path = os.path.join(temp_dir, f'result_{video_id}.json')
        
        if not os.path.exists(result_file_path):
            raise HTTPException(status_code=404, detail="결果를 찾을 수 없습니다.")
        
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
