# 1. Python 3.9 베이스 이미지 사용
FROM python:3.10

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 패키지 설치 (Alembic 포함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 애플리케이션 코드 및 Alembic 스크립트 복사
COPY . .

ARG DATABASE_URL
ENV DATABASE_URL=${DATABASE_URL}

# 5. 데이터베이스 마이그레이션 실행 (Alembic)
#RUN alembic upgrade head

# 6. FastAPI 서버 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
