# 1단계: 파이썬 실행 환경 설정
FROM python:3.9-slim

# 작업 디렉토리 생성
WORKDIR /app

# 필요한 패키지 설치 (numpy, websockets, aiohttp)
RUN pip install --no-cache-dir numpy websockets aiohttp
RUN pip install --no-cache-dir fastapi uvicorn websockets numpy httpx

# 현재 폴더(helios_ai)의 모든 파일을 컨테이너의 /app으로 복사
COPY . .

# AI 서버 포트
EXPOSE 8000

# 컨테이너 실행 시 파이썬 파일 실행 
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
