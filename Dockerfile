# 1단계: 파이썬 실행 환경 설정
FROM python:3.9-slim

# 작업 디렉토리 생성
WORKDIR /app

# 필요한 패키지 설치 (numpy, websockets, aiohttp)
RUN pip install --no-cache-dir numpy websockets aiohttp

# 현재 폴더(helios_ai)의 모든 파일을 컨테이너의 /app으로 복사
COPY . .

# AI 서버가 사용하는 포트 두 개를 열어줌
EXPOSE 8083 
EXPOSE 8080

# 컨테이너 실행 시 파이썬 파일 실행 
CMD ["python", "simple_server.py"]