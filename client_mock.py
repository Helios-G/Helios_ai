import asyncio
import websockets
import json

async def simulate_client(name):
    uri = "ws://localhost:8083"
    async with websockets.connect(uri) as websocket:
        print(f"[{name}] 서버 연결 완료")
        while True:
            try:
                msg = await websocket.recv()
                data = json.loads(msg)
                if data["type"] == "fit":
                    print(f"[{name}] 학습 시작 지시 받음 (Round {data.get('config', {}).get('epochs')})")
                    # 가상의 가중치 생성 (단순 리스트)
                    dummy_weights = [[0.1, 0.2], [0.3, 0.4]] 
                    response = json.dumps({"type": "fit_res", "parameters": dummy_weights})
                    await websocket.send(response)
                    print(f"[{name}] 가중치 전송 완료")
            except websockets.ConnectionClosed:
                print(f"[{name}] 연결 종료")
                break

# 두 명의 클라이언트를 동시에 실행
async def main():
    await asyncio.gather(simulate_client("Hospital-A"), simulate_client("Hospital-B"))

if __name__ == "__main__":
    asyncio.run(main())