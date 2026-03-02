import asyncio
import websockets
import json
        {/* 2. 가이드 박스 */}
        <div className="p-8 mb-12 rounded-lg border border-[#F5E6D3]" style={{ backgroundColor: '#FFF9F5' }}>
          <h3 className="text-lg font-bold mb-4" style={{ color: '#6B3131' }}>연합학습 참여 가이드</h3>
          <div className="space-y-2 text-gray-700 text-sm">
            <p>원하는 세션의 참여하기 버튼을 클릭 후, 안내에 따라 라벨링을 진행합니다.</p>
            <p>조건 기관 수가 채워지면 학습이 자동 시작됩니다.</p>
            <p>완료된 목록은 모델 다운로드 페이지에서 다운받아 사용이 가능합니다.</p>
          </div>
        </div>


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