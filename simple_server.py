import asyncio
import websockets
import json
import numpy as np

connected_clients = set()
MIN_CLIENTS = 2
TOTAL_ROUNDS = 5

def federated_averaging(weights_list):
    """클라이언트들의 가중치를 평균내는 함수"""
    if not weights_list: return []
    # 첫 번째 클라이언트의 가중치를 기준으로 초기화
    new_weights = [np.array(w, dtype=np.float64) for w in weights_list[0]]
    
    # 나머지 클라이언트들의 가중치를 더함
    for other_weights in weights_list[1:]:
        for i, w in enumerate(other_weights):
            new_weights[i] += np.array(w)
            
    # 전체 개수로 나누어 평균 계산
    averaged_weights = [w / len(weights_list) for w in new_weights]
    return [w.tolist() for w in averaged_weights]

async def register(websocket):
    """클라이언트 접속 처리"""
    connected_clients.add(websocket)
    print(f"✅ 클라이언트 접속! (현재 {len(connected_clients)}명)")
    try:
        await websocket.wait_closed()
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print("⚠️ 클라이언트 연결 해제")

async def training_coordinator():
    # 🔄 [수정 1] 무한 루프로 감싸서 서버가 죽지 않고 계속 다음 세션을 준비하게 함
    while True:
        print(f"\n⏳ [새로운 세션] {MIN_CLIENTS}명의 클라이언트를 기다리는 중...")
        
        # 최소 인원이 찰 때까지 대기
        while len(connected_clients) < MIN_CLIENTS:
            await asyncio.sleep(1)
        
        print("\n🚀 목표 인원 달성! 3초 후 연합학습을 시작합니다...")
        await asyncio.sleep(3)

        # 글로벌 가중치 초기화 (새 세션 시작 시)
        global_weights = []

        for round_num in range(1, TOTAL_ROUNDS + 1):
            print(f"\n🔄 --- Round {round_num}/{TOTAL_ROUNDS} Start ---")
            
            # 🔄 [수정 2] 이전 라운드에서 집계된 global_weights를 전송해야 함!
            # (첫 라운드는 빈 리스트, 이후부터는 평균낸 값 전송)
            fit_msg = json.dumps({
                "type": "fit", 
                "parameters": global_weights, 
                "config": {"epochs": 1}
            })
            
            # 접속자 목록 복사 (중간에 끊기는 경우 방지)
            current_clients = list(connected_clients)
            
            if len(current_clients) == 0:
                print("❌ 접속된 클라이언트가 없습니다. 세션 초기화...")
                break # 이번 세션 중단하고 다시 대기 상태로

            # 학습 요청 전송
            await asyncio.gather(
                *[client.send(fit_msg) for client in current_clients],
                return_exceptions=True
            )
            
            collected_weights = []
            
            # 응답 대기 (타임아웃 설정 권장하지만 여기선 단순화)
            for client in current_clients:
                try:
                    res = await client.recv()
                    data = json.loads(res)
                    if data.get("type") == "fit_res":
                        collected_weights.append(data["parameters"])
                        print(f"  📥 클라이언트 응답 수신 완료")
                except:
                    print("  ❌ 응답 대기 중 에러 (클라이언트 이탈)")

            # 가중치 집계 (Federated Averaging)
            if collected_weights:
                global_weights = federated_averaging(collected_weights)
                print(f"✅ Round {round_num} 집계 완료! (가중치 갱신됨)")
            else:
                print("⚠️ 이번 라운드에 수신된 가중치가 없습니다.")
            
            await asyncio.sleep(1)

        print("\n🎉 모든 라운드 종료! 이번 세션 학습 완료.")
        
        # 연결 종료 및 다음 세션 준비
        print("🔌 클라이언트 연결을 종료하고 다음 세션을 준비합니다...")
        # 현재 연결된 클라이언트들에게 종료 신호를 보내거나 연결을 끊음
        for client in list(connected_clients):
            await client.close()
            
        # 잠시 대기 후 루프 처음으로 돌아감
        await asyncio.sleep(2)

async def main():
    # ping_interval=None으로 설정하여 연결 끊김 방지 (필요시 조정)
    server = await websockets.serve(register, "0.0.0.0", 8080, max_size=None, ping_interval=None)
    await asyncio.gather(server.wait_closed(), training_coordinator())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("서버 종료")