import asyncio
import websockets
import json
import numpy as np

connected_clients = set()
MIN_CLIENTS = 2
TOTAL_ROUNDS = 5

def federated_averaging(weights_list):
    if not weights_list: return []
    new_weights = [np.array(w) for w in weights_list[0]]
    for other_weights in weights_list[1:]:
        for i, w in enumerate(other_weights):
            new_weights[i] += np.array(w)
    averaged_weights = [w / len(weights_list) for w in new_weights]
    return [w.tolist() for w in averaged_weights]

async def register(websocket):
    connected_clients.add(websocket)
    print(f"✅ 클라이언트 접속! (현재 {len(connected_clients)}명)")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print("⚠️ 클라이언트 연결 해제")

async def training_coordinator():
    print(f"⏳ {MIN_CLIENTS}명의 클라이언트를 기다리는 중...")
    
    while len(connected_clients) < MIN_CLIENTS:
        await asyncio.sleep(1)
    
    print("\n🚀 목표 인원 달성! 3초 후 연합학습을 시작합니다...")
    await asyncio.sleep(3)

    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"\n🔄 --- Round {round_num}/{TOTAL_ROUNDS} Start ---")
        
        fit_msg = json.dumps({
            "type": "fit", 
            "parameters": [], 
            "config": {"epochs": 1}
        })
        
        # ✅ [수정 핵심] 접속자 목록을 복사(list)해서 사용 -> 에러 방지
        current_clients = list(connected_clients)
        
        if len(current_clients) == 0:
            print("❌ 접속된 클라이언트가 없습니다. 대기 중...")
            await asyncio.sleep(2)
            continue

        # 복사한 목록으로 전송
        websockets.broadcast(current_clients, fit_msg)
        
        collected_weights = []
        
        # 복사한 목록으로 응답 대기
        for client in current_clients:
            try:
                res = await client.recv()
                data = json.loads(res)
                if data.get("type") == "fit_res":
                    collected_weights.append(data["parameters"])
                    print(f"  📥 클라이언트 응답 수신 완료")
            except:
                print("  ❌ 응답 대기 중 에러 (무시)")

        if collected_weights:
            global_weights = federated_averaging(collected_weights)
            print(f"✅ Round {round_num} 집계 완료!")
        
        await asyncio.sleep(1)

    print("\n🎉 모든 라운드 종료! 수고하셨습니다.")
    # ✅ [추가할 부분] 학습이 끝났으니 클라이언트 연결 종료!
    print("🔌 클라이언트 연결을 종료합니다...")
    for client in list(connected_clients):
        await client.close()
        
async def main():
    server = await websockets.serve(register, "0.0.0.0", 8080, max_size=None, ping_interval=None)
    await asyncio.gather(server.wait_closed(), training_coordinator())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("서버 종료")