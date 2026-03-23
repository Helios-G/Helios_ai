import asyncio
import json
import os
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI(title="HELIOS AI-FL Server")

SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8081")
sessions: Dict[str, Dict] = {}

class TrainStartRequest(BaseModel):
    participants: List[int]

def federated_averaging(weights_list):
    if not weights_list: return []
    new_weights = [np.array(w, dtype=np.float64) for w in weights_list[0]]
    for other_weights in weights_list[1:]:
        for i, w in enumerate(other_weights):
            new_weights[i] += np.array(w, dtype=np.float64)
    averaged_weights = [w / len(weights_list) for w in new_weights]
    return [w.tolist() for w in averaged_weights]

async def notify_backend_join(session_id: str, user_token: str, hospital_id: str):
    print(f"📤 [AI -> Backend] 조인 알림: 세션 {session_id}, 병원 {hospital_id}")
    async with httpx.AsyncClient() as client:
        try:
            # hospital_id를 숫자로 변환하여 전송
            h_id = int(str(hospital_id).replace("H", ""))
            payload = {
                "hospitalId": h_id,
                "labelingToken": f"ai-signed-{user_token}"
            }
            response = await client.post(f"{SPRING_BOOT_URL}/sessions/{session_id}/join", json=payload)
            print(f"   ㄴ Backend 응답 코드: {response.status_code}")
        except Exception as e:
            print(f"   ㄴ ⚠️ 백엔드 통신 실패: {e}")

async def run_fl_loop(session_id: str, rounds: int = 5):
    if session_id not in sessions: return
    session_data = sessions[session_id]
    clients = session_data["websockets"]
    
    print(f"\n🚀 [{session_id}] 백엔드 신호 수신! 연합학습 루프를 가동합니다.")
    
    for round_num in range(1, rounds + 1):
        print(f"🔄 Round {round_num}/{rounds} 진행 중...")
        fit_msg = json.dumps({
            "type": "fit", 
            "parameters": session_data["global_weights"], 
            "config": {"epochs": 1}
        })
        
        current_clients = list(clients)
        for client in current_clients:
            await client.send_text(fit_msg)
            
        collected_weights = []
        for client in current_clients:
            try:
                res = await client.receive_text()
                data = json.loads(res)
                if data.get("type") == "fit_res":
                    collected_weights.append(data["parameters"])
            except: pass
                
        if collected_weights:
            session_data["global_weights"] = federated_averaging(collected_weights)
            print(f"✅ Round {round_num} 집계 완료")
            
        await asyncio.sleep(1)
        
    print(f"🎉 [{session_id}] 학습 완료 및 세션 종료")
    for client in list(clients):
        await client.close()
    if session_id in sessions:
        del sessions[session_id]

# ---------------------------------------------------------
# ✅ [명세서 3번] 백엔드로부터 학습 시작 신호를 받는 엔드포인트
# ---------------------------------------------------------
@app.post("/sessions/{session_id}/start")
async def handle_start_command(session_id: str, req: TrainStartRequest):
    print(f"📩 [Backend -> AI] 세션 {session_id} 학습 시작 명령 수신!")
    
    if session_id not in sessions or not sessions[session_id]["websockets"]:
        raise HTTPException(status_code=400, detail="대기 중인 클라이언트가 없습니다.")
    
    # 비동기로 학습 루프 시작
    asyncio.create_task(run_fl_loop(session_id))
    
    return {"status": "PROGRESS", "message": "Started by Backend signal"}

# ---------------------------------------------------------
# [웹소켓] 클라이언트 대기실
# ---------------------------------------------------------
@app.websocket("/ws/fl/{session_id}/{user_token}")
async def websocket_endpoint(
    websocket: WebSocket, 
    session_id: str, 
    user_token: str, 
    hospitalId: str = Query("1")
):
    await websocket.accept()
    
    if session_id not in sessions:
        sessions[session_id] = {"websockets": set(), "global_weights": [], "status": "WAITING"}
        
    sessions[session_id]["websockets"].add(websocket)
    print(f"✅ [WebSocket] {user_token} 대기실 입장 (현재 {len(sessions[session_id]['websockets'])}명)")
    
    # 조인 알림 (흐름 2번)
    await notify_backend_join(session_id, user_token, hospitalId)
    
    try:
        # 백엔드에서 /start 명령을 쏠 때까지 연결만 유지하고 무한 대기
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if session_id in sessions:
            sessions[session_id]["websockets"].remove(websocket)
        print(f"👋 [WebSocket] {user_token} 퇴장")