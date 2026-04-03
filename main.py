import asyncio
import json
import os
import zipfile
import shutil
import datetime
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI(title="HELIOS AI-FL Server")

SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8081")
sessions: Dict[str, Dict] = {}

# 백엔드 학습 시작 요청 모델
class TrainStartRequest(BaseModel):
    participants: List[int]
    rounds: int

# 가중치 평균화 알고리즘 (FedAvg)
def federated_averaging(weights_list):
    if not weights_list: return []
    new_weights = [np.array(w, dtype=np.float64) for w in weights_list[0]]
    for other_weights in weights_list[1:]:
        for i, w in enumerate(other_weights):
            new_weights[i] += np.array(w, dtype=np.float64)
    averaged_weights = [w / len(weights_list) for w in new_weights]
    return [w.tolist() for w in averaged_weights]

# 유저 참여 시 백엔드에 알림 (참여 카운트 증가용)
async def notify_backend_join(session_id: str, user_token: str, user_id: str):
    print(f"📤 [AI -> Backend] 조인 알림: 세션 {session_id}, 유저 {user_id}")
    async with httpx.AsyncClient() as client:
        try:
            payload = {"userId": int(user_id), "labelingToken": ""} 
            response = await client.post(f"{SPRING_BOOT_URL}/sessions/{session_id}/join", json=payload)
            print(f"   ㄴ Backend 응답 코드: {response.status_code}")
        except Exception as e:
            print(f"   ㄴ ⚠️ 백엔드 통신 실패: {e}")

# 연합학습 핵심 루프
async def run_fl_loop(session_id: str, rounds: int = 5):
    if session_id not in sessions: return
    session_data = sessions[session_id]
    
    final_acc, final_loss = 0.0, 0.0
    print(f"\n🚀 [{session_id}] 연합학습 가동 (총 {rounds} 라운드)")
    
    try:
        for round_num in range(1, rounds + 1):
            clients = list(session_data["websockets"])
            if not clients:
                print(f"❌ [{session_id}] 참여자가 없어 학습을 중단합니다.")
                break

            # 1. 프론트엔드에 실시간 진행 상태 전송
            status_msg = json.dumps({
                "type": "status",
                "sessionId": int(session_id),
                "currentRound": round_num,
                "totalRounds": rounds,
                "progress": round((round_num / rounds) * 100, 1)
            })
            for client in clients:
                try: await client.send_text(status_msg)
                except: pass

            # 2. 글로벌 가중치 전송 및 로컬 학습 지시
            fit_msg = json.dumps({
                "type": "fit", 
                "parameters": session_data["global_weights"], 
                "config": {"epochs": 1}
            })
            
            active_clients = []
            for client in clients:
                try:
                    await client.send_text(fit_msg)
                    active_clients.append(client)
                except:
                    if client in session_data["websockets"]:
                        session_data["websockets"].remove(client)

            # 3. 로컬 학습 결과(가중치 및 메트릭) 수집
            collected_weights, collected_accs, collected_losses = [], [], []
            for client in active_clients:
                try:
                    res = await asyncio.wait_for(client.receive_text(), timeout=60.0)
                    data = json.loads(res)
                    if data.get("type") == "fit_res":
                        collected_weights.append(data["parameters"])
                        metrics = data.get("metrics", {})
                        collected_accs.append(metrics.get("accuracy", 0))
                        collected_losses.append(metrics.get("loss", 0))
                except Exception as e:
                    print(f"   ㄴ 수신 에러/타임아웃: {e}")
            
            # 4. 가중치 집계 및 글로벌 모델 업데이트
            if collected_weights:
                session_data["global_weights"] = federated_averaging(collected_weights)
                final_acc = sum(collected_accs) / len(collected_accs)
                final_loss = sum(collected_losses) / len(collected_losses)
                print(f"✅ Round {round_num} 완료 - Avg Acc: {final_acc:.4f}")
            
            await asyncio.sleep(1)
            
        print(f"🎉 [{session_id}] 학습 종료. S3 업로드 시작.")

        # ---------------------------------------------------------
        # 🌟 모델 압축 및 S3 업로드 프로세스 (요구사항 반영)
        # ---------------------------------------------------------
        run_date = datetime.datetime.now().strftime("%Y%m%d")
        file_name = f"{run_date}-Round{rounds}-v1.zip"
        zip_path = f"./{file_name}"
        
        # 가중치 파일 생성 및 압축
        json_temp = f"weights_{session_id}.json"
        with open(json_temp, "w") as f:
            json.dump(session_data["global_weights"], f)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(json_temp, arcname="model_weights.json")
        os.remove(json_temp)

        # 백엔드에 Presigned URL 요청 (POST /models/upload-url)
        async with httpx.AsyncClient() as client:
            url_payload = {"fileName": file_name, "sessionId": int(session_id)}
            url_res = await client.post(f"{SPRING_BOOT_URL}/models/upload-url", json=url_payload)
            
            if url_res.status_code == 200:
                upload_url = url_res.json().get("uploadUrl")
                
                # S3에 직접 업로드 (HTTP PUT 방식)
                with open(zip_path, "rb") as f:
                    headers = {
                        "x-amz-meta-accuracy": str(round(final_acc, 4)),
                        "x-amz-meta-loss": str(round(final_loss, 4)),
                        "x-amz-meta-modelid": str(session_id),
                        "Content-Type": "application/zip"
                    }
                    s3_res = await client.put(upload_url, content=f.read(), headers=headers)
                    if s3_res.status_code == 200:
                        print(f"🚀 S3 업로드 성공: {file_name}")
                    else:
                        print(f"❌ S3 업로드 실패: {s3_res.status_code}")
        
        if os.path.exists(zip_path): os.remove(zip_path)

    except Exception as e:
        print(f"🚨 학습 루프 에러: {e}")
    finally:
        if "done_event" in session_data:
            session_data["done_event"].set()

# 백엔드에서 호출하는 학습 시작 엔드포인트
@app.post("/train/start")
async def handle_start_command(req: TrainStartRequest):
    if not sessions:
        raise HTTPException(status_code=400, detail="생성된 세션이 없습니다.")
    
    session_id = list(sessions.keys())[-1]
    print(f"📩 [Backend -> AI] 세션 {session_id} 시작 명령 수신")
    
    asyncio.create_task(run_fl_loop(session_id, rounds=req.rounds))
    return {"status": "PROGRESS", "sessionId": session_id, "rounds": req.rounds}

# 웹소켓 엔드포인트
@app.websocket("/ws/fl/{session_id}/{user_token}")
async def websocket_endpoint(
    websocket: WebSocket, 
    session_id: str, 
    user_token: str, 
    userId: str = Query("1")
):
    await websocket.accept()
    
    if session_id not in sessions:
        sessions[session_id] = {
            "websockets": set(), 
            "global_weights": [], 
            "done_event": asyncio.Event() 
        }
        
    sessions[session_id]["websockets"].add(websocket)
    print(f"✅ [WebSocket] {user_token} 입장 (유저ID: {userId})")
    
    # 조인 사실 알림 (백엔드가 참여 인원을 파악하여 /train/start를 쏘게 함)
    await notify_backend_join(session_id, user_token, userId)
    
    try:
        # 학습 종료 이벤트가 올 때까지 대기
        await sessions[session_id]["done_event"].wait()
    except Exception as e:
        print(f"⚠️ {user_token} 연결 오류: {e}")
    finally:
        if session_id in sessions:
            if websocket in sessions[session_id]["websockets"]:
                sessions[session_id]["websockets"].remove(websocket)
        try:
            print(f"🚪 [WebSocket] {user_token} 퇴장")
            await websocket.close()
        except: pass
