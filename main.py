import asyncio
import json
import os
import zipfile
import datetime
from pathlib import Path
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def load_local_env(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')

        if key and key not in os.environ:
            os.environ[key] = value


load_local_env(ENV_FILE)

app = FastAPI(title="HELIOS AI-FL Server")


def parse_cors_origins() -> List[str]:
    raw_origins = os.getenv("CORS_ORIGINS")
    if raw_origins:
        return [
            origin.strip()
            for origin in raw_origins.split(",")
            if origin.strip()
        ]

    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


CORS_ORIGINS = parse_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPRING_BOOT_URL = os.getenv("SPRING_BOOT_URL", "http://localhost:8081")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
GEMINI_REPORT_MODEL = os.getenv("GEMINI_REPORT_MODEL", "gemini-2.5-flash")
sessions: Dict[str, Dict] = {}
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 백엔드 학습 시작 요청 모델
class TrainStartRequest(BaseModel):
    participants: List[int]
    rounds: int


class DiagnosticResultItem(BaseModel):
    name: str
    score: float


class DiagnosticReportRequest(BaseModel):
    sessionId: Optional[str] = None
    generatedAt: Optional[str] = None
    modelId: Optional[str] = None
    modelTitle: str
    domainLabel: str
    imageFileName: Optional[str] = None
    results: List[DiagnosticResultItem]
    notes: Optional[str] = None
    locale: str = "ko-KR"


class DiagnosticReportResponse(BaseModel):
    generatedAt: str
    provider: str
    model: str
    summary: str
    findings: str
    recommendations: List[str]
    caution: str
    draft: str
    storedPath: str


REPORT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "string"},
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "caution": {"type": "string"},
        "draft": {"type": "string"},
    },
    "required": ["summary", "findings", "recommendations", "caution", "draft"],
}


def normalize_domain(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if any(token in value for token in ["fundus", "retina", "dr"]):
        return "fundus"
    if any(token in value for token in ["x-ray", "xray", "chex", "chest"]):
        return "xray"
    return "unknown"


def get_session_state(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "websockets": set(),
            "global_weights": [],
            "done_event": asyncio.Event(),
            "clients": {},
            "session_meta": {},
            "expected_domain": "unknown",
            "review_cases": [],
            "review_report": None,
            "start_requested": False,
            "start_in_progress": False,
            "requested_rounds": 5,
            "expected_participants": [],
        }
    return sessions[session_id]


def maybe_start_requested_session(session_id: str) -> bool:
    session_state = get_session_state(session_id)
    if not session_state.get("start_requested"):
        return False
    if session_state.get("start_in_progress"):
        return False

    expected_participants = {
        str(participant_id) for participant_id in session_state.get("expected_participants", [])
    }
    connected_participants = {
        str(client_state.get("userId"))
        for client_state in session_state.get("clients", {}).values()
        if client_state.get("userId") is not None
    }

    if expected_participants and not expected_participants.issubset(connected_participants):
        print(
            f"⏳ [{session_id}] 시작 대기 중: connected={sorted(connected_participants)}, "
            f"expected={sorted(expected_participants)}"
        )
        return False

    session_state["start_in_progress"] = True
    rounds = int(session_state.get("requested_rounds") or 5)
    print(f"🚦 [{session_id}] 시작 조건 충족. 연합학습을 시작합니다. (rounds={rounds})")
    asyncio.create_task(run_fl_loop(session_id, rounds=rounds))
    return True


def report_path_for_session(session_id: str) -> Path:
    return REPORTS_DIR / f"session_{session_id}_screening_report.json"


def llm_report_path(session_id: Optional[str]) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_session_id = session_id or "playground"
    return REPORTS_DIR / f"llm_report_{safe_session_id}_{timestamp}.json"


def build_report_payload(request: DiagnosticReportRequest) -> Dict[str, Any]:
    sorted_results = sorted(request.results, key=lambda item: item.score, reverse=True)
    top_results = [
        {"name": item.name, "score": round(float(item.score), 1)}
        for item in sorted_results[:5]
    ]
    return {
        "sessionId": request.sessionId,
        "generatedAt": request.generatedAt,
        "modelId": request.modelId,
        "modelTitle": request.modelTitle,
        "domainLabel": request.domainLabel,
        "imageFileName": request.imageFileName,
        "topResults": top_results,
        "notes": request.notes or "",
        "locale": request.locale,
    }


def extract_response_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response candidates are missing")

    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise ValueError("Gemini response candidate is invalid")

    content = first_candidate.get("content", {})
    if not isinstance(content, dict):
        raise ValueError("Gemini response content is invalid")

    parts = content.get("parts", [])
    if not isinstance(parts, list):
        raise ValueError("Gemini response parts are missing")

    collected: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text_value = part.get("text")
        if isinstance(text_value, str) and text_value.strip():
            collected.append(text_value)

    if not collected:
        raise ValueError("Gemini response text is empty")

    return "\n".join(collected)


async def create_llm_report(request: DiagnosticReportRequest) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY or GOOGLE_API_KEY is not configured on the AI server.",
        )

    if not request.results:
        raise HTTPException(status_code=400, detail="results must contain at least one item")

    normalized_request = build_report_payload(request)
    endpoint = (
        f"{GEMINI_BASE_URL.rstrip('/')}/models/{GEMINI_REPORT_MODEL}:generateContent"
    )
    instructions = (
        "당신은 의료진의 AI 판독 보조 시스템이다. "
        "반드시 한국어로 작성하고, 제공된 정보만 사용해 진단 리포트 초안을 생성한다. "
        "환자 정보나 검사 정보를 임의로 지어내지 말고, 근거가 없는 확정 진단 표현을 피한다. "
        "recommendations는 2~4개의 짧은 문장 배열로 작성한다. "
        "draft는 실제 의료 리포트 초안처럼 자연스러운 문단 형태로 작성한다. "
        "항상 AI 보조 결과이며 최종 판단은 의료진이 해야 한다는 주의를 포함한다."
    )

    request_body = {
        "system_instruction": {
            "parts": [
                {
                    "text": instructions,
                }
            ]
        },
        "contents": [
            {
                "parts": [
                    {
                        "text": json.dumps(normalized_request, ensure_ascii=False, indent=2),
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": REPORT_RESPONSE_SCHEMA,
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            endpoint,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=request_body,
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini report generation failed: {response.text}",
        )

    raw_payload = response.json()
    try:
        parsed = json.loads(extract_response_text(raw_payload))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse Gemini report payload: {exc}",
        ) from exc

    return parsed


class ScreeningReviewAgent:
    report_version = "v1"

    def analyze_case(self, session_id: str, expected_domain: str, case: Dict[str, Any]) -> Dict[str, Any]:
        event_type = case.get("eventType", "unknown")
        user_id = case.get("userId")

        if event_type == "admission_rejected":
            detected_domain = case.get("detectedDomain", "unknown")
            compatibility = case.get("domainScore", 0)
            return {
                "caseId": case["caseId"],
                "eventType": event_type,
                "severity": "high",
                "decision": "block",
                "summary": (
                    f"클라이언트 {user_id}는 세션 기대 도메인 {expected_domain}과 다른 "
                    f"{detected_domain} 특성을 보여 admission 단계에서 차단됐다."
                ),
                "evidence": {
                    "expectedDomain": expected_domain,
                    "detectedDomain": detected_domain,
                    "domainScore": compatibility,
                    "reason": case.get("reason"),
                },
                "recommendedAction": "입력 데이터 원본과 세션 타입을 다시 확인하고 동일 도메인 데이터로만 재참여시킨다.",
            }

        if event_type == "slow_client_excluded":
            duration_ms = case.get("durationMs")
            threshold_ms = case.get("thresholdMs")
            return {
                "caseId": case["caseId"],
                "eventType": event_type,
                "severity": "medium",
                "decision": "quarantine_round",
                "summary": (
                    f"클라이언트 {user_id}는 round {case.get('round')}에서 응답 지연이 커 집계에서 제외됐다."
                ),
                "evidence": {
                    "durationMs": duration_ms,
                    "thresholdMs": threshold_ms,
                    "slowStrikes": case.get("slowStrikes", 0),
                },
                "recommendedAction": "동일 현상이 반복되면 학습 대상에서 장기 격리하고 클라이언트 성능 또는 네트워크 상태를 점검한다.",
            }

        if event_type == "outlier_update_excluded":
            return {
                "caseId": case["caseId"],
                "eventType": event_type,
                "severity": "high",
                "decision": "quarantine_round",
                "summary": (
                    f"클라이언트 {user_id}는 round {case.get('round')}에서 집단 중심과 크게 다른 업데이트를 보내 집계에서 제외됐다."
                ),
                "evidence": {
                    "cosineSimilarity": case.get("cosineSimilarity"),
                    "updateNorm": case.get("updateNorm"),
                    "normMedian": case.get("normMedian"),
                    "outlierStrikes": case.get("outlierStrikes", 0),
                },
                "recommendedAction": "데이터 도메인 혼입 또는 조작 가능성을 검토하고 동일 패턴이 반복되면 세션 전체에서 차단한다.",
            }

        return {
            "caseId": case["caseId"],
            "eventType": event_type,
            "severity": "low",
            "decision": "review",
            "summary": f"클라이언트 {user_id} 관련 screening 이벤트가 감지됐다.",
            "evidence": case,
            "recommendedAction": "운영자가 세부 로그를 검토한다.",
        }

    def build_report(self, session_id: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        expected_domain = session_state.get("expected_domain", "unknown")
        reviewed_cases = [
            self.analyze_case(session_id, expected_domain, case)
            for case in session_state.get("review_cases", [])
        ]
        counts = {
            "totalCases": len(reviewed_cases),
            "blocked": sum(1 for case in reviewed_cases if case["decision"] == "block"),
            "quarantined": sum(1 for case in reviewed_cases if case["decision"] == "quarantine_round"),
            "reviewOnly": sum(1 for case in reviewed_cases if case["decision"] == "review"),
        }
        return {
            "sessionId": session_id,
            "expectedDomain": expected_domain,
            "reportVersion": self.report_version,
            "generatedAt": datetime.datetime.now().isoformat(),
            "counts": counts,
            "cases": reviewed_cases,
        }


review_agent = ScreeningReviewAgent()


def persist_review_report(session_id: str):
    session_state = get_session_state(session_id)
    report = review_agent.build_report(session_id, session_state)
    session_state["review_report"] = report
    report_path_for_session(session_id).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_review_case(session_id: str, case: Dict[str, Any]):
    session_state = get_session_state(session_id)
    case_with_meta = {
        "caseId": f"{session_id}-{len(session_state['review_cases']) + 1}",
        "recordedAt": datetime.datetime.now().isoformat(),
        **case,
    }
    session_state["review_cases"].append(case_with_meta)
    persist_review_report(session_id)


async def ensure_session_contract(session_id: str):
    session_state = get_session_state(session_id)
    if session_state.get("session_meta"):
        return

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SPRING_BOOT_URL}/sessions/{session_id}")
            if response.status_code == 200:
                session_meta = response.json()
                session_state["session_meta"] = session_meta
                session_state["expected_domain"] = normalize_domain(session_meta.get("dataFormat"))
        except Exception as e:
            print(f"⚠️ [{session_id}] 세션 메타데이터 조회 실패: {e}")


def assess_client_admission(expected_domain: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    detected_domain = normalize_domain(profile.get("detectedDomain"))
    compatibility_score = float(profile.get("domainScore") or 0)
    if expected_domain == "unknown":
        return {"accepted": True, "reason": "session domain unknown; screening relaxed"}

    if detected_domain not in [expected_domain, "unknown"]:
        return {
            "accepted": False,
            "reason": f"domain mismatch: expected={expected_domain}, detected={detected_domain}",
        }

    if compatibility_score < 0.58:
        return {
            "accepted": False,
            "reason": f"domain compatibility too low ({compatibility_score:.2f}) for {expected_domain}",
        }

    return {
        "accepted": True,
        "reason": f"domain accepted ({expected_domain}, score={compatibility_score:.2f})",
    }


def flatten_weights(weights: List[List[float]]) -> np.ndarray:
    if not weights:
        return np.array([], dtype=np.float64)
    flattened = [np.asarray(layer, dtype=np.float64).reshape(-1) for layer in weights]
    return np.concatenate(flattened) if flattened else np.array([], dtype=np.float64)


def filter_round_responses(session_data: Dict[str, Any], responses: List[Dict[str, Any]], round_num: int) -> List[Dict[str, Any]]:
    if not responses:
        return []

    durations = [
        float(item["metrics"].get("fitDurationMs") or item["metrics"].get("serverDurationMs") or 0)
        for item in responses
    ]
    duration_median = float(np.median(durations)) if durations else 0.0
    slow_threshold = max(45000.0, duration_median * 2.5) if duration_median > 0 else 45000.0

    survivors: List[Dict[str, Any]] = []
    for item, duration in zip(responses, durations):
        client_state = session_data["clients"].get(item["client"], {})
        if duration > slow_threshold:
            client_state["slow_strikes"] = client_state.get("slow_strikes", 0) + 1
            print(
                f"⚠️ [Screening] Round {round_num}: slow client excluded "
                f"(userId={client_state.get('userId')}, duration={duration:.1f}ms, threshold={slow_threshold:.1f}ms)"
            )
            append_review_case(
                str(session_data["session_meta"].get("sessionId") or session_data.get("session_id")),
                {
                    "eventType": "slow_client_excluded",
                    "userId": client_state.get("userId"),
                    "round": round_num,
                    "durationMs": round(duration, 2),
                    "thresholdMs": round(slow_threshold, 2),
                    "slowStrikes": client_state["slow_strikes"],
                },
            )
            continue
        client_state["slow_strikes"] = 0
        survivors.append(item)

    if len(survivors) < 3:
        return survivors

    vectors = [flatten_weights(item["weights"]) for item in survivors]
    norms = [float(np.linalg.norm(vector)) for vector in vectors]
    norm_median = float(np.median(norms)) if norms else 0.0

    filtered: List[Dict[str, Any]] = []
    for index, (item, vector, norm_value) in enumerate(zip(survivors, vectors, norms)):
        if vector.size == 0:
            continue
        peer_vectors = [peer for peer_idx, peer in enumerate(vectors) if peer_idx != index and peer.size == vector.size]
        if not peer_vectors:
            filtered.append(item)
            continue
        peer_centroid = np.mean(peer_vectors, axis=0)
        peer_norm = float(np.linalg.norm(peer_centroid)) or 1.0
        cosine = float(np.dot(vector, peer_centroid) / ((np.linalg.norm(vector) or 1.0) * peer_norm))
        client_state = session_data["clients"].get(item["client"], {})
        if cosine < 0.2 and norm_median > 0 and norm_value > norm_median * 1.8:
            client_state["outlier_strikes"] = client_state.get("outlier_strikes", 0) + 1
            print(
                f"⚠️ [Screening] Round {round_num}: outlier update excluded "
                f"(userId={client_state.get('userId')}, cosine={cosine:.3f}, norm={norm_value:.3f})"
            )
            append_review_case(
                str(session_data["session_meta"].get("sessionId") or session_data.get("session_id")),
                {
                    "eventType": "outlier_update_excluded",
                    "userId": client_state.get("userId"),
                    "round": round_num,
                    "cosineSimilarity": round(cosine, 4),
                    "updateNorm": round(norm_value, 4),
                    "normMedian": round(norm_median, 4),
                    "outlierStrikes": client_state["outlier_strikes"],
                },
            )
            continue
        client_state["outlier_strikes"] = 0
        filtered.append(item)

    return filtered

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
                "config": {
                    "epochs": 1,
                    "round": round_num,
                    "expectedDomain": session_data.get("expected_domain", "unknown"),
                }
            })
            
            active_clients = []
            for client in clients:
                try:
                    await client.send_text(fit_msg)
                    session_data["clients"].setdefault(client, {})["last_fit_request_at"] = asyncio.get_running_loop().time()
                    active_clients.append(client)
                except:
                    if client in session_data["websockets"]:
                        session_data["websockets"].remove(client)

            # 3. 로컬 학습 결과(가중치 및 메트릭) 수집
            responses = []
            for client in active_clients:
                try:
                    res = await asyncio.wait_for(client.receive_text(), timeout=60.0)
                    data = json.loads(res)
                    if data.get("type") == "fit_res":
                        metrics = data.get("metrics", {})
                        started_at = session_data["clients"].get(client, {}).get("last_fit_request_at")
                        if started_at is not None:
                            metrics["serverDurationMs"] = round((asyncio.get_running_loop().time() - started_at) * 1000, 2)
                        responses.append({
                            "client": client,
                            "weights": data["parameters"],
                            "metrics": metrics,
                        })
                except Exception as e:
                    print(f"   ㄴ 수신 에러/타임아웃: {e}")

            filtered_responses = filter_round_responses(session_data, responses, round_num)
            collected_weights = [item["weights"] for item in filtered_responses]
            collected_accs = [float(item["metrics"].get("accuracy", 0)) for item in filtered_responses]
            collected_losses = [float(item["metrics"].get("loss", 0)) for item in filtered_responses]
            
            # 4. 가중치 집계 및 글로벌 모델 업데이트
            if collected_weights:
                session_data["global_weights"] = federated_averaging(collected_weights)
                final_acc = sum(collected_accs) / len(collected_accs)
                final_loss = sum(collected_losses) / len(collected_losses)
                print(f"✅ Round {round_num} 완료 - Avg Acc: {final_acc:.4f} (accepted {len(collected_weights)}/{len(responses)})")
            else:
                print(f"⚠️ [{session_id}] Round {round_num}: surviving client update가 없어 집계를 건너뜁니다.")
            
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

async def start_training_for_session(session_id: str, req: TrainStartRequest):
    session_state = get_session_state(session_id)
    session_state["start_requested"] = True
    session_state["requested_rounds"] = req.rounds
    session_state["expected_participants"] = req.participants
    print(f"📩 [Backend -> AI] 세션 {session_id} 시작 명령 수신")
    started = maybe_start_requested_session(session_id)
    status = "PROGRESS" if started else "WAITING_FOR_CLIENTS"
    return {"status": status, "sessionId": session_id, "rounds": req.rounds}


# 레거시 시작 엔드포인트
@app.post("/train/start")
async def handle_start_command(req: TrainStartRequest):
    if not sessions:
        raise HTTPException(status_code=400, detail="생성된 세션이 없습니다.")

    session_id = list(sessions.keys())[-1]
    return await start_training_for_session(session_id, req)


# 백엔드가 사용하는 세션 스코프 시작 엔드포인트
@app.post("/sessions/{session_id}/start")
async def handle_session_start_command(session_id: str, req: TrainStartRequest):
    return await start_training_for_session(session_id, req)


@app.get("/sessions/{session_id}/screening-report")
async def get_screening_report(session_id: str):
    session_state = sessions.get(session_id)
    if session_state and session_state.get("review_report") is not None:
        return session_state["review_report"]

    report_path = report_path_for_session(session_id)
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail="screening report not found")


@app.post("/reports/diagnostic-draft", response_model=DiagnosticReportResponse)
async def generate_diagnostic_report(req: DiagnosticReportRequest):
    llm_result = await create_llm_report(req)
    generated_at = datetime.datetime.now().isoformat()
    stored_path = llm_report_path(req.sessionId)
    response_payload = {
        "generatedAt": generated_at,
        "provider": "gemini",
        "model": GEMINI_REPORT_MODEL,
        "summary": llm_result["summary"],
        "findings": llm_result["findings"],
        "recommendations": llm_result["recommendations"],
        "caution": llm_result["caution"],
        "draft": llm_result["draft"],
        "storedPath": str(stored_path.relative_to(REPORTS_DIR.parent)),
    }
    stored_path.write_text(
        json.dumps(
            {
                "request": build_report_payload(req),
                "response": response_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return response_payload

# 웹소켓 엔드포인트
@app.websocket("/ws/fl/{session_id}/{user_token}")
async def websocket_endpoint(
    websocket: WebSocket, 
    session_id: str, 
    user_token: str, 
    userId: str = Query("1")
):
    await websocket.accept()
    await ensure_session_contract(session_id)
    session_state = get_session_state(session_id)

    try:
        raw_hello = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        hello = json.loads(raw_hello)
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "admission",
            "accepted": False,
            "reason": f"client_hello missing or invalid: {e}",
        }))
        await websocket.close(code=1008)
        return

    if hello.get("type") != "client_hello":
        await websocket.send_text(json.dumps({
            "type": "admission",
            "accepted": False,
            "reason": "first websocket message must be client_hello",
        }))
        await websocket.close(code=1008)
        return

    profile = hello.get("profile", {})
    admission = assess_client_admission(session_state.get("expected_domain", "unknown"), profile)
    await websocket.send_text(json.dumps({"type": "admission", **admission}))
    if not admission["accepted"]:
        print(f"⛔ [WebSocket] {user_token} admission rejected: {admission['reason']}")
        append_review_case(
            session_id,
            {
                "eventType": "admission_rejected",
                "userId": userId,
                "expectedDomain": session_state.get("expected_domain", "unknown"),
                "detectedDomain": normalize_domain(profile.get("detectedDomain")),
                "domainScore": float(profile.get("domainScore") or 0),
                "reason": admission["reason"],
            },
        )
        await websocket.close(code=1008)
        return

    session_state["websockets"].add(websocket)
    session_state["session_id"] = session_id
    session_state["clients"][websocket] = {
        **profile,
        "userId": userId,
        "userToken": user_token,
        "slow_strikes": 0,
        "outlier_strikes": 0,
    }
    print(
        f"✅ [WebSocket] {user_token} 입장 (유저ID: {userId}, expected={session_state.get('expected_domain')}, "
        f"detected={profile.get('detectedDomain')}, score={profile.get('domainScore')})"
    )
    maybe_start_requested_session(session_id)
    
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
            sessions[session_id]["clients"].pop(websocket, None)
        try:
            print(f"🚪 [WebSocket] {user_token} 퇴장")
            await websocket.close()
        except: pass
