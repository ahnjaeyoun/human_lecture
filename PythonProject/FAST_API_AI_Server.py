# =========================================================
# FastAPI AI 서버
# - 클라이언트로부터 분석 요청을 받아
# - 길이, 감성, 키워드 탐지 분석 결과를 반환하는 예제
# =========================================================

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# =========================================================
# 1. 감성 분석 모델 로드
# =========================================================
# Hugging Face의 사전 학습된 모델을 사용
sentiment_analyzer = pipeline("sentiment-analysis")   # 모델 결과 다운로드

# =========================================================
# 2. FAST API 앱 생성
# =========================================================
app = FastAPI(title="ai 분석 서버")

# =========================================================
# 3. 요청 데이터 구조 정의(Pydantic)
# =========================================================
# 클라이언트가 보낸 json 데이터를 자동 검증 및 파싱
class AnalysisRequest(BaseModel):
    mode: str  # 분석 모드 (length / sentiment / keyword)
    text: str  # 분석할 문장

# =========================================================
# 4. 분석 API 앤드포인트 정의
# =========================================================
@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    # 요청값 일기
    mode = request.mode.lower()
    text = request.text

    # 1) 문장 길이 분석
    if mode == "length":
        result = {
            "result" : len(text),
            "desc" : f"문장 길이는 {len(text)}자 입니다."
        }

    # 2) 감성 분석(transformers 이용)
    elif mode == "sentiment":
        analysis = sentiment_analyzer(text)[0]           # 감성 결과
        label = analysis["label"]
        score = round(analysis["score"], 3)
        # 신뢰도 (0~1)
        result = {
            "result" : label,
            "confidence" : score,
            "desc" : f"감정: {label}, 신뢰도 : {score}"
        }
    # 3) 키워드 탐지
    elif mode == 'keywords':
        keywords = ["ai","press","factory","defect","data","불량"]
        found = [w for w in keywords if w.lower() in text.lower()]
        result = {
            "result" : found,
            "desc": f"키워드 발견 : {','.join(found) if found else '없음'}"
        }
    # 4) 지원하지 않는 모드 처리
    else:
        result = {
            "error": f"지원하지 않는 모드입니다: {mode}"
        }
    return result  # json 결과 반환

# =========================================================
# 5. 서버 실행부
# =========================================================

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.47", port=8000)


