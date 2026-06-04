import io
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image


# CIFAR10 클래스 정의
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 학습된 모델 가져오기
print("AI 모델 가져오는 중...")
# 학습 시 저장했던 모델 읽기
model = tf.keras.models.load_model("./model/cifar10_model.h5")
print("AI 모델 로딩 완료!")

# FastAPI 앱 생성
app = FastAPI(title="CIFAR10 이미지 분류 분석 서버")

# 이미지 추론 함수
def predict_image(image_bytes: bytes) -> dict: # :bytes는 데이터 타입 힌트,
                                               # -> dict:는  return값이 dict라는 힌트
                                               # 없어도 상관 없다

    # 1. Byte -> PIL Image -> RGB 변환
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. CIFAR10 크기(32x32) 맞춤 및 numpy 배열 변환
    image = image.resize((32, 32))
    image = np.array(image)

    # 3. 정규화 (0 ~ 1)
    image = image / 255.0

    # 4. 배치 차원 추가 (Shape: (1, 32, 32, 3))
    image = np.expand_dims(image, axis=0)
    print("입력 Shape :", image.shape)

    # 5. 모델 추론
    pred = model.predict(image, verbose=0)

    # 6. 가장 높은 확률의 클래스 및 신뢰도 추출
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))

    print(f"예측 클래스: {class_idx} ({CLASS_NAMES[class_idx]}), 신뢰도: {confidence:.4f}")

    # 7. 결과 반환 딕셔너리 생성
    result = {
        "class_id": class_idx,
        "class_name": CLASS_NAMES[class_idx],
        "confidence": confidence,
    }
    return result


# =====================================================
# 이미지 예측 API 엔드포인트 정의
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """클라이언트로부터 이미지 파일을 받아 추론 결과를 반환합니다."""
    try:
        # 업로드된 파일의 바이너리 데이터 읽기
        image_bytes = await file.read()

        # AI 추론 실행
        result = predict_image(image_bytes)

        return {"success": True, "result": result}

    except Exception as e:
        print(f"오류 발생 : {e}")
        return {"success": False, "error": str(e)}


# =====================================================
# 서버 실행부
# =====================================================
if __name__ == "__main__":
    # 제공해주신 기존 IP와 FastAPI 포트(8000) 구성
    uvicorn.run(app, host="192.168.0.47", port=8000)