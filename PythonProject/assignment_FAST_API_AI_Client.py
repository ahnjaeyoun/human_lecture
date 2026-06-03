import os
import requests

# =====================================================
# 서버 정보 및 설정
# =====================================================
SERVER_URL = "http://192.168.219.102:8000/predict"

print("CIFAR10 AI 이미지 분류 클라이언트 시작 (종료하려면 'exit' 입력)\n")

# =====================================================
# 사용자 입력 및 전송 반복 루프
# =====================================================
while True:
    # 1. 전송할 이미지 파일 경로 입력 받기 (예: ./dog.jpg)
    image_path = input("전송할 이미지 파일 경로 입력: ").strip()
    # strip()은 파이썬 내장함수로써 양쪽 끝의 공백이나 줄바꿈 문자를 잘라낸다

    # exit 입력 시 종료
    if image_path.lower() == "exit":
        print("클라이언트를 종료합니다.")
        break

    # 2. 파일 존재 여부 확인
    if not os.path.exists(image_path):
        print(" 파일이 존재하지 않습니다. 경로를 다시 확인해주세요.")
        continue

    print(f"이미지 읽기 완료 (크기: {os.path.getsize(image_path)} byte)")
    print("서버에 분석 요청 중...")

    # 3. 파일 딕셔너리 생성 및 POST 요청 전송
    try:
        # 'rb'(바이너리 읽기) 모드로 파일을 열어 전송 구조 생성
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}

            # FastAPI 서버로 multipart/form-data 전송
            response = requests.post(SERVER_URL, files=files)

    except requests.exceptions.RequestException as e:
        print(f" 서버 연결 오류가 발생했습니다: {e}\n")
        continue

    # 4. 서버 응답 처리 및 출력
    if response.status_code == 200:
        response_json = response.json()

        if response_json.get("success"):
            result = response_json["result"]

            print("\n" + "=" * 30)
            print(" AI 이미지 분류 결과")
            print("=" * 30)
            print(f" Class ID    : {result['class_id']}")
            print(f" Class Name  : {result['class_name'].upper()}")
            print(f" Confidence  : {result['confidence']:.4f} ({round(result['confidence'] * 100, 2)}%)")
            print("=" * 30 + "\n")
        else:
            print(f" 서버 내부 추론 실패: {response_json.get('error')}\n")
    else:
        print(f" HTTP 오류 발생: {response.status_code}, {response.text}\n")