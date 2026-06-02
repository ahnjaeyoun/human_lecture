# TCP/IP 소켓 통신을 위한 라이브러리
import socket

# 데이터 길이를 byte 형태로 변환하기 위한 라이브러리
import struct

# JSON 데이터 생성
import json

# 배열 연산
import numpy as np

# TensorFlow 사용
import tensorflow as tf

# 이미지 처리
from PIL import Image

# 메모리상의 이미지 처리
from io import BytesIO


# =====================================================
# CIFAR10 클래스 정의
# =====================================================

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
    "truck"
]


# =====================================================
# 학습된 모델 로드
# =====================================================

print("AI 모델 로딩 시작")

# 학습 시 저장했던 모델 읽기
model = tf.keras.models.load_model(
    "./model/cifar10_model.h5"
)

print("AI 모델 로딩 완료")


# =====================================================
# 지정 크기만큼 데이터 수신 함수
# =====================================================

def recv_all(sock, size):

    """
    TCP는 데이터가 나눠서 올 수 있음

    예)

    10000 byte 전송

    ↓

    3000 byte
    4000 byte
    3000 byte

    로 나눠질 수 있음

    따라서 원하는 크기만큼
    반복 수신 필요
    """

    data = b''

    while len(data) < size:

        packet = sock.recv(
            size - len(data)
        )

        if not packet:

            return None

        data += packet

    return data


# =====================================================
# 이미지 추론 함수
# =====================================================

def predict_image(image_bytes):

    """
    입력

    이미지 Byte

    출력

    분류 결과
    """

    # -----------------------------------
    # Byte → PIL Image
    # -----------------------------------

    image = Image.open(
        BytesIO(image_bytes)
    )

    # RGB 변환
    image = image.convert("RGB")

    # -----------------------------------
    # CIFAR10 크기 맞춤
    # -----------------------------------

    image = image.resize(
        (32, 32)
    )

    # -----------------------------------
    # numpy 배열 변환
    # -----------------------------------

    image = np.array(image)

    # -----------------------------------
    # 정규화
    # -----------------------------------

    image = image / 255.0

    # -----------------------------------
    # 배치 차원 추가
    # -----------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )

    # shape

    print(
        "입력 Shape :",
        image.shape
    )

    # -----------------------------------
    # 모델 추론
    # -----------------------------------

    pred = model.predict(
        image,
        verbose=0
    )

    # 예측 결과 출력

    print("예측 확률")

    print(pred)

    # -----------------------------------
    # 가장 높은 확률 찾기
    # -----------------------------------

    class_idx = np.argmax(pred)

    confidence = np.max(pred)

    print("예측 클래스 :", class_idx)

    print("신뢰도 :", confidence)

    # -----------------------------------
    # 결과 생성
    # -----------------------------------

    result = {

        "class_id": int(class_idx),

        "class_name":
            CLASS_NAMES[class_idx],

        "confidence":
            float(confidence)
    }

    return result

# =====================================================
# 서버 설정
# =====================================================
# 서버 ip/port 설정
HOST = "192.168.0.47"
PORT = 8954

# =====================================================
# TCP 소켓 설정
# =====================================================
server_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

print("소켓 생성 완료")

# =====================================================
# 포트 바인딩
# =====================================================
server_socket.bind((HOST, PORT))
print("포트 바인딩 완료")

# =====================================================
# 클라이언트 접속 대기
# =====================================================
server_socket.listen(5)
print(f"AI 서버 시작: {HOST}:{PORT}")

# =====================================================
# 데이터 송수신 무한 루프
# =====================================================

while True:
    print("\n클라이언트 접속 대기 중")

    # 클라이언트 접속
    client_socket, addr = server_socket.accept()

    print("클라이언트 접속 :", addr)

    try:

        # =================================
        # 이미지 크기 수신
        # =================================

        """
        Client가 먼저

        이미지 크기

        를 보냄

        예)

        125000 byte
        """

        header = recv_all(

            client_socket,

            4

        )

        if header is None:
            continue

        # 4byte → 정수 변환

        image_size = struct.unpack(

            ">I",

            header

        )[0]

        print(
            "이미지 크기 :",
            image_size
        )

        # =================================
        # 이미지 수신
        # =================================

        image_bytes = recv_all(

            client_socket,

            image_size

        )

        print(
            "이미지 수신 완료"
        )

        # =================================
        # AI 추론
        # =================================

        result = predict_image(

            image_bytes

        )

        print(
            "추론 결과 :",
            result
        )

        # =================================
        # JSON 변환
        # =================================

        result_json = json.dumps(

            result,

            ensure_ascii=False

        ).encode()

        # =================================
        # 결과 길이 전송
        # =================================

        client_socket.sendall(

            struct.pack(

                ">I",

                len(result_json)

            )

        )

        # =================================
        # 결과 데이터 전송
        # =================================

        client_socket.sendall(

            result_json

        )

        print(
            "결과 전송 완료"
        )

    except Exception as e:

        print(
            "오류 발생 :",
            e
        )

    finally:
        # 연결 종료
        client_socket.close()
        print("클라이언트 연결 종료")












