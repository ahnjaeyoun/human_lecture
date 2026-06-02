import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras import models

# ============================================================
# CIFAR 10 데이터셋 로드
# ============================================================
'''
32x32 크기의 컬러 이미지이며,
총 10개 클래스로 구성된 데이터셋입니다.
airplane, autombile, bird, cat, deer, dog, frog, horse, ship, truck
'''


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
'''
x_train : 학습 이미지
y_train : 학습 정답(label)
x_test  : 테스트 이미지
y_test : 테스트 정답(label)

'''
# ============================================================
# 데이터 확인
# ============================================================
# 학습 데이터 개수 출력
print("학습 데이터 개수 :", len(x_train))

# 테스트 데이터 개수 출력
print("테스트 데이터 개수 : ", len(x_test))

# 이미지 크기 출력
print("이미지 shape : ", x_train.shape)

# label shape 출력
print("label shape : ", y_train.shape)

# ============================================================
# 정규화
# ============================================================
'''
이미지 픽셀값은 0~255 범위의 데이터를 가지고 있음
딥러닝 학습 안정성을 위해 0~1 범위로 스케일 정규화를 진행
'''
x_train = x_train / 255.0
x_test = x_test / 255.0

# ============================================================
# CNN 모델 생성
# ===========================================================
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3) ,activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3) ,activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3) ,activation='relu'),
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),

    layers.Dense(units=10, activation='softmax')
])

# ============================================================
# CNN 모델 구조 출력
# ============================================================
model.summary()

# ============================================================
# 모델 컴파일 설정
# ============================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 학습 시작
# ============================================================
print("\n학습 시작\n")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# ============================================================
# 모델 테스트 성능 평가
# ============================================================
print("\n테스트 평가 시작\n")

# 테스트 데이터로 평가
test_loss, test_acc = model.evaluate(x_test, y_test)

# 손실값 출력
print("Test loss : ", test_loss)
# 정확도 출력
print("Test accuracy : ", test_acc)

# ============================================================
# 모델 학습 결과 저장
# ============================================================
model.save(
    "./model/cifar10_model.h5",
)

# ============================================================
# 클래스 정보 출력
# ============================================================

CLASS_NAMES = [
    "airplane",
    "autombile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
print("\n클래스 목록\n")

for idx, name in enumerate(CLASS_NAMES):
    print(idx, " : ", name)






















