from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO("best.pt")

# 카메라 열기
cap = cv2.VideoCapture(0)

# 프레임 읽기
ret, frame = cap.read()
if ret:
    # 예측 수행
    results = model.predict(source=frame, conf=0.2)

    # 결과 이미지에 박스 그리기
    annotated_frame = results[0].plot()

    # 결과 저장
    cv2.imwrite("dog_detect_result.jpg", annotated_frame)
    print("✅ 결과 이미지 dog_detect_result.jpg로 저장 완료!")
else:
    print("❌ 카메라에서 프레임을 읽지 못했습니다.")

cap.release()
