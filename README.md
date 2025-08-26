# cctv

# project/
│── app.py               # Flask 서버 실행 파일
│── detect_log.txt       # 로그 파일 (자동 생성됨)
│
├─ templates/
│   ├── map.html         # 지도 페이지 (위도/경도 표시)
│   └── log.html         # 로그 확인 페이지
│
├─ yolov8s.pt            # yolo 모델
├─ ex.csv                # 차량 번호판 데이터
│
├─ car_detected/         # YOLO 검출된 차량 이미지 저장 폴더
├─ person_detected/      # YOLO 검출된 사람 이미지 저장 폴더
├─ trash_detected/       # YOLO 검출된 의자/소파 저장 폴더
