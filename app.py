import cv2
import numpy as np
import os
import re
import time
from datetime import datetime
from pathlib import Path
import easyocr
from ultralytics import YOLO
from flask import Flask, render_template, Response, send_from_directory, request, jsonify

########################################
# Flask 앱 설정
########################################
app = Flask(__name__)

# 저장 폴더
output_dirs = {
    "car": Path("car_detected"),
    "furniture": Path("trash_detected"),
    "fallen": Path("fallen_detected")
}
for d in output_dirs.values():
    d.mkdir(exist_ok=True)

# 로그 파일
LOG_FILE = "detect_log.txt"

# 최신 GPS (오드로이드에서 전송받음)
latest_gps = {"lat": None, "lon": None}

########################################
# 차량 번호판 등록 리스트 로드
########################################
def load_known_plates(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='cp949') as f:
        return set(line.strip() for line in f if line.strip())

known_plates_from_file = load_known_plates('ex.csv')

########################################
# 모델 및 OCR 로드
########################################
model = YOLO('yolov8s.pt')
names = model.names
ocr_reader = easyocr.Reader(['ko', 'en'])

########################################
# 상태 저장 변수
########################################
trackers = {}
vehicle_still_frames = {}
already_saved = {}

fps = 30
STILL_FRAME_THRESHOLD = int(fps * 2)  # 2초 이상 정지 차량

########################################
# 비디오 소스
########################################
video_source = 0  # 웹캠 (또는 RTSP 주소)
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise Exception("❌ 영상 소스를 열 수 없습니다.")

########################################
# 로그 기록
########################################
def write_log(obj_type, gps):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {obj_type} detected at {gps}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())

########################################
# 유틸 - 번호판 인식
########################################
def recognize_plate(plate_img, frame, offset=(0, 0)):
    results = ocr_reader.readtext(plate_img)
    for bbox, text, conf in results:
        text = text.strip().replace(" ", "")
        if conf > 0.3 and re.match(r'^\d{2,3}[가-힣]\d{3,4}$', text):
            (tl, tr, br, bl) = bbox
            pts = np.array([
                [int(tl[0] + offset[0]), int(tl[1] + offset[1])],
                [int(tr[0] + offset[0]), int(tr[1] + offset[1])],
                [int(br[0] + offset[0]), int(br[1] + offset[1])],
                [int(bl[0] + offset[0]), int(bl[1] + offset[1])]
            ], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            return text
    return None

########################################
# 유틸 - 쓰러짐 판단
########################################
def is_fallen_bbox(xyxy, ratio_thresh=1.2):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        return False
    return (w / h) > ratio_thresh

########################################
# 프레임 생성 (세 가지 감지 동시 수행)
########################################
def generate_frames():
    saved_furniture = False
    saved_fallen = False
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ===== 1. 차량 번호판 감지 =====
            if class_name == "car":
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                key = f"v_{cls_id}_{x1}_{y1}"

                last_pos = trackers.get(key)
                if last_pos and np.linalg.norm(np.array(center) - np.array(last_pos)) < 5:
                    vehicle_still_frames[key] = vehicle_still_frames.get(key, 0) + 1
                else:
                    vehicle_still_frames[key] = 0

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                if vehicle_still_frames[key] > STILL_FRAME_THRESHOLD and not already_saved.get(key, False):
                    vehicle_crop = frame[y1:y2, x1:x2]
                    h_vehicle = vehicle_crop.shape[0]
                    start_y = int(h_vehicle * 0.4)
                    end_y = int(h_vehicle * 0.8)
                    cropped_area = vehicle_crop[start_y:end_y, :]
                    crop_offset = (x1, y1 + start_y)

                    plate_number = recognize_plate(cropped_area, frame, offset=crop_offset)
                    if plate_number:
                        cv2.putText(frame, plate_number, (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if plate_number not in known_plates_from_file:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            alert_frame = frame.copy()
                            cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                            filename = f"{plate_number}_{timestamp}.jpg"
                            save_path = output_dirs["car"] / filename
                            cv2.imwrite(str(save_path), alert_frame)
                            print(f"[ALERT] 미등록 차량: {plate_number} 저장됨: {save_path}")
                            write_log("car", latest_gps)

                    already_saved[key] = True
                trackers[key] = center

            # ===== 2. 가구(소파, 의자) 감지 =====
            elif class_name in ["chair", "couch"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if not saved_furniture:
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = output_dirs["furniture"] / f"{class_name}_{now_str}.jpg"
                    cv2.imwrite(str(file_path), frame)
                    print(f"[✔] 가구 감지 → 저장: {file_path}")
                    write_log("furniture", latest_gps)
                    saved_furniture = True

            # ===== 3. 쓰러진 사람 감지 =====
            elif class_name == "person":
                fallen = is_fallen_bbox((x1, y1, x2, y2))
                color = (0, 0, 255) if fallen else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = "FALLEN" if fallen else "Normal"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if fallen and not saved_fallen:
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = output_dirs["fallen"] / f"{now_str}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    print(f"[💾] 쓰러진 사람 → 저장: {out_path}")
                    write_log("fallen", latest_gps)
                    saved_fallen = True

        # ===== 스트리밍 전송 =====
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

########################################
# Flask 라우트
########################################
@app.route('/')
def index():
    files = {name: sorted(os.listdir(path), reverse=True)
             for name, path in output_dirs.items()}
    return render_template('index.html', files=files)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/downloads/<category>/<path:filename>')
def download_file(category, filename):
    if category in output_dirs:
        return send_from_directory(output_dirs[category], filename, as_attachment=True)
    return "잘못된 카테고리", 404

@app.route("/logs")
def logs_page():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
    else:
        log_content = "아직 검출 로그가 없습니다."
    return f"<h2>검출 로그</h2><pre>{log_content}</pre>"

@app.route("/update_gps", methods=["POST"])
def update_gps():
    data = request.json
    latest_gps["lat"] = data.get("lat")
    latest_gps["lon"] = data.get("lon")
    return jsonify({"status": "GPS updated", "gps": latest_gps})

@app.route("/get_gps")
def get_gps():
    return jsonify(latest_gps)

########################################
# 실행
########################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

