from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import shutil
import threading
import json
import os
import time
from datetime import datetime

# Firebase imports
import firebase_admin
from firebase_admin import credentials, db

app = FastAPI()
model = YOLO("model.pt")

CLASS_LABELS = {0: "fire", 1: "smoke"}
Images = ["jpg", "jpeg", "png"]
Videos = ["mp4", "avi", "mov", "mkv"]

Detection_file = "fire_detections.json"
Sensor_file = "sensor_readings.json"
Output_Image = "last_detected_frame.jpg"

camera_url_store = {"url": None}
stop_flag = {"stop": False, "error": None}
shared_camera_frames = {"frame": None, "processed_frame": None, "detection_enabled": {"run": False}} # to enable detection
live_detections = []

TEMPERATURE_THRESHOLD = 20 # need change 
GAS_THRESHOLD = 60 # also


# for avoid conflict
stop_flag_lock = threading.Lock()
camera_running = False
camera_running_lock = threading.Lock()

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in Images)

def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in Videos)

def save_detection_to_json(detections):
    data = []
    if os.path.exists(Detection_file):
        with open(Detection_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
    data.extend(detections)
    with open(Detection_file, "w") as f:
        json.dump(data, f, indent=4)

def save_sensor_data(sensor_data: dict):
    data = []
    if os.path.exists(Sensor_file):
        with open(Sensor_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
    data.append(sensor_data)
    with open(Sensor_file, "w") as f:
        json.dump(data, f, indent=4)

def save_last_detected_frame(frame, results):
    if len(results) > 0 and len(results[0].boxes) > 0:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                conf = box.conf.tolist()[0]
                class_id = int(box.cls.tolist()[0])
                class_label = CLASS_LABELS.get(class_id, "unknown")
                color = (0, 0, 255) if class_label == "fire" else (255, 165, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(Output_Image, frame)
    else:
        cv2.imwrite(Output_Image, frame)

def extract_detections(results, source_name):
    detections = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for result in results:
        has_detection = False
        for box in result.boxes:
            has_detection = True
            class_id = int(box.cls.tolist()[0])
            class_label = CLASS_LABELS.get(class_id, "unknown")
            detections.append({
                "timestamp": timestamp,
                "source": source_name,
                "class": class_label
            })
        if not has_detection:
            detections.append(generate_no_detection_entry(source_name))
    return detections

def generate_no_detection_entry(source_name):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_name,
        "class": "no_detect"
    }

def start_camera_thread(camera_url):
    global camera_running

    with camera_running_lock:
        if camera_running:
            print("[INFO] Camera thread is already running. Skipping start.")
            return
        else:
            camera_running = True
            print("[INFO] Camera thread starting...")

    def on_camera_stop():
        global camera_running
        with camera_running_lock:
            camera_running = False
        print("[INFO] Camera thread stopped.")

    def create_camera_connection(url):
        if url == "0":
            cap = cv2.VideoCapture(0)
            source_description = "lab camera (index 0)"
        else:
            cap = cv2.VideoCapture(url)
            source_description = f"IP camera at {url}"
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, source_description

    def frame_reader(url, shared, stop_flag):
        cap, source_description = create_camera_connection(url)
        if not cap.isOpened():
            with stop_flag_lock:
                stop_flag["stop"] = True
                stop_flag["error"] = f"Could not connect to {source_description}"
            on_camera_stop()
            return
        while cap.isOpened():
            with stop_flag_lock:
                if stop_flag["stop"]:
                    break
            ret, frame = cap.read()
            if not ret:
                break
            shared['frame'] = frame
        cap.release()
        on_camera_stop()

# apply when threshold 
    def frame_detector(shared, source_name, stop_flag):
        detection_enabled = shared.get("detection_enabled", {"run": False})
        while True:
            with stop_flag_lock:
                if stop_flag["stop"]:
                    break
            frame = shared.get('frame')
            if frame is not None:
                processed_frame = frame.copy()
                if detection_enabled["run"]:
                    results = model(frame) # apply model  on shared
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                                conf = box.conf.tolist()[0]
                                class_id = int(box.cls.tolist()[0])
                                class_label = CLASS_LABELS.get(class_id, "unknown")
                                color = (0, 0, 255) if class_label == "fire" else (255, 165, 0)
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(processed_frame, f"{class_label} {conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        detections = extract_detections(results, source_name)
                        save_detection_to_json(detections)
                        save_last_detected_frame(frame, results)
                        live_detections.extend(detections)
                shared['processed_frame'] = processed_frame
                cv2.imshow("Live Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    with stop_flag_lock:
                        stop_flag["stop"] = True
                    break
        on_camera_stop()

    cv2.destroyAllWindows()

    global shared_camera_frames
    shared_camera_frames = {"frame": None, "processed_frame": None, "detection_enabled": {"run": False}}
    source_name = f"ip_camera_{camera_url.split('/')[-1]}"
    with stop_flag_lock:
        stop_flag.update({"stop": False, "error": None})

    threading.Thread(target=frame_reader, args=(camera_url, shared_camera_frames, stop_flag), daemon=True).start()
    threading.Thread(target=frame_detector, args=(shared_camera_frames, source_name, stop_flag), daemon=True).start()

class CameraUrlRequest(BaseModel):
    camera_url: str

@app.post("/set-camera-url/")
def set_camera_url(data: CameraUrlRequest):
    camera_url_store["url"] = data.camera_url
    threading.Thread(target=start_camera_thread, args=(data.camera_url,), daemon=True).start()
    return {"message": "Camera URL saved and camera stream started.", "camera_url": data.camera_url}

@app.get("/video-stream/")
def video_stream():
    if camera_url_store["url"] is None:
        raise HTTPException(status_code=400, detail="No camera URL set.")
    with stop_flag_lock:
        if stop_flag["stop"]:
            raise HTTPException(status_code=400, detail="Camera stream is stopped.")

    def generate_frames():
        while True:
            with stop_flag_lock:
                if stop_flag["stop"]:
                    break
            frame = shared_camera_frames.get('frame')
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.001)

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/stop-stream/")
def stop_stream():
    with stop_flag_lock:
        stop_flag["stop"] = True
    return {"message": "Camera stream stopped."}

@app.get("/last-frame/")
def last_frame():
    if not os.path.exists(Output_Image):
        raise HTTPException(status_code=404, detail="No last frame available.")
    return FileResponse(Output_Image, media_type="image/jpeg")

@app.get("/camera-detections/")
def get_camera_detections():
    return JSONResponse(content=live_detections)

def process_sensor_data(data):
    try:
        temperature = data.get("DHT11", {}).get("Temperature")
        humidity = data.get("DHT11", {}).get("Humidity")
        gas = data.get("GasSensor", {}).get("Percentage")
        flame_status = data.get("FlameSensor", {}).get("Status")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sensor_entry = {
            "timestamp": timestamp,
            "Temperature": temperature,
            "Humidity": humidity,
            "GasLevel": gas,
            "FlameStatus": flame_status
        }
        save_sensor_data(sensor_entry)

        if ((temperature is not None and temperature >= TEMPERATURE_THRESHOLD) or
            (gas is not None and gas >= GAS_THRESHOLD) or
            flame_status == 1):
            print("[INFO] Threshold exceeded. Enabling model detection...")
            shared_camera_frames.get("detection_enabled", {})["run"] = True

    except Exception as e:
        print("[ERROR] Failed to process sensor data:", e)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
        shutil.copyfileobj(file.file, tmp_file)

    try:
        detections = []
        frame_count = 0

        if is_image_file(file.filename):
            img = cv2.imread(tmp_file_path)
            results = model(img)
            detections = extract_detections(results, file.filename)
            save_last_detected_frame(img, results)
            save_detection_to_json(detections)

        elif is_video_file(file.filename):
            cap = cv2.VideoCapture(tmp_file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                results = model(frame)
                frame_detections = extract_detections(results, file.filename)
                save_detection_to_json(frame_detections)
                detections.extend(frame_detections)
                save_last_detected_frame(frame, results)
            cap.release()

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not detections:
            detections.append(generate_no_detection_entry(file.filename))
            save_detection_to_json(detections)

    finally:
        file.file.close()

    return JSONResponse(content=detections if detections else [generate_no_detection_entry(file.filename)])

def firebase_polling():
    while True:
        try:
            full_data = db.reference("/").get()
            if isinstance(full_data, dict):
                process_sensor_data(full_data)
        except Exception as e:
            print("[ERROR] Firebase polling failed:", e)
        time.sleep(5)

def firebase_listener():
    try:
        ref = db.reference("/")
        def sensor_data_listener(event):
            try:
                full_data = ref.get()
                if isinstance(full_data, dict):
                    process_sensor_data(full_data)
                else:
                    print("[INFO] Full data is not a dict:", full_data)
            except Exception as e:
                print("[ERROR] Failed to process sensor data from listener:", e)
        ref.listen(sensor_data_listener)
    except Exception as e:
        print("[WARNING] Listener failed, switching to polling:", e)
        firebase_polling()

if not firebase_admin._apps:
    cred = credentials.Certificate("firesdk.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": r"https://fire-5a45f-default-rtdb.firebaseio.com/"
    })

threading.Thread(target=firebase_listener, daemon=True).start()
#done
