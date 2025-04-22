import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from tqdm import tqdm
import numpy as np
from deepface import DeepFace
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')  # Tắt các cảnh báo không cần thiết

# Tắt các thông báo debug của DeepFace
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt các log của TensorFlow

# Kiểm tra GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO-face model và đẩy lên GPU
face_model = YOLO("yolo-face/yolov11l-face.pt")
face_model.to(device)

# Khởi tạo DeepSort
tracker = DeepSort()

# Dictionary để lưu lịch sử dự đoán cho mỗi ID
prediction_history = defaultdict(list)
# Dictionary để lưu thời gian cho mỗi prediction
prediction_times = defaultdict(list)

# Load database khuôn mặt
db_path = "main/face_images"
print("Loading face database...")
known_face_dirs = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d))]
print(f"Found {len(known_face_dirs)} people in database")

# Mở video
cap = cv2.VideoCapture("main/videos/close.mp4")

# Thông tin video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Biến đếm frame để tính thời gian
frame_count = 0

# Ghi video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("main/videos/out_face_recognition.mp4", fourcc, fps, (width, height))

# Màu duy nhất cho bbox (Xanh lá)
bbox_color = (0, 255, 0)

# Tạo progress bar và tắt các thông báo khác
for _ in tqdm(range(total_frames), desc="Processing video", ncols=100):
    ret, frame = cap.read()
    if not ret:
        break

    # Tính thời gian hiện tại trong video
    current_time = frame_count / fps
    frame_count += 1

    # Chạy YOLO-face để phát hiện khuôn mặt
    face_results = face_model(frame, device=device, verbose=False)[0]
    detections = []

    # Chuyển đổi kết quả phát hiện khuôn mặt
    for result in face_results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        w = x2 - x1
        h = y2 - y1
        detections.append([[x1, y1, w, h], conf, 'face'])

    # Cập nhật tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Xử lý mỗi khuôn mặt được track
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        face_box = list(map(int, track.to_ltrb()))

        try:
            track_id_int = int(track_id)
            face_img = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
            if face_img.size == 0:
                continue

            # Nhận diện khuôn mặt bằng DeepFace
            result = DeepFace.find(img_path=face_img, db_path=db_path, model_name="Facenet512",
                                   threshold=0.4, enforce_detection=False, align=True,distance_metric="cosine",
                                   normalization = "Facenet2018", silent= True, detector_backend="skip",
                                   )
            
            if len(result[0]) > 0:
                best_match = result[0].iloc[0]
                predicted_path = best_match['identity']
                # Lấy tên sau dấu "_"
                folder_name = os.path.basename(os.path.dirname(predicted_path))
                predicted_name = folder_name.split('_')[1] if '_' in folder_name else folder_name
                prediction_history[track_id].append(predicted_name)
                prediction_times[track_id].append(current_time)
                
                predictions = prediction_history[track_id]
                if len(predictions) > 0:
                    from collections import Counter
                    count = Counter(predictions)
                    most_common = count.most_common(1)[0]
                    accuracy = most_common[1] / len(predictions)
                    predicted_label = f"{most_common[0]} ({accuracy:.2%})"
                else:
                    predicted_label = "Unknown"
            else:
                predicted_label = "Unknown"
                
        except Exception:
            predicted_label = "Error"

        # Vẽ bounding box và thông tin với màu duy nhất
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), bbox_color, 2)
        cv2.putText(frame, f'ID: {track_id} - {predicted_label}', 
                    (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

    out.write(frame)

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Tính accuracy tại một mốc thời gian cụ thể
def calculate_accuracy_at_time(predictions, times, target_time):
    relevant_preds = [pred for pred, time in zip(predictions, times) if time <= target_time]
    if not relevant_preds:
        return None, 0
    from collections import Counter
    count = Counter(relevant_preds)
    most_common = count.most_common(1)[0]
    accuracy = most_common[1] / len(relevant_preds)
    return most_common[0], accuracy

# In kết quả accuracy tại các mốc thời gian
time_points = [3, 5, 10]  # Cập nhật các mốc thời gian
print("\nKết quả accuracy theo thời gian:")
print("-" * 50)

for track_id in prediction_history.keys():
    predictions = prediction_history[track_id]
    times = prediction_times[track_id]
    
    print(f"\nĐối tượng ID {track_id}:")
    for t in time_points:
        pred_name, acc = calculate_accuracy_at_time(predictions, times, t)
        if pred_name:
            print(f"- Sau {t}s: {pred_name} (Accuracy: {acc:.3%})")
        else:
            print(f"- Sau {t}s: Chưa có dự đoán")