import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
import tempfile
import time


# Các biến đường dẫn
MODEL_PATH = "../yolo-face/yolov11l-face.pt"
FACE_DATABASE = "main/face_images"
CSV_PATH = "main/attendance.csv"



# Configure the web page
st.set_page_config(page_title="Face Attendance System", layout="wide", initial_sidebar_state="expanded")

# Thêm CSS để làm đẹp giao diện
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .status-success {
        padding: 0.5rem;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .status-warning {
        padding: 0.5rem;
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        margin: 1rem 0;
    }
    .status-error {
        padding: 0.5rem;
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📷 Student Face Attendance System</h1>", unsafe_allow_html=True)

# Khởi tạo session state để lưu trữ thông tin phiên 
if 'recognized_students' not in st.session_state:
    st.session_state.recognized_students = []
if 'attendance_count' not in st.session_state:
    st.session_state.attendance_count = 0
if 'last_recognition_time' not in st.session_state:
    st.session_state.last_recognition_time = None
# Khởi tạo dictionary để lưu trữ số phiếu bầu cho mỗi khuôn mặt
if 'face_votes' not in st.session_state:
    st.session_state.face_votes = {}



# Kiểm tra thư mục database khuôn mặt
if not os.path.exists(FACE_DATABASE):
    os.makedirs(FACE_DATABASE)
    st.warning(f"Face database directory created at {FACE_DATABASE}. Please add student face images.")

# Hiển thị trạng thái loading khi tải mô hình YOLO
with st.spinner("Loading YOLO face detection model..."):
    try:
        yolo_model = YOLO(MODEL_PATH)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Kiểm tra và tạo CSV file nếu chưa tồn tại
if not os.path.exists(CSV_PATH):
    st.info("Creating new attendance file...")
    df = pd.DataFrame(columns=["StudentID"])
    df.to_csv(CSV_PATH, index=False)
    st.success("Attendance file created successfully!")
else:
    # Đọc danh sách sinh viên và khởi tạo điểm danh cho ngày hiện tại
    df = pd.read_csv(CSV_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Thêm cột ngày mới nếu chưa tồn tại
    if today not in df.columns:
        # Đặt tất cả sinh viên là 'Absent' (vắng mặt) ban đầu
        df[today] = 'Absent'
        df.to_csv(CSV_PATH, index=False)
        st.sidebar.success(f"Initialized attendance for today: {today}")
    
    # Hiển thị thông tin về file điểm danh hiện tại
    st.sidebar.info(f"Current attendance file has {len(df)} students and {len(df.columns)-1} attendance dates.")

# Tạo danh sách sinh viên có trong file để kiểm tra
VALID_STUDENT_IDS = []
if os.path.exists(CSV_PATH):
    student_df = pd.read_csv(CSV_PATH)
    VALID_STUDENT_IDS = student_df['StudentID'].unique().tolist()


# Function to detect faces with YOLO
def detect_faces(frame):
    results = yolo_model(frame, verbose=False)
    faces, bboxes = [], []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append(face)
                bboxes.append((x1, y1, x2, y2))
    return faces, bboxes

# Function to recognize face with DeepFace
def recognize_face(face):
    try:
        distance_threshold=0.3
        result = DeepFace.find(
            img_path=face, 
            db_path=FACE_DATABASE, 
            model_name="Facenet512",
            enforce_detection=False, 
            align=True, 
            distance_metric="cosine", 
            silent=True,
            normalization="Facenet2018",
            detector_backend="skip",
            threshold=distance_threshold,
        )
        # if len(result) > 0:  # Kiểm tra danh sách không rỗng
        #     df = result[0]  # Lấy DataFrame đầu tiên trong danh sách
        #     if "identity" in df.columns:  # Kiểm tra cột "identity" có tồn tại không
        #         identities = df["identity"].tolist()  # Lấy danh sách giá trị của cột "identity"
        #         print(identities)
        #     else:
        print(result)
        print("_____________________________")

        if result and len(result) > 0 and not result[0]['distance'].empty:
            # Lấy kết quả có khoảng cách gần nhất
            min_distance_idx = result[0]['distance'].idxmin()
            identity_path = result[0]['identity'][min_distance_idx]
            distance = result[0]['distance'][min_distance_idx]
            
            # Chỉ chấp nhận khuôn mặt có khoảng cách dưới ngưỡng
            if distance < recognition_threshold:  # Sử dụng ngưỡng từ settings
                # Tách mã sinh viên từ tên thư mục
                folder_name = identity_path.split(os.sep)[-2]  # Lấy tên thư mục
                student_id = int(folder_name.split("_")[0])  # Lấy phần trước dấu gạch dưới
                
                # Kiểm tra xem ID sinh viên có hợp lệ không (có trong danh sách)
                
                if int(student_id) in VALID_STUDENT_IDS:
                    st.sidebar.info(f"Face recognized: {student_id}, Distance: {distance:.4f}")
                    return student_id, distance
                else:
                    st.sidebar.warning(f"ID {student_id} found in face database but not in attendance list!")
                    return "Unknown", distance
            else:
                return "Unknown", distance
        return "Unknown", 1.0
    except Exception as e:
        st.error(f"Error during face recognition: {str(e)}")
        return "Error", 1.0

# Function to mark attendance
def mark_attendance(student_id):
    if student_id == "Unknown" or student_id == "Error":
        return False
    
    try:
        df = pd.read_csv(CSV_PATH, encoding='ISO-8859-1')
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Thêm cột ngày mới nếu chưa tồn tại
        if today not in df.columns:
            # Đặt tất cả sinh viên là 'Absent' (vắng mặt) ban đầu
            df[today] = 'Absent'
            df.to_csv(CSV_PATH, index=False)
            st.sidebar.success(f"Khởi tạo cột điểm danh mới cho ngày: {today}")
        
        # Kiểm tra sinh viên có trong danh sách
        if student_id in df['StudentID'].values:
            # Đánh dấu sinh viên có mặt
            if df.loc[df['StudentID'] == student_id, today].values[0] != 'Present':
                df.loc[df['StudentID'] == student_id, today] = 'Present'
                # Lưu lại file CSV
                df.to_csv(CSV_PATH, index=False)
                st.sidebar.success(f"Marked student {student_id} as present")
                return True
            return False  # Không điểm danh lại nếu đã có mặt
        else:
            # Nếu sinh viên không có trong danh sách, hiển thị cảnh báo
            st.sidebar.warning(f"Student {student_id} not in attendance list. Cannot mark attendance.")
            return False
            
    except Exception as e:
        st.error(f"Error marking attendance: {str(e)}")
        return False

# Thêm thanh sidebar cho các tuỳ chọn
st.sidebar.markdown("## Attendance Options")
option = st.sidebar.radio("Select attendance method", ["Camera", "Upload Video"])

# Tuỳ chọn cài đặt hiển thị trong thanh sidebar
with st.sidebar.expander("Advanced Settings"):
    recognition_threshold = st.slider("Recognition Threshold", 0.1, 0.5, 0.3, 0.01, 
                                   help="Lower values are more strict, higher values are more permissive")
    camera_duration = st.slider("Camera Capture Duration (seconds)", 1, 10, 3, 
                             help="How long to capture from camera for attendance")
    # Thêm slider cho số phiếu cần thiết
    votes_required = st.slider("Recognition Votes Required", 1, 10, 3,
                            help="Number of times a face must be recognized before marking attendance")
    show_confidence = st.checkbox("Show Confidence Score", value=True)

# Hiển thị kết quả điểm danh trong sidebar
if st.session_state.recognized_students:
    st.sidebar.markdown("### Recent Attendance")
    for student in st.session_state.recognized_students[-5:]:  # Show last 5 students
        st.sidebar.markdown(f"- {student['id']} at {student['time']}")

# Camera Option
if option == "Camera":
    st.markdown("<h2 class='sub-header'>Use Camera for Attendance</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tạo placeholder cho camera feed
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.markdown("### Instructions")
        st.markdown("""
        1. Click 'Start Attendance' to begin
        2. Position your face in the camera
        3. Hold still for a moment
        4. System will automatically mark attendance
        """)
        
        # Button to start attendance
        if st.button("Start Attendance", key="camera_start"):
            # Reset bộ đếm phiếu bầu khi bắt đầu phiên mới
            st.session_state.face_votes = {}
            
            # Đảm bảo ngày hôm nay đã được khởi tạo trong CSV
            today = datetime.now().strftime("%Y-%m-%d")
            df = pd.read_csv(CSV_PATH)
            if today not in df.columns:
                df[today] = 'Absent'  # Mặc định tất cả sinh viên vắng mặt
                df.to_csv(CSV_PATH, index=False)
                st.success(f"Initialized attendance for today: {today}")
            
            # Cập nhật lại danh sách ID sinh viên hợp lệ
            VALID_STUDENT_IDS = df['StudentID'].unique().tolist()
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open camera. Please check your camera connection.")
            else:
                with st.spinner("Initializing camera..."):
                    ret, frame = cap.read()  # Warm up camera
                
                # Set up progress bar
                progress_bar = st.progress(0)
                status_placeholder.markdown("<div class='status-warning'>Capturing...</div>", unsafe_allow_html=True)
                
                # Capture for specified duration
                start_time = time.time()
                recognized_faces = set()  # Để tránh điểm danh trùng lặp
                
                while time.time() - start_time < camera_duration:
                    # Update progress bar
                    elapsed = time.time() - start_time
                    progress = min(1.0, elapsed / camera_duration)
                    progress_bar.progress(progress)
                    
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.markdown("<div class='status-error'>Error reading from camera</div>", unsafe_allow_html=True)
                        break
                    
                    # Detect and recognize faces
                    faces, bboxes = detect_faces(frame)
                    
                    for face, (x1, y1, x2, y2) in zip(faces, bboxes):
                        student_id, confidence = recognize_face(face)
                        confidence_text = f" ({1-confidence:.2f})" if show_confidence else ""
                        
                        # Vẽ khung và thông tin
                        if student_id != "Unknown" and student_id != "Error":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Cập nhật số phiếu cho ID này
                            if student_id in st.session_state.face_votes:
                                st.session_state.face_votes[student_id] += 1
                            else:
                                st.session_state.face_votes[student_id] = 1
                            
                            # Lấy số phiếu hiện tại
                            current_votes = st.session_state.face_votes[student_id]
                            
                            # Hiển thị thông tin khuôn mặt cùng với số phiếu
                            cv2.putText(frame, f"ID: {student_id}{confidence_text} Votes: {current_votes}/{votes_required}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Chỉ đánh dấu điểm danh khi đủ số phiếu cần thiết
                            if current_votes >= votes_required and student_id not in recognized_faces:
                                if mark_attendance(student_id):
                                    recognized_faces.add(student_id)
                                    st.session_state.attendance_count += 1
                                    current_time = datetime.now().strftime("%H:%M:%S")
                                    st.session_state.last_recognition_time = current_time
                                    st.session_state.recognized_students.append({
                                        'id': student_id,
                                        'time': current_time,
                                        'confidence': 1-confidence,
                                        'votes': current_votes
                                    })
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Unknown{confidence_text}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Hiển thị frame
                    camera_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Hoàn thành quá trình
                progress_bar.progress(1.0)
                cap.release()
                
                if len(recognized_faces) > 0:
                    status_placeholder.markdown("<div class='status-success'>Attendance completed successfully!</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.markdown("<div class='status-warning'>No students recognized. Try again.</div>", unsafe_allow_html=True)
                
                st.write(f"Recognized {len(recognized_faces)} students in this session.")

# Upload Video Option
elif option == "Upload Video":
    st.markdown("<h2 class='sub-header'>Upload Video for Attendance</h2>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
        
        # Video processing controls
        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown("### Video Controls")
            process_video = st.button("Process Video")
            stop_processing = st.button("Stop Processing")
        
        with col1:
            # Create placeholders
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
        
        if process_video:
            # Reset bộ đếm phiếu bầu khi bắt đầu phiên mới
            st.session_state.face_votes = {}
            
            # Cập nhật lại danh sách ID sinh viên hợp lệ
            if os.path.exists(CSV_PATH):
                student_df = pd.read_csv(CSV_PATH)
                VALID_STUDENT_IDS = student_df['StudentID'].unique().tolist()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
            else:
                # Get video details
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                
                status_placeholder.markdown("<div class='status-warning'>Processing video...</div>", unsafe_allow_html=True)
                progress_bar = progress_placeholder.progress(0)
                
                frame_count = 0
                processing = True
                recognized_faces = set()  # Để tránh điểm danh trùng lặp
                
                # Process every 5th frame to speed up
                frame_step = 5
                
                while processing and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    # Only process every frame_step frames
                    if frame_count % frame_step != 0:
                        continue
                    
                    # Check if stop button pressed
                    if stop_processing:
                        processing = False
                        status_placeholder.markdown("<div class='status-warning'>Processing stopped by user</div>", unsafe_allow_html=True)
                        break
                    
                    # Update progress
                    if total_frames > 0:
                        progress = min(1.0, frame_count / total_frames)
                        progress_bar.progress(progress)
                    
                    # Process frame
                    faces, bboxes = detect_faces(frame)
                    for face, (x1, y1, x2, y2) in zip(faces, bboxes):
                        student_id, confidence = recognize_face(face)
                        confidence_text = f" ({1-confidence:.2f})" if show_confidence else ""
                        
                        if student_id != "Unknown" and student_id != "Error":
                            color = (0, 255, 0)  # Green for recognized
                            
                            # Cập nhật số phiếu cho ID này
                            if student_id in st.session_state.face_votes:
                                st.session_state.face_votes[student_id] += 1
                            else:
                                st.session_state.face_votes[student_id] = 1
                            
                            # Lấy số phiếu hiện tại
                            current_votes = st.session_state.face_votes[student_id]
                            
                            # Chỉ đánh dấu điểm danh khi đủ số phiếu cần thiết
                            if current_votes >= votes_required and student_id not in recognized_faces:
                                if mark_attendance(student_id):
                                    recognized_faces.add(student_id)
                                    current_time = datetime.now().strftime("%H:%M:%S")
                                    st.session_state.recognized_students.append({
                                        'id': student_id,
                                        'time': current_time,
                                        'confidence': 1-confidence,
                                        'votes': current_votes
                                    })
                        else:
                            color = (0, 0, 255)  # Red for unknown
                        
                        # Draw box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{student_id}{confidence_text}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Display the frame
                    video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Clean up
                cap.release()
                progress_bar.progress(1.0)
                
                if len(recognized_faces) > 0:
                    status_placeholder.markdown("<div class='status-success'>Video processing completed!</div>", unsafe_allow_html=True)
                    st.write(f"Recognized {len(recognized_faces)} students from video.")
                else:
                    status_placeholder.markdown("<div class='status-warning'>No students recognized in the video.</div>", unsafe_allow_html=True)
                
                # Clean up temp file
                try:
                    os.unlink(video_path)
                except:
                    pass

# Attendance Results Section
st.markdown("<h2 class='sub-header'>Attendance Results</h2>", unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Today's Attendance", "Full Attendance Record", "Manual Attendance"])

with tab1:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today in df.columns:
            # Filter to show only present students today
            present_today = df[df[today] == 'Present']
            absent_today = df[df[today] == 'Absent']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Present Students", len(present_today))
            with col2:
                st.metric("Absent Students", len(absent_today))
            
            if not present_today.empty:
                st.write("### Students Present Today")
                st.dataframe(present_today[['StudentID', today]], width=400)
            else:
                st.info("No students marked present today.")
        else:
            st.info(f"No attendance record for today ({today}) yet.")

with tab2:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        
        # Add options to filter data
        st.write("### Full Attendance Record")
        
        # Date filter if there are dates in the dataset
        date_columns = [col for col in df.columns if col != 'StudentID']
        if date_columns:
            selected_dates = st.multiselect("Select dates to view:", date_columns, default=date_columns[-1:])
            
            if selected_dates:
                filtered_df = df[['StudentID'] + selected_dates]
                st.dataframe(filtered_df)
                
                # Download button for CSV
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Attendance Data",
                    data=csv,
                    file_name="attendance_export.csv",
                    mime="text/csv"
                )
            else:
                st.info("Please select at least one date to view.")
        else:
            st.info("No attendance dates found in the record.")
    else:
        st.error("Attendance file not found.")

with tab3:
    st.write("### Manual Attendance")
    st.write("Use this section to manually edit student attendance status for today.")
    
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Đảm bảo cột ngày hôm nay tồn tại
        if today not in df.columns:
            df[today] = 'Absent'
            df.to_csv(CSV_PATH, index=False)
            st.success(f"Initialized attendance column for today: {today}")
        
        # Hiển thị thống kê
        present_count = len(df[df[today] == 'Present'])
        absent_count = len(df[df[today] == 'Absent'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Present", present_count, f"{present_count/len(df)*100:.1f}%")
        with col3:
            st.metric("Absent", absent_count, f"{absent_count/len(df)*100:.1f}%")
        
        # Thêm tính năng tìm kiếm sinh viên
        search_query = st.text_input("🔍 Search student by ID:", "")
        
        # Tạo 2 tab cho 2 cách chỉnh sửa khác nhau
        edit_tab1, edit_tab2 = st.tabs(["Individual Editing", "Batch Selection"])
        
        with edit_tab1:
            st.write("Edit attendance status for individual students:")
            
            # Lọc sinh viên dựa trên từ khóa tìm kiếm
            filtered_df = df
            if search_query:
                filtered_df = df[df['StudentID'].str.contains(search_query, case=False)]
            
            if filtered_df.empty:
                st.warning("No students match your search criteria.")
            else:
                # Hiển thị từng sinh viên với nút chuyển đổi trạng thái
                for index, row in filtered_df.iterrows():
                    student_id = row['StudentID']
                    current_status = row[today]
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**{student_id}**")
                    
                    with col2:
                        status_color = "green" if current_status == "Present" else "red"
                        st.markdown(f"<span style='color:{status_color};font-weight:bold'>{current_status}</span>", unsafe_allow_html=True)
                    
                    with col3:
                        # Nút để chuyển đổi trạng thái
                        new_status = "Absent" if current_status == "Present" else "Present"
                        button_label = f"Mark as {new_status}" 
                        
                        if st.button(button_label, key=f"toggle_{student_id}"):
                            df.loc[index, today] = new_status
                            df.to_csv(CSV_PATH, index=False)
                            st.success(f"Updated {student_id} to {new_status}")
                            st.rerun()
                    
                    st.markdown("---")
        
        with edit_tab2:
            st.write("Select multiple students to mark as present:")
            
            # Tạo danh sách sinh viên để lựa chọn
            student_ids = df['StudentID'].tolist()
            
            # Mặc định chọn những sinh viên đã có mặt
            selected_students = st.multiselect(
                "Select students to mark as present:",
                student_ids,
                default=[sid for sid in student_ids if df.loc[df['StudentID'] == sid, today].values[0] == 'Present']
            )
            
            if st.button("Update Attendance", key="batch_update"):
                # Đánh dấu tất cả là vắng mặt trước
                df[today] = 'Absent'
                
                # Đánh dấu những sinh viên được chọn là có mặt
                for student_id in selected_students:
                    df.loc[df['StudentID'] == student_id, today] = 'Present'
                
                # Lưu lại file CSV
                df.to_csv(CSV_PATH, index=False)
                st.success(f"Updated attendance for all students. {len(selected_students)} marked as present!")
                
                # Cập nhật lại hiển thị
                st.rerun()
    else:
        st.error("Attendance file not found.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for student attendance management")
