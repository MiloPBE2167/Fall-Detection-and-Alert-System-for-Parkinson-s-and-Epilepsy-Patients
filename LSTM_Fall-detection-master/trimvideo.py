import cv2
import os

def trim_video(input_path, start_time, end_time, output_dir):
    videocurrent = 0
    
    # Duyệt qua từng video trong thư mục input_path
    for video in os.listdir(input_path):
        video_path = os.path.join(input_path, video)
        videocurrent += 1
        print(f"Processing video {videocurrent}: {video_path}")
        
        # Mở tệp video input
        cap = cv2.VideoCapture(video_path)
        
        # Lấy frame rate của video input
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Tính toán start và end frames
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Lấy tổng số frame của video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Đảm bảo end_frame không vượt quá tổng số frame
        end_frame = min(end_frame, total_frames)
        
        # Đặt video tới vị trí frame start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Lấy thông số video (width, height)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Tạo đường dẫn và tên tệp cho video đã cắt
        output_filename = f"trimmed_{videocurrent}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Tạo đối tượng VideoWriter để lưu video đã cắt
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
        # Đọc và lưu các frame từ start đến end_frame
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1
        
        # Giải phóng các tài nguyên
        cap.release()
        out.release()
        print(f"Trimmed video saved to {output_path}")

# Đường dẫn tới thư mục chứa video gốc
input_path = 'video_train/github/Subject 4/Fall'

# Đường dẫn tới thư mục lưu video đã cắt
output_dir = 'trimmed_videos/subject4'

# Tạo thư mục lưu video nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Khoảng thời gian cần cắt (tính bằng giây)
start_time = 0.2  # Start time in seconds
end_time = 3.5    # End time in seconds

# Gọi hàm trim_video để cắt và lưu video
trim_video(input_path, start_time, end_time, output_dir)