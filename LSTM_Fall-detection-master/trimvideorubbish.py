import cv2
import os

def trim_video(input_path, start_time, end_time, output_dir,videocurrent):

    
    # Duyệt qua từng video trong thư mục input_path
   
      
       
      
        
        # Mở tệp video input
    cap = cv2.VideoCapture(input_path)
    
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







start_time = 1.5
end_time = 4.5



input_path = 'video_train/github/Subject 4/Fall/17.mp4'
output_dir = 'trimmed_videos/subject4'
os.makedirs(output_dir, exist_ok=True)
videocurrent = 17



trim_video(input_path, start_time, end_time, output_dir,videocurrent)