import cv2
import os
from glob import glob

def images_to_video(image_folder, output_path, fps=10):
    # 이미지 파일 리스트 정렬
    image_files = sorted(glob(os.path.join(image_folder, '*.jpg')))
    if not image_files:
        print("No jpg files found in the folder.")
        return

    # 첫 이미지로 프레임 크기 결정
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # 비디오 라이터 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_file in image_files:
        img = cv2.imread(img_file)
        video.write(img)

    video.release()
    print(f"Video saved to {output_path}")

# 사용 예시
images_to_video('results/trt/fp16', 'output_video_fp16.mp4', fps=30)