import os
import sys
import cv2
from decord import VideoReader
from decord import cpu

def save_frames(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 初始化 VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))
    
    # 获取视频的总帧数
    total_frames = len(vr)
    
    # 遍历每一帧并保存
    for i in range(total_frames):
        # 读取第 i 帧
        frame = vr[i].asnumpy()
        
        # 保存帧为图片
        frame_path = os.path.join(output_folder, f"{i:04d}.jpg")
        # must CV2 需要的事 BGR 格式的数组
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # 显式转换
        cv2.imwrite(frame_path, frame)
        
        print(f"Saved frame {i} to {frame_path}")

# 示例用法
video_path = sys.argv[1]
output_folder = sys.argv[2]
save_frames(video_path, output_folder)
