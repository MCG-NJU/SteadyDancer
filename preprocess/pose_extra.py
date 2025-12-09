import numpy as np
import argparse
import torch
import copy
import cv2
import os
import moviepy.video.io.ImageSequenceClip

from pose.script.dwpose import DWposeDetector, draw_pose
from pose.script.util import size_calculate, warpAffine_kps
from utils_aug import pose_aug_diff


def run_align_video_with_filterPose_translate_smooth(args):

    vidfn=args.vidfn
    # imgfn_refer=args.imgfn_refer
    outfn_all=args.outfn_all
    
    video = cv2.VideoCapture(vidfn)
    width= video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height= video.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    total_frame= video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps= video.get(cv2.CAP_PROP_FPS)

    print("height:", height)
    print("width:", width)
    print("fps:", fps)

    H_in, W_in  = height, width
    H_out, W_out = size_calculate(H_in,W_in,args.detect_resolution) 
    H_out, W_out = size_calculate(H_out,W_out,args.image_resolution) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DWposeDetector(
        det_config = args.yolox_config, 
        det_ckpt = args.yolox_ckpt,
        pose_config = args.dwpose_config, 
        pose_ckpt = args.dwpose_ckpt, 
        keypoints_only=False
        )    
    detector = detector.to(device)

    skip_frames = args.align_frame
    max_frame = args.max_frame
    pose_list, video_frame_buffer, video_pose_buffer = [], [], []

    for i in range(max_frame):
        ret, img = video.read()
        if img is None: 
            break 
        else: 
            if i < skip_frames:
                continue           
            video_frame_buffer.append(img)
    
        # pose align
        pose_img, pose_ori = detector(img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
        video_pose_buffer.append(pose_img)

    H = 768 # paint height
    # H = ref_H # paint height
    # W1 = int((H/ref_H * ref_W)//2 *2)
    W2 = int((H/height * width)//2 *2)
    result_demo = [] # = Writer(args, None, H, 3*W1+2*W2, outfn_all, fps)
    result_pose_only = [] # Writer(args, None, H, W1, args.outfn_single, fps)
    for i in range(len(video_frame_buffer)):
        
        video_frame = cv2.resize(video_frame_buffer[i], (W2, H), interpolation=cv2.INTER_CUBIC)
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_pose  = cv2.resize(video_pose_buffer[i], (W2, H), interpolation=cv2.INTER_CUBIC)

        res_all = np.concatenate([video_frame, video_pose], axis=1) # all.mp4
        result_demo.append(res_all) # all.mp4
        res_single = np.concatenate([video_pose], axis=1) # single.mp4
        result_pose_only.append(res_single) # single.mp4

    print(f"pose_list len: {len(pose_list)}")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_demo, fps=fps)
    clip.write_videofile(outfn_all, fps=fps, codec="libx264") # all.mp4
    
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_pose_only, fps=fps)
    clip.write_videofile(args.outfn_single, fps=fps, codec="libx264") # single.mp4
    
    print('pose align done')



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--detect_resolution', type=int, default=512, help='detect_resolution')
    # parser.add_argument('--image_resolution', type=int, default=720, help='image_resolution')
    parser.add_argument('--detect_resolution', type=int, default=1024, help='detect_resolution')
    parser.add_argument('--image_resolution', type=int, default=720, help='image_resolution')

    parser.add_argument("--yolox_config",  type=str, default=f"{os.path.dirname(__file__)}/pose/config/yolox_l_8xb8-300e_coco.py")
    parser.add_argument("--dwpose_config", type=str, default=f"{os.path.dirname(__file__)}/pose/config/dwpose-l_384x288.py")
    parser.add_argument("--yolox_ckpt",  type=str, default=f"{os.path.dirname(__file__)}/pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth")
    parser.add_argument("--dwpose_ckpt", type=str, default=f"{os.path.dirname(__file__)}/pretrained_weights/dwpose/dw-ll_ucoco_384.pth")


    parser.add_argument('--align_frame', type=int, default=0, help='the frame index of the video to align')
    parser.add_argument('--max_frame', type=int, default=300, help='maximum frame number of the video to align')
    parser.add_argument('--vidfn', type=str, default="./assets/videos/0.mp4", help='Input video path')
    parser.add_argument('--outfn_all', type=str, default=None, help='Output path of the alignment visualization')
    parser.add_argument('--outfn_single', type=str, default=None, help='output path of the aligned video of the refer img')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfn_all), exist_ok=True)
    os.makedirs(os.path.dirname(args.outfn_single), exist_ok=True)

    run_align_video_with_filterPose_translate_smooth(args)


    
if __name__ == '__main__':
    main()
