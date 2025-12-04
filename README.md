<p align="center">

  <h2 align="center">SteadyDancer: Harmonized and Coherent Human Image Animation with First-Frame Preservation</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=0lLB3fsAAAAJ"><strong>Jiaming Zhang</strong></a>
    ·
    <a href="https://dblp.org/pid/316/8117.html"><strong>Shengming Cao</strong></a>
    ·
    <a href="https://qianduoduolr.github.io/"><strong>Rui Li</strong></a>
    ·
    <a href="https://openreview.net/profile?id=~Xiaotong_Zhao1"><strong>Xiaotong Zhao</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TSMchWcAAAAJ&hl=en&oi=ao"><strong>Yutao Cui</strong></a>
    <br>
    <a href=""><strong>Xinglin Hou</strong></a>
    ·
    <a href="https://mcg.nju.edu.cn/member/gswu/en/index.html"><strong>Gangshan Wu</strong></a>
    ·
    <a href="https://openreview.net/profile?id=~Haolan_Chen1"><strong>Haolan Chen</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=FHvejDIAAAAJ"><strong>Yu Xu</strong></a> 
    ·
    <a href="https://scholar.google.com/citations?user=TSMchWcAAAAJ&hl=en&oi=ao"><strong>Limin Wang</strong></a>
    ·
    <a href="https://openreview.net/profile?id=~Kai_Ma4"><strong>Kai Ma</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2511.19320"><img src='https://img.shields.io/badge/arXiv-2511.19320-red' alt='Paper PDF'></a>
        <a href='https://mcg-nju.github.io/steadydancer-web'><img src='https://img.shields.io/badge/Project-Page-blue' alt='Project Page'></a>
        <a href='https://huggingface.co/MCG-NJU/SteadyDancer-14B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
        <a href='https://huggingface.co/datasets/MCG-NJU/X-Dance'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-X--Dance-green'></a>
    <br>
    <b></a>Multimedia Computing Group, Nanjing University &nbsp; | &nbsp; </a>Platform and Content Group (PCG), Tencent  </b>
    <br>
  </p>
</p>

This repository is the official implementation of paper "SteadyDancer: Harmonized and Coherent Human Image Animation with First-Frame Preservation". SteadyDancer is a strong animation framework based on **Image-to-Video paradigm**, ensuring **robust first-frame preservation**. In contrast to prior *Reference-to-Video* approaches that often suffer from identity drift due to **spatio-temporal misalignments** common in real-world applications, SteadyDancer generates **high-fidelity and temporally coherent** human animations, outperforming existing methods in visual quality and control while **requiring significantly fewer training resources**.

![teaser](assets/teaser.png?raw=true)

## 📣 Updates

- **2025-12-04**: 🔥 Released our weight in [GGUF format](https://huggingface.co/MCG-NJU/SteadyDancer-GGUF), which coverted from [Kijai weight](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/SteadyDancer) by [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/tree/main/tools). It offers lower cost in [Kijai ComfyUI](https://github.com/kijai/ComfyUI-WanVideoWrapper).
- **2025-11-27**: 🔥 Supported Multi-GPU inference with FSDP + xDiT USP in the inference code.
- **2025-11-24**: 🔥 Released the X-Dance Benchmark on [huggingface](https://huggingface.co/datasets/MCG-NJU/X-Dance).
- **2025-11-24**: 🔥 Released the inference code and [weights](https://huggingface.co/MCG-NJU/SteadyDancer-14B) of SteadyDancer.
- **2025-11-24**: 🔥 Our paper is in public on [arxiv](https://arxiv.org/abs/2511.19320).

## 🏘️ Community Works

We warmly welcome community contributions to SteadyDancer! If your work has any relation or help to SteadyDancer and you would like more people to see it, please inform us.

- **2025-12-03**: 🔥 SteadyDancer is now supported in [WanGP](https://github.com/deepbeepmeep/Wan2GP). [deepbeepmeep](https://github.com/deepbeepmeep) said WanGP supports full preprocessing pipeline with augmented poses, or use Loras accelerators for a quick generation. Thanks for their contributions!
- **2025-11-30**: 🔥 SteadyDancer now supports ComfyUI in [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). **Thanks to [kijai](https://github.com/kijai) for the tremendous contribution 🙏🙏 !!** Please note that our pose detector, alignment, and augmentor are still missing in the current version, which will impact performance, but you can already enjoy it with vitpose/dwpose and lightx2v. **Stay tuned for the full version later!!**

## 🎯 Motivation

![motivation](assets/motivation.png?raw=true)

- **Spatio-temporal Misalignments**: We identify and tackle the prevalent issues of **spatial-structural inconsistencies** and **temporal start-gaps** between source images and driving videos common in real-world scenarios, which often lead to identity drift in generated animations.
- **Image-to-Video (I2V) v.s. Reference-to-Video (R2V) paradigm**: The R2V paradigm treats animation as **binding a reference image to a driven pose**. However, this **relaxation of alignment constraints** fails under spatio-temporal misalignments, causing artifacts and abrupt transitions in spatial inconsistencies or temporal start-gap scenarios. Conversely, the I2V paradigm is superior as it inherently guarantees **first-frame preservation**, , and its **Motion-to-Image Alignment** ensures high-fidelity and coherent video generation starting directly from the reference state.


## 🖼️ Gallery

- Results on **X-Dance Benchmark**, which focus on 1) the spatio-temporal misalignments by **different-source image-video pairs**; and 2) visual identity preservation, temporal coherence, and motion accuracy by **complex motion and appearance variations**.

<table class="center">
    <tr>
    <td><img src="assets/X-1.gif"></td>
    <td><img src="assets/X-3.gif"></td>
    </tr>
    <tr>
    <td><img src="assets/X-2.gif"></td>
    <td><img src="assets/X-4.gif"></td>
    </tr>
    <tr>
    <td><img src="assets/X-5.gif"></td>
    <td><img src="assets/X-6.gif"></td>
    </tr>
</table>

- Results on **RealisDance-Val Benchmark**, which focus on 1) **real-world dance videos** with same-source image-video pairs; and 2) synthesize **realistic object dynamics** that are physically consistent with the driving actions.

<table class="center">
    <tr>
    <td><img src="assets/R-1.gif"></td>
    <td><img src="assets/R-2.gif"></td>
    </tr>
    <tr>
    <td><img src="assets/R-3.gif"></td>
    <td><img src="assets/R-4.gif"></td>
    </tr>
    <tr>
    <td><img src="assets/R-5.gif"></td>
    <td><img src="assets/R-6.gif"></td>
    </tr>
</table>

## 🛠️ Installation
```
# Clone this repository
git clone https://github.com/MCG-NJU/SteadyDancer.git
cd SteadyDancer

# Create and activate conda environment
conda create -n steadydancer python=3.10 -y
conda activate steadydancer

# Install animate generation dependencies (Pytorch 2.5.1, CUDA 12.1 for example)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && python -c "import flash_attn"
pip install xformers==0.0.29.post1
pip install "xfuser[diffusers,flash-attn]"
pip install -r requirements.txt

# Install pose extraction dependencies
pip install moviepy decord              # moviepy-2.2.1, decord-0.6.0
pip install --no-cache-dir -U openmim   # openmim-0.3.9
mim install mmengine                    # mmengine-0.10.7
mim install "mmcv==2.1.0"               # mmcv-2.1.0
mim install "mmdet>=3.1.0"              # mmdet-3.3.0
pip install mmpose                      # mmpose-1.3.2
```

- Errors consistently occur during the installation of the mmcv and mmpose packages, so please verify that both packages were installed successfully:
```
python -c "import mmcv"
python -c "import mmpose"
python -c "from mmpose.apis import inference_topdown"
python -c "from mmpose.apis import init_model as init_pose_estimator"
python -c "from mmpose.evaluation.functional import nms"
python -c "from mmpose.utils import adapt_mmdet_pipeline"
python -c "from mmpose.structures import merge_data_samples"
```

- If you encounter "*ModuleNotFoundError: No module named 'mmcv._ext'*" issue during installation, please re-install mmcv manually (We haven't found a more convenient and stable method. If you have a better method, please submit a pull request to help us. We would greatly appreciate it 😊.):
```
mim uninstall mmcv -y
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout v2.1.0
pip install -r requirements/optional.txt
gcc --version                               # Check the gcc version (requires 5.4+)
python setup.py build_ext                   # Build the C++ and CUDA extensions, may take a while
python setup.py develop
pip install -e . -v                         # Install mmcv in editable mode
python .dev_scripts/check_installation.py   # just verify the installation was successful by running this script, ignore the last verify script
cd ../
```

## 📥 Download Checkpoints
```
# Download DW-Pose pretrained weights
mkdir -p ./preprocess/pretrained_weights/dwpose
huggingface-cli download yzd-v/DWPose --local-dir ./preprocess/pretrained_weights/dwpose --include "dw-ll_ucoco_384.pth"
wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth -O ./preprocess/pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth

# Download SteadyDancer-14B model weights
huggingface-cli download MCG-NJU/SteadyDancer-14B --local-dir ./SteadyDancer-14B
```

## 🚀 Inference

To generate dance video from a source image and a driving video (We have provided pose example in `preprocess/output/video00001_img00001/example` and `preprocess/output/video00002_img00002/example` to try our model quickly), please follow the steps below:
- Pose extraction and alignment:
```
ref_image_path="data/images/00001.png"
driving_video_path="data/videos/00001"
pair_id="video00001_img00001"
output=./preprocess/output/${pair_id}/$(date +"%Y%m%d%H%M%S")

## Extract and align pose (Positive Condition)
outfn=$output/positive/all.mp4
outfn_align_pose_video=$output/positive/single.mp4
python preprocess/pose_align.py \
    --imgfn_refer "$ref_image_path" \
    --vidfn "${driving_video_path}/video.mp4" \
    --outfn "$outfn" \
    --outfn_align_pose_video "$outfn_align_pose_video"

outfn_align_pose_video=$output/positive/single.mp4
python preprocess/dump_video_images.py "$outfn_align_pose_video" "$(dirname "$outfn_align_pose_video")"


## Extract and align pose (Negative Condition)
outfn=$output/negative/all.mp4
outfn_align_pose_video=$output/negative/single.mp4
python preprocess/pose_align_withdiffaug.py \
    --imgfn_refer "$ref_image_path" \
    --vidfn "${driving_video_path}/video.mp4" \
    --outfn "$outfn" \
    --outfn_align_pose_video "$outfn_align_pose_video"

outfn_align_pose_video=$output/negative/single_aug.mp4
python preprocess/dump_video_images.py "$outfn_align_pose_video" "$(dirname "$outfn_align_pose_video")"


## copy other files
cp "$ref_image_path" "$output/ref_image.png"
cp "${driving_video_path}/video.mp4" "$output/driving_video.mp4"
cp "${driving_video_path}/prompt.txt" "$output/prompt.txt"


## (Optional) Visualization of original pose without alignment
driving_video_path="data/videos/00001"
python preprocess/pose_extra.py \
    --vidfn $driving_video_path/video.mp4 \
    --outfn_all $driving_video_path/pose_ori_all.mp4 \
    --outfn_single $driving_video_path/pose_ori_single.mp4
```

- Generate animation video with SteadyDancer:
```
ckpt_dir="./SteadyDancer-14B"

input_dir="preprocess/output/video00001_img00001/example"   # </path/to/preprocess/output/> contains ref_image.png, driving_video.mp4, prompt.txt, positive/, negative/ folders, e.g. the above ./preprocess/output/${pair_id}/$(date +"%Y%m%d%H%M%S")
image="$input_dir/ref_image.png"          # reference image path
cond_pos_folder="$input_dir/positive/"    # positive condition pose folder
cond_neg_folder="$input_dir/negative/"    # negative condition pose folder
prompt=$(cat $input_dir/prompt.txt)       # read prompt from file
save_file="$(basename "$(dirname "$input_dir")")--Pair$(basename "$input_dir").mp4"  # save file name

cfg_scale=5.0
condition_guide_scale=1.0
pro=0.4
base_seed=106060

# Single-GPU inference
CUDA_VISIBLE_DEVICES=0 python generate_dancer.py \
    --task i2v-14B --size 1024*576 \
    --ckpt_dir $ckpt_dir \
    --prompt "$prompt" \
    --image $image \
    --cond_pos_folder $cond_pos_folder \
    --cond_neg_folder $cond_neg_folder \
    --sample_guide_scale $cfg_scale \
    --condition_guide_scale $condition_guide_scale \
    --end_cond_cfg $pro \
    --base_seed $base_seed \
    --save_file "${save_file}--$(date +"%Y%m%d%H%M%S")"

# Multi-GPU inference using FSDP + xDiT USP
GPUs=2
torchrun --nproc_per_node=${GPUs} generate_dancer.py \
    --dit_fsdp --t5_fsdp --ulysses_size ${GPUs} \
    --task i2v-14B --size 1024*576 \
    --ckpt_dir $ckpt_dir \
    --prompt "$prompt" \
    --image $image \
    --cond_pos_folder $cond_pos_folder \
    --cond_neg_folder $cond_neg_folder \
    --sample_guide_scale $cfg_scale \
    --condition_guide_scale $condition_guide_scale \
    --end_cond_cfg $pro \
    --base_seed $base_seed \
    --save_file "${save_file}--$(date +"%Y%m%d%H%M%S")--xDiTUSP${GPUs}"
```
NOTE: Multi-GPU inference may be faster and use less memory than Single-GPU inference, but [it may be different with Single-GPU results](https://github.com/Wan-Video/Wan2.1/issues/304) due to the non-deterministic nature of distributed computing, **so we recommend using Single-GPU inference for better reproducibility**.

## 🎥 X-Dance Benchmark
To fill the void left by existing same-source benchmarks (such as TikTok), which fail to evaluate spatio-temporal misalignments, we propose **X-Dance**, a new benchmark that focuses on these challenges. The X-Dance benchmark is constructed from diverse image categories (male/female/cartoon, and upper-/full-body shots) and challenging driving videos (complex motions with blur and occlusion). Its curated set of pairings intentionally introduces spatial-structural inconsistencies and temporal start-gaps, allowing for a more robust evaluation of model generalization in the real world.
You can download the X-Dance benchmark from [huggingface](https://huggingface.co/datasets/MCG-NJU/X-Dance).

![X-Dance](assets/X-Dance.png?raw=true)

## ❤️ Acknowledgements
Our implementation is based on [Wan 2.1](https://github.com/Wan-Video/Wan2.1). We modify [MusePose](https://github.com/TMElyralab/MusePose/tree/main) to generate and align pose video. Thanks for their remarkable contribution and released code! Thanks to everyone in the community who has contributed to SteadyDancer.

## 📚 Citation

If you find our paper or this codebase useful for your research, please cite us.
```BibTeX
@misc{zhang2025steadydancer,
      title={SteadyDancer: Harmonized and Coherent Human Image Animation with First-Frame Preservation}, 
      author={Jiaming Zhang and Shengming Cao and Rui Li and Xiaotong Zhao and Yutao Cui and Xinglin Hou and Gangshan Wu and Haolan Chen and Yu Xu and Limin Wang and Kai Ma},
      year={2025},
      eprint={2511.19320},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19320}, 
}
```

## 📄 License
This repository is released under the Apache-2.0 license as found in the [LICENSE](LICENSE) file.
