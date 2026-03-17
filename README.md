# 🚗 LiDAR-Camera Late Fusion for Depth Estimation on KITTI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/notebooks/SensorFusion_KITTI.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A clean implementation of late sensor fusion** combining sparse LiDAR points with dense monocular depth (MiDaS) to produce accurate, metric depth maps on the KITTI dataset. This project demonstrates how fusion overcomes the limitations of each individual sensor: LiDAR provides precise but sparse measurements, while monocular depth is dense but scale‑ambiguous. By aligning and averaging both sources, we obtain a depth map that is both **dense** and **metrically accurate** – essential for autonomous driving, robotics, and AR/VR applications.

---

## ✨ Features

- **LiDAR projection** onto the image plane using KITTI calibration matrices.
- **Sparse-to-dense interpolation** of LiDAR depth via nearest neighbors.
- **Monocular depth estimation** with a state‑of‑the‑art transformer (MiDaS DPT‑Large).
- **Scale alignment** by matching the median of monocular depth to LiDAR depth.
- **Late fusion** by pixel‑wise averaging of scaled monocular and dense LiDAR depths.
- **Comprehensive visualization** of each processing step for easy benchmarking.
- **Automated output generation**: six‑panel benchmark images, animated GIF, and MP4 video.

---

## 📂 Dataset

We use the **KITTI Object Detection** training set (first 100 samples). The data is organized as follows:

data_object_image_2/training/ # .png images
data_object_velodyne/training/ # .bin point clouds
data_object_calib/training/ # .txt calibration files


> **Note:** The dataset is **not** included in this repository due to size constraints. Follow the instructions below to download and place it correctly.

---

## 🧠 Methodology (Late Fusion)

1. **Project LiDAR points** to image coordinates using calibration.
2. **Create sparse depth map** only at projected pixel locations.
3. **Interpolate** to obtain a dense LiDAR depth map.
4. **Estimate monocular depth** with MiDaS (raw, scale‑ambiguous).
5. **Scale** the monocular depth so its median matches the LiDAR median (metric scale).
6. **Fuse** by averaging the scaled monocular and dense LiDAR depth maps.

All metric depth maps are displayed with the same color scale (0–50 m) for direct comparison.

---

## 📊 Results

### Six‑Panel Benchmark
For each frame, the pipeline generates a figure showing:
- (a) Camera image (reference)
- (b) Sparse LiDAR depth (accurate but sparse)
- (c) Dense LiDAR depth (interpolated, fills gaps but may blur edges)
- (d) Raw monocular depth (dense but scale‑ambiguous)
- (e) Scaled monocular depth (now metric, preserves edges)
- (f) **Fused depth** ✅ – the best of both worlds: accurate + dense

![Benchmark sample](assets/frame_000000_benchmark.png)  
*Example benchmark image for frame 000000. (Place your actual image in `assets/`)*

### Animated GIF (Camera + Fused Depth)
The side‑by‑side GIF shows the camera image (left) and the fused depth map (right) over 10 consecutive frames, illustrating temporal consistency.

![Fusion demo GIF](assets/fusion_demo.gif)  
*GIF of fused depth over 10 frames. (Place your GIF in `assets/`)*

### Video (MP4)
A higher‑quality MP4 version is also available for embedding or presentations.

[![Fusion demo video](assets/video_thumbnail.png)](assets/fusion_demo.mp4)  
*Click the thumbnail to download the MP4 video. (Place your video in `assets/`)*

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/VasuTammisetti/KITTI_LiDAR_Camera_Fusion_For-Better_Deapth-map.git
cd KITTI_LiDAR_Camera_Fusion_For-Better_Deapth-map
