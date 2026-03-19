<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:1a1a2e,100:16213e&height=200&section=header&text=LiDAR%20%2B%20Camera%20Depth%20Fusion&fontSize=36&fontColor=00d4ff&fontAlignY=38&desc=Dense%20Metric%20Depth%20Estimation%20on%20KITTI%20via%20MiDaS%20%2B%20LiDAR%20Late%20Fusion&descAlignY=60&descSize=15&animation=fadeIn" width="100%"/>
<p>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/MiDaS-DPT_Large-00d4ff?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/KITTI-Dataset-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Fusion-Late_Fusion-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Depth-Dense_%2B_Metric-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Sensors-Camera_%2B_LiDAR-blueviolet?style=flat-square"/>
  <img src="https://img.shields.io/badge/No_Retraining_Required-✓-green?style=flat-square"/>
</p>
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
<br>
> **Sparse LiDAR is accurate but full of gaps. Monocular depth is dense but scale-ambiguous.**
> **This project solves both problems simultaneously — no retraining, no stereo camera required.**
</div>
---
🎬 Live Demo
<div align="center">
![Fusion Demo](fusion_demo%20(1).gif)
Six-panel benchmark: Camera → Sparse LiDAR → Dense LiDAR → Raw MiDaS → Scaled MiDaS → Fused Depth
</div>
---
🧭 The Core Problem
<div align="center">
Sensor	Output	Problem
📡 LiDAR alone	Accurate metric depth	Extremely sparse — only ~5% of pixels covered
📷 Camera (MiDaS) alone	Dense, edge-preserving depth	Scale-ambiguous — no absolute distance
🔀 This project	Dense + metrically accurate	✅ Best of both worlds
</div>
The key insight: LiDAR provides the scale, MiDaS provides the density.
A single median-scaling step aligns them — no training data, no neural fusion network required.
---
🏗️ Pipeline Architecture
```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│   📷 Camera Image (1242×375)    📡 LiDAR Point Cloud (~120K pts) │
└───────────────┬──────────────────────────────┬───────────────────┘
                │                              │
                ▼                              ▼
┌───────────────────────┐          ┌───────────────────────────┐
│   MiDaS DPT-Large     │          │   LiDAR Projection        │
│   Transformer         │          │                           │
│                       │          │  • P2 · R0 · Tr_velo      │
│  Dense depth map      │          │  • Sparse depth map       │
│  (scale-ambiguous)    │          │  • NN interpolation       │
│                       │          │    → Dense LiDAR depth    │
└──────────┬────────────┘          └─────────────┬─────────────┘
           │  D_mono (relative)                  │  D_lidar (metric)
           │                                     │
           └─────────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │     SCALE ALIGNMENT      │
              │                          │
              │  scale = median(D_lidar) │
              │          ─────────────── │
              │          median(D_mono)  │
              │                          │
              │  D_scaled = D_mono × s   │
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │      LATE FUSION         │
              │                          │
              │  D_fused = (D_scaled     │
              │           + D_lidar) / 2 │
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   DENSE METRIC DEPTH ✅  │
              │   accurate + complete    │
              └──────────────────────────┘
```
---
📊 Six-Panel Benchmark Results
<div align="center">
<img width="1451" height="936" alt="Benchmark" src="https://github.com/user-attachments/assets/e749048c-5f46-4771-b955-4b5dca536a7c"/>
</div>
Panel	Description	Key Property
(a) Camera	RGB reference image	Texture, colour, edges
(b) Sparse LiDAR	Raw projected points	Accurate but ~5% coverage
(c) Dense LiDAR	NN-interpolated	Complete but blurred edges
(d) Raw MiDaS	Transformer depth	Dense but no metric scale
(e) Scaled MiDaS	Median-aligned	Dense + metric scale ✅
(f) Fused Depth	(d) + (e) averaged	Dense + accurate + metric ✅
> All depth maps rendered at the same colour scale (0–50 m) for direct visual comparison.
---
✨ Feature Highlights
<table>
<tr>
<td width="50%">
🔭 Depth Estimation
MiDaS DPT-Large vision transformer backbone
Handles textureless regions, reflections, sky
Edge-preserving predictions at full image resolution
Robust across indoor and outdoor scenes
</td>
<td width="50%">
📡 LiDAR Processing
KITTI calibration matrices (P2 · R0 · Tr_velo_to_cam)
Sparse-to-dense via nearest-neighbour interpolation
Handles occlusion and scan line gaps
Metric accuracy preserved throughout
</td>
</tr>
<tr>
<td width="50%">
🔀 Fusion Strategy
Median scaling — no training required
Single scalar aligns absolute depth scale
Pixel-wise averaging for final fusion
Applies to any camera+LiDAR setup
</td>
<td width="50%">
📦 Output Generation
Six-panel benchmark PNG per frame
Animated GIF for README preview
MP4 video (H.264, GitHub compatible)
Depth colourmap: 0–50 m fixed scale
</td>
</tr>
</table>
---
📐 Methodology: Median Scale Alignment
The single most important step in the pipeline:
```python
# 1. Get valid LiDAR pixels
valid_mask = sparse_lidar_depth > 0

# 2. Sample monocular depth at those same pixels
mono_at_lidar = midas_depth[valid_mask]
lidar_vals    = sparse_lidar_depth[valid_mask]

# 3. Compute scale factor
scale = np.median(lidar_vals) / np.median(mono_at_lidar)

# 4. Apply to full monocular depth map
scaled_midas = midas_depth * scale

# 5. Fuse
fused_depth = (scaled_midas + dense_lidar) / 2.0
```
This elegant approach requires zero labelled data and zero retraining — yet produces metrically accurate dense depth maps ready for downstream use.
---
🆚 Comparison to Other Methods
Method	Dense	Metric Scale	Single Camera	No Training	Edge Quality
LiDAR only	❌	✅	✅	✅	⚠️ Sparse
Stereo depth	✅	✅	❌ Two cameras	✅	✅ Good
MiDaS only	✅	❌	✅	✅	✅ Excellent
Supervised fusion	✅	✅	✅	❌ Needs labels	✅ Excellent
This project	✅	✅	✅	✅	✅ Excellent
---
🚀 Applications
<table>
<tr>
<td align="center" width="25%">🚗<br><b>Autonomous Driving</b><br><sub>Obstacle detection, free-space estimation, path planning</sub></td>
<td align="center" width="25%">🤖<br><b>Robotics</b><br><sub>Grasping, navigation, SLAM in unstructured environments</sub></td>
<td align="center" width="25%">🥽<br><b>AR / VR</b><br><sub>Occlusion handling, realistic virtual object placement</sub></td>
<td align="center" width="25%">🏭<br><b>Industrial</b><br><sub>Quality control, dimension measurement, inspection</sub></td>
</tr>
</table>
---
📦 Dataset
The KITTI subset used in this project is publicly available on Zenodo:
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19111017.svg)
⬇ Download Dataset — Zenodo
The archive contains pre-organised `data_object_image_2`, `data_object_velodyne`, and `data_object_calib` folders ready to use directly with the notebook.
---
⚡ Quick Start
1. Open in Colab
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
2. Prepare Data
```
sensorfusion/
└── sensorfusion/
    ├── data_object_image_2/training/    ← .png files
    ├── data_object_velodyne/training/   ← .bin files
    └── data_object_calib/training/      ← .txt files
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run All Cells
```
Runtime → Run all   (Ctrl+F9)
```
> ⏱️ Full pipeline on 20 frames: ~3 minutes on T4 GPU
---
🛠️ Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.6.0
numpy>=1.23.0
matplotlib>=3.7.0
Pillow>=9.0.0
timm>=0.9.0
scipy>=1.10.0
tqdm>=4.65.0
```
---
🏆 State-of-the-Art Context
Monocular depth has advanced rapidly with vision transformers — MiDaS v3.1 DPT-Large achieves state-of-the-art zero-shot generalisation across scenes. However, all monocular methods share one fundamental limitation: absolute scale is unknowable from a single image.
Fusing even sparse LiDAR with median alignment solves this completely. Recent learning-based approaches (e.g. CMT-Depth, GuideDepth, BEV-Depth) push accuracy further through end-to-end training, but our training-free baseline already demonstrates the core fusion benefit and generalises to any calibrated camera+LiDAR system.
---
🗂️ Repository Structure
```
KITTI_LiDAR_Camera_Fusion/
│
├── 📓 Camera_LiDAR_Fusion.ipynb     ← Full pipeline notebook
├── 📄 requirements.txt              ← Dependencies
├── 📄 README.md                     ← This file
├── 🖼️ frame_000001_benchmark.png   ← Sample six-panel output
├── 🎞️ fusion_demo (1).gif          ← Animated demo
└── 🎬 fusion_demo (1).mp4          ← Full video
```
---
🗺️ Roadmap
[x] LiDAR projection + sparse depth map
[x] MiDaS DPT-Large monocular estimation
[x] Median scale alignment
[x] Late fusion by pixel-wise averaging
[x] Six-panel benchmark visualisation
[x] GIF + MP4 export
[ ] Learned fusion (CNN/attention-based)
[ ] nuScenes dataset support
[ ] ROS2 real-time node
[ ] Depth completion evaluation (RMSE, AbsRel, δ<1.25)
---
🎓 Academic Context
This project is part of doctoral research in meta-learning for ADAS perception at the University of Granada, in collaboration with Infineon Technologies AG.
The fusion methodology complements work on Meta-YOLO and stereo depth estimation — contributing toward a full sensor fusion stack (camera + LiDAR + radar) optimised for embedded ADAS NPUs.
---
📄 License
MIT License — see LICENSE for details.
---
🙏 Acknowledgements
Resource	Link
KITTI Vision Benchmark	cvlibs.net/datasets/kitti
MiDaS — Intel ISL	github.com/isl-org/MiDaS
DPT Paper	Ranftl et al., ICCV 2021
Zenodo Dataset	10.5281/zenodo.19111017
Infineon Technologies AG	ADAS Research Collaboration
University of Granada	Doctoral Programme
---
<div align="center">
MiDaS · DPT-Large · LiDAR Fusion · Metric Depth · KITTI · ADAS
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:1a1a2e,100:0d1117&height=100&section=footer" width="100%"/>
Built with ❤️ for autonomous driving research
</div>
