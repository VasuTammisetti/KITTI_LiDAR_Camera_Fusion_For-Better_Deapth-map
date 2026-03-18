# KITTI_LiDAR_Camera (MiDAS)_Fusion_For-Better_Deapth-map

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VasuTammisetti/KITTI_LiDAR_Camera_Fusion_For-Better_Deapth-map/blob/main/Camera_LiDAR_Fusion.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Late sensor fusion of LiDAR and camera data for dense metric depth estimation on KITTI.**  
This project combines the **accuracy of sparse LiDAR** with the **density of monocular depth (MiDaS)** via median scaling and averaging, producing a depth map that is both **dense** and **metrically accurate** – a critical capability for autonomous driving, robotics, and augmented reality.

---

**Demo**  
[![Fusion Demo GIF](https://github.com/VasuTammisetti/KITTI_LiDAR_Camera_Fusion_For_Better_Depth-map/raw/main/fusion_demo%20(1).gif)
](https://github.com/VasuTammisetti/KITTI_LiDAR_Camera_Fusion_For_Better_Depth-map/raw/refs/heads/main/fusion_demo%20(1).gif)
 <img width="767" height="490" alt="image" src="https://github.com/user-attachments/assets/034e5c2c-eb8f-4520-aba0-de6b78b3cc97" />

<img width="1451" height="936" alt="image" src="https://github.com/user-attachments/assets/e749048c-5f46-4771-b955-4b5dca536a7c" />




- **LiDAR projection** onto the image plane using KITTI calibration matrices.
- **Sparse-to-dense interpolation** of LiDAR depth via nearest‑neighbor search.
- **Monocular depth estimation** with a state‑of‑the‑art transformer (MiDaS DPT‑Large).
- **Scale alignment** by matching the median of monocular depth to LiDAR depth.
- **Late fusion** by pixel‑wise averaging of scaled monocular and dense LiDAR depths.
- **Comprehensive visualization** of each processing step for easy benchmarking.
- **Automated output generation**: six‑panel benchmark images, animated GIF, and MP4 video.

---

## 🧠 How It Works (Methodology)

The pipeline follows a **late fusion** approach:

1. **Project LiDAR points** to image coordinates using calibration.
2. **Create a sparse depth map** only at projected pixel locations.
3. **Interpolate** to obtain a dense LiDAR depth map.
4. **Estimate monocular depth** with MiDaS (raw, scale‑ambiguous).
5. **Scale** the monocular depth so its median matches the LiDAR median – this gives it a metric scale.
6. **Fuse** by averaging the scaled monocular and dense LiDAR depth maps.

All metric depth maps are displayed with the same color scale (0–50 m) for direct visual comparison.

---

## 📊 Comparison to Conventional Methods

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **LiDAR only** | Accurate metric depth | Extremely sparse, gaps between scan lines |
| **Monocular depth (MiDaS)** | Dense, preserves object boundaries | Scale‑ambiguous, may have local errors |
| **Stereo depth** | Dense, metric if calibrated | Requires two cameras, calibration, higher computational cost |
| **MiDAS fusion** | ✅ **Dense + metric** – combines the best of both sensors | Requires calibration and synchronized data |

Unlike stereo, our method needs only one camera. Unlike pure monocular depth, it provides absolute scale. Compared to LiDAR alone, it fills all gaps and preserves edges.

---

## 🚀 Applications

- **Autonomous driving**: Dense metric depth for obstacle detection, free space estimation, and path planning.
- **Robotics**: Grasping, navigation, and SLAM in unstructured environments.
- **Augmented / Virtual Reality**: Occlusion handling, realistic placement of virtual objects.
- **Surveillance**: People counting and tracking with accurate 3D positions.
- **Industrial automation**: Quality control and dimension measurement.

---

## 🏆 State‑of‑the‑Art Context

Monocular depth estimation has advanced rapidly with vision transformers (e.g., **MiDaS v3.1**, **DPT**). However, single‑image methods cannot recover absolute scale without additional cues. Fusing with even sparse LiDAR provides that scale while retaining the dense prediction capability. This late‑fusion approach is lightweight, requires no retraining, and can be applied to any camera–LiDAR setup with known calibration.

Recent research (e.g., **CMT‑Depth**, **GuideDepth**) shows that learning‑based fusion can achieve even higher accuracy, but our simple median‑scaling baseline already demonstrates the core benefit of fusion and is easy to implement.

---

## 📈 Results

<img width="1451" height="936" alt="image" src="https://github.com/user-attachments/assets/c1c483a1-67a7-447e-8924-b53c551bacf9" />




- (a) Camera image (reference)
- (b) Sparse LiDAR depth (accurate but sparse)
- (c) Dense LiDAR depth (interpolated, fills gaps but may blur edges)
- (d) Raw monocular depth (dense but scale‑ambiguous)
- (e) Scaled monocular depth (now metric, preserves edges)
- (f) **Fused depth** ✅ – the best of both worlds: accurate + dense

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8 or higher
- A machine with a GPU (optional, but recommended for faster inference)
- The **KITTI Object Detection** dataset (training split)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VasuTammisetti/KITTI_LiDAR_Camera_Fusion_For-Better_Deapth-map.git
   cd KITTI_LiDAR_Camera_Fusion_For-Better_Deapth-map
