# Research context and sources

These sources establish that video-based pose estimation and occlusion robustness are active research areas. They do **not** establish that FrameTrust's specific evaluation setup is novel.

1. Hongwei Fang, Jiahang Cai, Xun Wang, Wenwu Yang. “Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation.” CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Fang_Beyond_Static_Frames_Temporal_Aggregate-and-Restore_Vision_Transformer_for_Human_Pose_CVPR_2026_paper.html
   - Relevant because it studies temporal coherence in video-based human-pose estimation and discusses unstable predictions in motion blur, occlusion, and defocus.
2. Rohit Lal et al. “STRIDE: Single-Video Based Temporally Continuous Occlusion-Robust 3D Pose Estimation.” WACV 2025. https://openaccess.thecvf.com/content/WACV2025/html/Lal_STRIDE_Single-Video_Based_Temporally_Continuous_Occlusion-Robust_3D_Pose_Estimation_WACV_2025_paper.html
   - Relevant because it studies temporal continuity and occlusion robustness for 3D pose estimation.
3. MediaPipe Pose documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
   - Consult the current upstream documentation if MediaPipe APIs change; this starter currently uses the legacy `mp.solutions.pose` interface for a short setup path.

## Data policy

No video is redistributed in this repository. Use your own recordings, videos you have permission to process, or a properly licensed research dataset. Record provenance and license for every clip. Obtain consent before processing identifiable people where required.
