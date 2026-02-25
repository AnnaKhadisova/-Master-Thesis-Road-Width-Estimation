# -Master-Thesis-Road-Width-Estimation
This repository contains the implementation used for estimating road width from monocular video. The script processes each frame of a video and estimates the road width at a predefined depth. The script produces:
1) Output Video
The output video contains:
- Raw road segmentation overlay (before filtering)
- Estimated road width
- Depth at which width is measured
- Current frame number (top right corner)
2)  CSV File

Contains per-frame:
Frame number
Depth
Estimated road width

3) PLY Files in archive
3D point clouds are saved per frame.


Installation:
git clone <repository-link>
cd Master-Thesis-Road-Width-Estimation

Run:
python road_width_estimate.py

Before running the script, you must configure the following parameters inside road_width_estimate.py:
1) Path to the original input video
video_path = "path/to/your/video.mp4"
2) Depth Value (depth_m)
This defines the distance (in meters) at which the road width is measured.
This must be manually adjusted.

You should:
Run the script
Observe the output
Adjust the depth value until the measurement is stable and meaningful
3) YOLO weights
model = YOLO("path/to/your/best.pt")

In the repository two files with weights are uploaded. For the road width estimation pipeline please use best.pt file. Another weights best_mapillary_vistas.pt are the weights to detect lane markings and crosswalks

4) Output Resolution (out_w, out_h)
This must match the resolution used by the Depth Anything model.
You must:
- Run depth estimation on a single frame.
- Check the output depth resolution.
- Adjust all resizing operations in the script accordingly.

In our case:
(504, 322)

5) Camera Intrinsic Parameters
fx, fy
cx, cy
and distortion coefficients.
These values must come from your camera calibration.
