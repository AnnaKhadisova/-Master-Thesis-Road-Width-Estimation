import os
import cv2
import csv
import numpy as np
import open3d as o3d
import semantic_depth_lib.pcl as pcl
from ultralytics import YOLO
import torch
from depth_anything_3.api import DepthAnything3
import tarfile
import tempfile

depth_m = 7  # you need to adjust it
video_path = "/home/k64684/data_from_Marcus/Export_Ulm/extracted_ulm/export_ulm/video_ulm.mp4"
output_csv_path = "/home/k64684/master_thesis/road_width_results.csv"
output_ply_archive_path = "/home/k64684/master_thesis/ply_outputs.tar.gz"
output_video_path = "/home/k64684/master_thesis/road_width_output.mp4"
start_frame = 0
end_frame = 900

# csv output
csv_file = open(output_csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "depth_m", "road_width_m"])

# YOLOv11  weights
model = YOLO("/home/k64684/master_thesis/best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model = DepthAnything3.from_pretrained(
    "depth-anything/DA3NESTED-GIANT-LARGE"
).to(device).eval()

# video input
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
fps = cap.get(cv2.CAP_PROP_FPS)
out_w, out_h = 504, 322
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
frame_idx = start_frame

# create line between end points in point cloud
def create_3d_line_points(left_pt, right_pt, color_rgb=(255, 0, 0), step=0.002):
    p1 = left_pt[0].reshape(3)
    p2 = right_pt[0].reshape(3)
    p1[1] += 0.01
    p2[1] += 0.01
    v = p2 - p1
    t_vals = np.arange(0.0, 1.0 + step, step)
    line_pts = np.array([p1 + t * v for t in t_vals]).reshape(-1, 3)
    line_colors = np.tile(np.array(color_rgb)/255.0, (line_pts.shape[0], 1))
    return line_pts, line_colors

tar = tarfile.open(output_ply_archive_path, mode="w:gz")

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
    #if not ret or frame_idx >= end_frame:
        break

    frame_idx += 1
    original = cv2.resize(frame, (1920, 1080))
    frame_small = cv2.resize(original, (504, 322))
    # Image segmentation step
    result = model.predict(
        source=original,
        task="segment",
        conf=0.25,
        save=False
    )[0]

    road_mask = None
    if result.masks is not None:
        for i, cls in enumerate(result.boxes.cls):
            if result.names[int(cls)] == "road":
                road_mask = result.masks.data[i]
                break

    if road_mask is None:
        # ROAD NOT DETECTED
        csv_writer.writerow([frame_idx, depth_m, "Road not detected"])

        # Overlay message on video
        vis = frame_small.copy()
        cv2.putText(vis, "Road not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis, f"Depth: {depth_m} m", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, f"{frame_idx}", (vis.shape[1] - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        video_writer.write(vis)
        continue

    road_mask = cv2.resize(
        road_mask.cpu().numpy().astype(np.uint8),
        (504, 322),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Depth estimation step
    prediction = depth_model.inference(
        image=[frame_small],
        process_res=504,
        process_res_method="upper_bound_resize"
    )
    depth = prediction.depth[0]

    fx, fy = 468.79080982, 464.59313093
    cx, cy = 251.96219581, 158.37158287

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    # Distortion coefficients
    dist = np.array([
        -0.23365291,   
        0.22819251,  
        -0.00038628,  
        0.00059548,   
        -0.29640178   
    ], dtype=np.float64)
    pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2).astype(np.float32)  # (N,1,2)
    pts_undist = cv2.undistortPoints(pts, K, dist, P=K)  # (N,1,2)
    u_undist = pts_undist[:, 0, 0].reshape(h, w)
    v_undist = pts_undist[:, 0, 1].reshape(h, w)

    X = (u - cx) * depth / fx   # here you can use u_undist, v_undist 
    Y = (v - cy) * depth / fy
    Z = depth
    points3D = np.stack((X, -Y, -Z), axis=-1)
    colors = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB) / 255.0

    road3D = points3D[road_mask]
    road_colors = colors[road_mask]

    points3D = points3D.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    # Point cloud filtering
    road3D, road_colors = pcl.remove_from_to(road3D, road_colors, 2, 0.0, 7.0)
    road3D, road_colors = pcl.remove_noise_by_mad(road3D, road_colors, 1, 15.0)
    road3D, road_colors = pcl.remove_noise_by_mad(road3D, road_colors, 0, 2.0)
    road3D, road_colors, _, _, _ = pcl.remove_noise_by_fitting_plane(
        road3D, road_colors, axis=1, threshold=5.0
    )

    # Road width calculation
    left_pt, right_pt = pcl.get_end_points_of_road(road3D, depth_m - 0.02)

    if left_pt is None or right_pt is None:
        csv_writer.writerow([frame_idx, depth_m, "Road not detected"])

        # Overlay message on video
        vis = frame_small.copy()
        cv2.putText(vis, "Road not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis, f"Depth: {depth_m} m", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, f"{frame_idx}", (vis.shape[1] - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        video_writer.write(vis)
        continue

    dist_rw = abs(left_pt[0][0] - right_pt[0][0])
    csv_writer.writerow([frame_idx, depth_m, f"{dist_rw:.2f}"])
    print(f"[Frame {frame_idx}] Road width = {dist_rw:.2f} m")

    # Create line between end points and add it to the point cloud
    line_pts, line_colors = create_3d_line_points(left_pt, right_pt)
    points3D_all = np.vstack((points3D, line_pts))
    colors_all = np.vstack((colors, line_colors))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D_all)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)

    # Save into TAR.GZ
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmpfile:
        tmp_path = tmpfile.name
    o3d.io.write_point_cloud(tmp_path, pcd)

    tarinfo = tarfile.TarInfo(name=f"frame_{frame_idx:05d}.ply")
    tarinfo.size = os.path.getsize(tmp_path)
    with open(tmp_path, "rb") as f:
        tar.addfile(tarinfo, f)
    os.remove(tmp_path)

    # Video overlay
    vis = frame_small.copy()
    vis[road_mask] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 0.5, frame_small, 0.5, 0)
    cv2.putText(vis, f"Road width: {dist_rw:.2f} m", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, f"Depth: {depth_m} m", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, f"{frame_idx}", (vis.shape[1] - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    video_writer.write(vis)

cap.release()
video_writer.release()
csv_file.close()
tar.close()
print("Done. CSV, video, and PLY archive saved.")


