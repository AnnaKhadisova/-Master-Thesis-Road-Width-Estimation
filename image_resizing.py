import cv2
import glob
import os

input_folder = "/home/k64684/semantic-depth/calibration/images/Camera_Calibration"   
output_folder = "/home/k64684/semantic-depth/calibration/images_calibration_depth_anything" 
os.makedirs(output_folder, exist_ok=True)

target_width  = 504
target_height = 322

png_files = glob.glob(os.path.join(input_folder, "*.jpg"))

print(f"Found {len(png_files)} PNG images.")

for path in png_files:
    img = cv2.imread(path)

    if img is None:
        print(f"Could not read {path}")
        continue

    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    filename = os.path.basename(path)
    save_path = os.path.join(output_folder, filename)

    cv2.imwrite(save_path, resized)
    print(f"Saved {save_path}")

print("DONE! All images resized.")
