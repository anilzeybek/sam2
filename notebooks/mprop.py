import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import cv2
import time


video_dir = "./videos/kitchen/frames/"
masks_dir = "./videos/kitchen/masks/"
# reference frames (indices) for which corresponding mask PNG files exist in masks_dir
ref_frames = [17, 340]

####

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# (Optional) collect available mask files (named like '<frame_idx>.png')
available_mask_files = {
    int(os.path.splitext(p)[0]): os.path.join(masks_dir, p)
    for p in os.listdir(masks_dir)
    if os.path.splitext(p)[-1].lower() == ".png" and os.path.splitext(p)[0].isdigit()
}

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

#######
# Add masks for each reference frame listed in ref_frames.
# Assumes mask file naming pattern '<frame_idx>.png'. Converts to boolean array.

ann_obj_id = 1  # single object id used across reference frames
for frame_idx in ref_frames:
    if frame_idx not in available_mask_files:
        raise FileNotFoundError(f"Mask file for frame {frame_idx} not found in {masks_dir}")
    mask_path = available_mask_files[frame_idx]
    mask_img = Image.open(mask_path)
    mask_np = np.array(mask_img)
    # if mask has multiple channels, take first / convert to grayscale
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    # binarize (treat any non-zero as foreground)
    mask_bool = mask_np > 0
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=ann_obj_id,
        mask=mask_bool,
    )
    print(f"Added mask for frame {frame_idx} (obj {ann_obj_id}) from '{mask_path}'.")

start_time = time.perf_counter()
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
end_time = time.perf_counter()
print(f"Video segmentation completed in {end_time - start_time:.2f} seconds")


# =====================
# Create visualization video with overlaid masks
# =====================

if len(frame_names) == 0:
    raise RuntimeError("No frames found to write video.")

print(f"Found {len(frame_names)} frame files")
print(f"Video segments available for {len(video_segments)} frames")

first_frame_path = os.path.join(video_dir, frame_names[0])
first_img = Image.open(first_frame_path).convert("RGB")
frame_w, frame_h = first_img.size
print(f"Frame dimensions: {frame_w}x{frame_h}")

# Prepare video writer (OpenCV expects BGR)
out_path = "output-sam2.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30  # default; replace with actual if known
writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

if not writer.isOpened():
    print("Error: Could not open video writer")
    # Try alternative codec
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = "output.avi"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print("Error: Could not open video writer with XVID codec either")
        exit(1)

# Pre-choose a color map for up to N objects (here single object, but generalize)
obj_colors = {}
rng = np.random.default_rng(0)

frames_written = 0
sorted_frame_indices = sorted(video_segments.keys())
print(f"Processing {len(sorted_frame_indices)} frames with predictions")

for idx in sorted_frame_indices:
    # Try different frame file patterns
    frame_file = None
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        # Try exact match first
        candidate = f"{idx}{ext}"
        if candidate in frame_names:
            frame_file = candidate
            break
        # Try zero-padded versions
        for pad in range(1, 8):
            candidate = f"{idx:0{pad}d}{ext}"
            if candidate in frame_names:
                frame_file = candidate
                break
        if frame_file:
            break
    
    if frame_file is None:
        print(f"Warning: Could not find frame file for index {idx}")
        continue
        
    img_path = os.path.join(video_dir, frame_file)
    if not os.path.exists(img_path):
        print(f"Warning: Frame file {img_path} does not exist")
        continue
        
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    overlay = img_np.copy()
    masks_dict = video_segments[idx]
    for obj_id, mask in masks_dict.items():
        if obj_id not in obj_colors:
            color = rng.integers(64, 255, size=3, dtype=np.uint8)  # Brighter colors
            obj_colors[obj_id] = color
        else:
            color = obj_colors[obj_id]
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask_bool = mask.astype(bool)
        
        if mask_bool.any():  # Only process if mask has content
            # Create colored overlay
            overlay[mask_bool] = (0.6 * overlay[mask_bool] + 0.4 * color).astype(np.uint8)
            # Optionally draw contour
            try:
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color.tolist(), thickness=2)
            except Exception as e:
                print(f"Warning: Could not draw contours for frame {idx}: {e}")

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    writer.write(bgr)
    frames_written += 1

writer.release()
print(f"Saved visualization video to {out_path} with {frames_written} frames")
