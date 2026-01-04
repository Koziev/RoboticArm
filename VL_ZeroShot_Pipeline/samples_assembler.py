import re
import os
import glob
import json
import shutil
import pathlib
import tqdm
import cv2
import PIL
from PIL import Image, ImageDraw
import numpy as np


if __name__ == "__main__":
    output_dir = "./compiler_samples"
    pathlib.Path(output_dir).mkdir(exist_ok=True)

    data_root_dir = "."

    frames = []
    for frame_fp in glob.glob(os.path.join(data_root_dir, "camera_frames/*.json")):
        with open(frame_fp) as f:
            frame_data = json.load(f)
            frames.append(frame_data)

    assembled_samples = []
    samples_dir = os.path.join(data_root_dir, "samples")
    for sample_dir in os.listdir(samples_dir):
        data_fp = os.path.join(samples_dir, sample_dir, "data.json")
        if os.path.exists(data_fp):
            with open(data_fp) as f:
                sample_data = json.load(f)

            action_point = None

            if sample_data["touch"] is True:
                action_point = sample_data["R_touch"]
            else:
                if "Z_by_touch2" in sample_data:
                    z = sample_data["Z_by_touch2"]
                    action_point = sample_data["finish"]

            if action_point is not None:
                sample_data["action_point"] = action_point
                sample_data["sample_dir"] = sample_dir

                ts_ns = sample_data["timestamp_ns"]
                best_delta = 1_000_000_000
                best_frame = None
                for frame in frames:
                    delta = abs(frame["timestamp_ns"] - ts_ns)
                    if delta < best_delta:
                        best_delta = delta
                        best_frame = frame

                if best_delta < 1_000_000_000:  # менее 1 сек
                    if best_frame["gripper"] is not None:
                        sample_data["frame"] = best_frame
                        assembled_samples.append(sample_data)

    for sample in tqdm.tqdm(assembled_samples, desc="Copying bound frames"):
        frame_fp = sample["frame"]["frame_fp"]
        frame_fn = os.path.basename(frame_fp)
        output_fp = os.path.join(output_dir, frame_fn)
        sample["frame"]["frame_fp"] = output_fp
        shutil.copy(frame_fp, output_fp)

    print(f"Saving {len(assembled_samples)} assembled samples")
    with open(os.path.join(output_dir, "trajectories.json"), "w") as f:
        json.dump(assembled_samples, f, indent=4)

    # Для визуального контроля отобразим все финальные точки.
    frame0 = PIL.Image.open(assembled_samples[0]["frame"]["frame_fp"]).convert('RGB')

    background_color = (0, 0, 0)
    image = Image.new('RGB', frame0.size, color=background_color)
    draw = ImageDraw.Draw(image)

    for sample in assembled_samples:
        #x, y, radius, color = circle
        x, y = sample["frame"]["gripper"]["center"]
        color = (255,255,0)
        radius = 5

        # Calculate bounding box for the circle
        left = x - radius
        top = y - radius
        right = x + radius
        bottom = y + radius

        # Draw the circle
        draw.ellipse([left, top, right, bottom], fill=color, outline=color)

        # Save the image
    image.save(os.path.join(output_dir, "action_points.png"), 'PNG')

    print("All done :)")

