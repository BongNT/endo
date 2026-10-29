"""
YOLO Video Inference Script
This script performs inference on a video using a trained YOLO model.
Outputs:
- Annotated video with bounding boxes
- Text files with bounding box coordinates for each frame.
"""

from __future__ import annotations

import gc
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO


def inference_video(
    model_path: str,
    video_path: str,
    output_dir: str | None = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    save_video: bool = True,
    save_txt: bool = True,
):
    """Run YOLO inference on a video file.

    Args:
        model_path: Path to the YOLO model weights (.pt file)
        video_path: Path to the input video file
        output_dir: Directory to save outputs (default: creates 'inference_output' in same dir as video)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        save_video: Whether to save annotated video
        save_txt: Whether to save bounding box text files
    """
    # Load YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Setup paths
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent / "inference_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for video output
    video_output_dir = output_dir / "videos"
    video_output_dir.mkdir(exist_ok=True)

    # Output paths
    output_video_path = video_output_dir / f"{video_path.stem}_annotated.mp4"
    output_txt_path = output_dir / f"{video_path.stem}.txt"

    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print("-" * 60)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print("-" * 60)

    # Setup video writer if saving video
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Open single text file for all detections
    txt_file = None
    if save_txt:
        txt_file = open(output_txt_path, "w")

    # Process video frame by frame
    frame_idx = 0

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]

        # Save bounding boxes to text file (YOLO format with frame number)
        if save_txt and txt_file is not None:
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates in xyxy format and move to CPU immediately
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Convert to YOLO format (normalized xywh)
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / width
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / height
                    w = (xyxy[2] - xyxy[0]) / width
                    h = (xyxy[3] - xyxy[1]) / height

                    # Write to file: frame class x_center y_center width height confidence
                    txt_file.write(f"{frame_idx} {cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        # Save annotated frame to video
        if save_video and video_writer is not None:
            # Get annotated frame from results
            annotated_frame = results.plot(labels=False, txt_color=(255, 255, 0))
            video_writer.write(annotated_frame)
            # Delete annotated frame to free memory
            del annotated_frame

        # Explicitly delete results to free memory
        del results

        # Clear GPU cache periodically to prevent memory accumulation
        if frame_idx % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Progress update
        if (frame_idx + 1) % 100 == 0 or frame_idx == 0:
            print(f"Processed frame {frame_idx + 1}/{total_frames}")

        frame_idx += 1

    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    if txt_file is not None:
        txt_file.close()

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("-" * 60)
    print("Processing complete!")
    print(f"Total frames processed: {frame_idx}")

    if save_txt:
        print(f"Bounding box labels saved to: {output_txt_path}")

    if save_video:
        print(f"Annotated video saved to: {output_video_path}")

    print(f"\nAll outputs saved in: {output_dir}")

    return output_dir


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "/home/bongmedai/Endo/ultralytics/runs/detect/yolo_det_single_cls/weights/best.pt"
    VIDEO_PATH = "/home/bongmedai/Endo/datasets/2.mpg"
    OUTPUT_DIR = "/home/bongmedai/Endo/ultralytics/video_inference_output"

    # Run inference
    inference_video(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.5,
        iou_threshold=0.45,
        save_video=True,
        save_txt=True,
    )
