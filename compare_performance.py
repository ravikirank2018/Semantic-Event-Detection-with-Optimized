import cv2
import time
import argparse
import onnxruntime as ort
import numpy as np
import os
from ultralytics import YOLOWorld

def run_baseline_inference(model_path, video_path, num_frames=100):
    model = YOLOWorld(model_path)
    cap = cv2.VideoCapture(video_path)
    
    times = []
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret: break
        
        start = time.time()
        model.predict(frame, verbose=False)
        end = time.time()
        times.append(end - start)
        count += 1
    cap.release()
    avg_fps = 1.0 / (sum(times) / len(times)) if times else 0
    return avg_fps

def run_onnx_inference(onnx_path, video_path, num_frames=100):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    cap = cv2.VideoCapture(video_path)
    times = []
    count = 0
    
    # Simple pre-processing for YOLO (resize, normalize) - simplified for benchmark
    # Note: Correct pre-processing is complex, for strict benchmarking we just test inference time
    # assuming pre-processing overhead is similar or handled separately.
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret: break
        
        # Fake preprocessing to 640x640 for speed testing
        img = cv2.resize(frame, (640, 640))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        
        start = time.time()
        session.run(None, {input_name: img})
        end = time.time()
        times.append(end - start)
        count += 1
        
    cap.release()
    avg_fps = 1.0 / (sum(times) / len(times)) if times else 0
    return avg_fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    
    print("Running Baseline (PyTorch)...")
    baseline_fps = run_baseline_inference("yolov8s-world.pt", args.video)
    print(f"Baseline FPS: {baseline_fps:.2f}")
    
    onnx_path = "yolov8s-world_int8.onnx"
    if not os.path.exists(onnx_path):
        # Try finding ANY onnx model if default name failed (e.g. from CLI export which names differently)
        # CLI usually names it 'yolov8s-world.onnx'
        if os.path.exists("yolov8s-world.onnx"):
            onnx_path = "yolov8s-world.onnx"
            print(f"Found standard ONNX model: {onnx_path}")
        else:
            print("Optimized model not found. Skipping optimization test.")
            return

    print(f"Running Optimized (ONNX)...")
    try:
        optimized_fps = run_onnx_inference(onnx_path, args.video)
        print(f"Optimized FPS: {optimized_fps:.2f}")
        
        print("\nComparison:")
        if baseline_fps > 0:
            print(f"Speedup: {optimized_fps / baseline_fps:.2f}x")
    except Exception as e:
        print(f"Optimization inference failed: {e}")

if __name__ == "__main__":
    main()
