import cv2
from ultralytics import YOLOWorld
import argparse
import numpy as np
import time

class EventDetector:
    def __init__(self, model_path="yolov8s-world.pt", classes=None):
        self.model = YOLOWorld(model_path)
        if classes:
            self.model.set_classes(classes)
        self.classes = classes
        self.track_history = {} # id -> list of (x, y)
        self.stop_counters = {} # id -> frame_count
        
        # Thresholds
        self.WALKING_THRESHOLD = 5.0 # pixels displacement
        self.STOP_FRAMES_THRESHOLD = 15 # frames to consider stopped
        self.CROWD_THRESHOLD = 5 # people count
        
    def detect_events(self, frame):
        # Run tracking
        results = self.model.track(frame, persist=True, verbose=False)
        
        events = []
        annotated_frame = results[0].plot()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()
            
            person_count = 0
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x, y, w, h = box
                cls_name = results[0].names[cls_id]
                
                # Update history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                    self.stop_counters[track_id] = 0
                
                self.track_history[track_id].append((float(x), float(y)))
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                    
                # Logic: Person Walking
                if cls_name == 'person':
                    person_count += 1
                    if len(self.track_history[track_id]) > 1:
                        prev_x, prev_y = self.track_history[track_id][-2]
                        dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                        if dist > self.WALKING_THRESHOLD:
                            events.append(f"Person {track_id} Walking")
                            cv2.putText(annotated_frame, "Walking", (int(x-w/2), int(y-h/2)-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Logic: Vehicle Stopping
                if cls_name in ['car', 'truck', 'bus', 'vehicle']:
                    if len(self.track_history[track_id]) > 1:
                        prev_x, prev_y = self.track_history[track_id][-2]
                        dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                        
                        if dist < self.WALKING_THRESHOLD: # Using same threshold for "not moving much"
                            self.stop_counters[track_id] += 1
                        else:
                            self.stop_counters[track_id] = 0
                            
                        if self.stop_counters[track_id] > self.STOP_FRAMES_THRESHOLD:
                            events.append(f"Vehicle {track_id} Stopping")
                            cv2.putText(annotated_frame, "Stopping", (int(x-w/2), int(y-h/2)-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Logic: Crowded Scene
            if person_count >= self.CROWD_THRESHOLD:
                events.append(f"Crowded Scene ({person_count} people)")
                cv2.putText(annotated_frame, "CROWDED", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                          
        return annotated_frame, events

def main(video_path, output_path="output.mp4"):
    detector = EventDetector(classes=["person", "car", "bus", "truck"])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Video writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, events = detector.detect_events(frame)
        
        # Display FPS
        frame_count += 1
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed
        cv2.putText(processed_frame, f"FPS: {current_fps:.2f}", (width - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if events:
            print(f"Frame {frame_count}: {events}")
            
        out.write(processed_frame)
        
    cap.release()
    out.release()
    print(f"Processing complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video")
    args = parser.parse_args()
    
    main(args.video, args.output)
