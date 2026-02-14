import cv2
import numpy as np

def create_video(output_path="sample.mp4", duration=10, fps=30):
    width, height = 1280, 720
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # "Person" (Green ball) moving from left to right
    person_x = 100
    person_y = 500
    
    # "Vehicle" (Red box) stationary
    vehicle_x = 800
    vehicle_y = 500
    
    # "Crowd" (Blue balls) appear later
    crowd_start_frame = 0
    
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw ground
        cv2.rectangle(frame, (0, 600), (1280, 720), (50, 50, 50), -1)
        
        # Draw "Vehicle" (Stationary)
        # To make YOLO detect it as a car, it helps if it looks vaguely like one or we just rely on shape/color context if training was specialized, 
        # but standard YOLO trained on COCO might struggle with a simple red rectangle. 
        # We will just label it ourselves if we were training, but for pre-trained YOLO, we hope it picks up something or we'll just test the logic with the code assuming detections work.
        # Actually, standard YOLO won't detect a red rectangle as a car. 
        # This dummy video is mainly for *logic* testing if we mock the detector, but here we are using a REAL detector.
        # So this dummy video might fail to trigger detections. 
        # We should probably download a real sample video if possible, or paste a real image into the video frames.
        # However, since I can't access external internet for random large files easily without tools, I'll rely on the user providing a video 
        # OR I will try to make the "shapes" slightly more realistic or just accept that I might need to mock the detector for the "verification" step if real detection fails on shapes.
        
        # Let's try to make a simple "car" shape
        cv2.rectangle(frame, (vehicle_x, vehicle_y), (vehicle_x+200, vehicle_y+100), (0, 0, 255), -1) # Body
        cv2.rectangle(frame, (vehicle_x+50, vehicle_y-50), (vehicle_x+150, vehicle_y), (0, 0, 255), -1) # Roof
        cv2.circle(frame, (vehicle_x+40, vehicle_y+100), 30, (0, 0, 0), -1) # Wheel
        cv2.circle(frame, (vehicle_x+160, vehicle_y+100), 30, (0, 0, 0), -1) # Wheel
        
        # Draw "Person" (Moving)
        # Simple Stick figure / Blob
        cv2.circle(frame, (int(person_x), person_y), 20, (200, 200, 200), -1) # Head
        cv2.line(frame, (int(person_x), person_y+20), (int(person_x), person_y+80), (200, 200, 200), 5) # Body
        
        person_x += 2 # Move right
        
        out.write(frame)
        
    out.release()
    print(f"Created {output_path}")

if __name__ == "__main__":
    create_video()
