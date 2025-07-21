import cv2
from ultralytics import YOLO
import numpy as np

class RealTimeObjectTracker:
    def __init__(self, model_name='yolo11s.pt'):
        """
        Initialize the real-time object tracker
        
        Args:
            model_name (str): YOLO model name (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        """
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.cap = None
        
    def start_tracking(self, camera_index=0, confidence_threshold=0.5, show_fps=True):
        """
        Start real-time object tracking
        
        Args:
            camera_index (int): Camera index (usually 0 for default webcam)
            confidence_threshold (float): Minimum confidence threshold for detections
            show_fps (bool): Whether to display FPS counter
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting real-time object tracking...")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        frame_count = 0
        fps_counter = 0
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Run YOLO inference
                results = self.model(frame, conf=confidence_threshold, verbose=False)
                
                # Annotate the frame with detection results
                annotated_frame = results[0].plot()
                
                # Add FPS counter if requested
                if show_fps:
                    fps_counter += 1
                    if fps_counter % 30 == 0:  # Update FPS every 30 frames
                        fps = self.cap.get(cv2.CAP_PROP_FPS)
                        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the annotated frame
                cv2.imshow('Real-time Object Tracking', annotated_frame)
                
                frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Tracking stopped and resources cleaned up")

def main():
    """Main function to run the object tracker"""
    try:
        # Create tracker instance
        tracker = RealTimeObjectTracker(model_name='yolo11s.pt')  # You can change to yolo11n.pt, yolo11m.pt, etc.

        # Start tracking with custom parameters
        tracker.start_tracking(
            camera_index=0,           # Change if you have multiple cameras
            confidence_threshold=0.5,  # Adjust confidence threshold as needed
            show_fps=True             # Set to False if you don't want FPS display
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Application terminated")

if __name__ == "__main__":
    main()