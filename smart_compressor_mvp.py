import cv2
import numpy as np
import mss

def start_timeline_compressor():
    sct = mss.mss()
    monitor = sct.monitors[1]  # Primary monitor
    
    # 1. Setup the Video Writer (Saving as MP4)
    width = monitor['width']
    height = monitor['height']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Recording at 10 FPS to keep the prototype lightweight
    out = cv2.VideoWriter('smart_recording.mp4', fourcc, 10.0, (width, height))
    
    average_bg = None
    last_active_frame = None
    
    print("Timeline Preserver Active. Press 'q' to stop recording.")
    
    while True:
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        # Convert from screen format (BGRA) to standard video format (BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        if last_active_frame is None:
            last_active_frame = frame
            
        # Downscale for the motion gate math
        small_frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        
        if average_bg is None:
            average_bg = blur.copy().astype("float")
            continue
            
        cv2.accumulateWeighted(blur, average_bg, 0.05)
        diff = cv2.absdiff(blur, cv2.convertScaleAbs(average_bg))
        
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        changed_pixels = cv2.countNonZero(thresh)        
        
        # --- THE TIMELINE COMPRESSION LOGIC ---
        if changed_pixels > 1000:
            # MOTION DETECTED: Update to the new frame
            last_active_frame = frame  
            status_text = "RECORDING (Motion Detected)"
            color = (0, 0, 255)
        else:
            # IDLE: Do not update the frame. We will duplicate the old one.
            status_text = "IDLE (Duplicating Frame for Compression)"
            color = (0, 255, 0)
            
        # 2. Write the frame to the MP4 file
        # If idle, feeding the exact same frame allows the MP4 codec to compress it massively.
        out.write(last_active_frame)
        
        # 3. Create a preview window (we use a copy so the text isn't burned into the final video)
        preview = cv2.resize(last_active_frame, (640, 360))
        cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Smart Video Compressor', preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup and save the file
    out.release()
    cv2.destroyAllWindows()
    print("\nRecording Finished! Video saved to your folder as 'smart_recording.mp4'.")

if __name__ == "__main__":
    start_timeline_compressor()