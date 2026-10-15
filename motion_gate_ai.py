import cv2
import time
import csv
import psutil
import mediapipe as mp

def start_benchmarking_pipeline():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_ai = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    
    average_bg = None
    print("Benchmarking Active. Press 'q' to stop and save data.")

    # Create and open a CSV file to store our research metrics
    with open('research_metrics.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Timestamp", "System_State", "FPS", "CPU_Usage_%", "RAM_Usage_%"])
        
        while cap.isOpened():
            start_time = time.time() # Start stopwatch for FPS calculation
            
            ret, frame = cap.read()
            if not ret: break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            
            if average_bg is None:
                average_bg = blur.copy().astype("float")
                continue
                
            cv2.accumulateWeighted(blur, average_bg, 0.01)
            diff = cv2.absdiff(blur, cv2.convertScaleAbs(average_bg))
            
            _, thresh = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            changed_pixels = cv2.countNonZero(thresh)        
            current_state = "IDLE"
            
            if changed_pixels > 500:
                current_state = "ACTIVE"
                cv2.putText(frame, "MOTION -> AI ACTIVE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_ai.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(frame, 'IDLE -> AI SLEEPING', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow('Benchmarking Pipeline', frame)
            
            # Stop stopwatch and calculate metrics
            fps = 1.0 / (time.time() - start_time)
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            
            # Log the data to the CSV file
            csv_writer.writerow([time.time(), current_state, round(fps, 2), cpu, ram])
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    hands_ai.close()
    print("Data saved to research_metrics.csv!")
        
if __name__ == "__main__":
    start_benchmarking_pipeline()