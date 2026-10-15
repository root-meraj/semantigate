import cv2
import time
import mediapipe as mp

def start_gesture_pipeline():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    # Setting max hands to 1 keeps the math clean and fast for a real-time demo
    hands_ai = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    time.sleep(1) 
    
    average_bg = None
    
    # MediaPipe IDs for the tips of the Index, Middle, Ring, and Pinky fingers
    tip_ids = [8, 12, 16, 20]
    
    print("Gesture Pipeline Activated. Press 'q' to stop.")
    
    while cap.isOpened():
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
        
        # --- THE DECISION LOGIC ---
        if changed_pixels > 500:
            cv2.putText(frame, "MOTION -> AI ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_ai.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # --- GESTURE MATH ---
                    open_fingers = 0
                    lm = hand_landmarks.landmark
                    
                    # Check the 4 main fingers
                    for tip_id in tip_ids:
                        # If the Tip Y < Lower Knuckle Y, the finger is pointing up
                        # (Remember: In computer vision, a lower Y value means higher up on the screen)
                        if lm[tip_id].y < lm[tip_id - 2].y:
                            open_fingers += 1
                            
                    # Map the finger count to a gesture name
                    gesture_name = "UNKNOWN"
                    if open_fingers == 0:
                        gesture_name = "FIST"
                    elif open_fingers == 2:
                        gesture_name = "PEACE SIGN"
                    elif open_fingers == 4:
                        gesture_name = "OPEN HAND / STOP"
                        
                    # Print the gesture on the screen in big blue text
                    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'IDLE -> AI SLEEPING', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.imshow('Gesture Edge Pipeline', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    hands_ai.close()
        
if __name__ == "__main__":
    start_gesture_pipeline()