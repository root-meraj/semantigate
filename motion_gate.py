import cv2

def start_motion_gate():
    cap= cv2.VideoCapture(0)
    
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    print("motion gate activated press 'q' to stop")
    
    while cap.isOpened():
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2= cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        #applying gaussion blur to smooth our sensor noise 
        
        blur1 = cv2.GaussianBlur(gray1, (11,11),0)
        blur2 = cv2.GaussianBlur(gray2, (11,11),0) #(11,11) is the kernal size bigger the kernal more blur
        
        #now calculating the absolute difference between the two frames
        diff = cv2.absdiff(blur1,blur2)
        # this is will help us to change the pixels id the pixel changes from
        # more than 25 outof 255 and turn it to 255 (white))
        _, thresh = cv2.threshold(diff,12,255,cv2.THRESH_BINARY)
        thersh = cv2.dilate(thresh, None, iterations=2)
        #count the changed pixels
        changed_pixels = cv2.countNonZero(thresh)         
        
        #motion detection gateway
        #if the pixels changes the model wil trigger yolo to check for frames
        if changed_pixels > 500:
            cv2.putText(frame1, "motion detected" , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            print(f"Motion! Changed pixels: {changed_pixels} -> Waking ML Model...")
        else:
            cv2.putText(frame1, 'NO motion detected',(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            
            
        cv2.imshow('webcam feed', frame1)
        cv2.imshow("threshold", thresh)
        frame1= frame2
        ret,frame2= cap.read()
        
        if cv2.waitKey(10) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    start_motion_gate()