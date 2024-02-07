import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    # Video capture settings
    cap = cv2.VideoCapture(0)

    #Hand definitions
    mpHand = mp.solutions.hands
    hands = mpHand.Hands()
    mpDraw = mp.solutions.drawing_utils

    # Actually the program calculates distance between two finger, we must give which fingers
    calculated_distances = [[5, 4], [6,8], [10,12], [14,16], [18,20]]

    while True:
        success, img = cap.read()
        
        if success:
            # It's optional, we used mirror effect
            img = cv2.flip(img, 1)
            
            # BGR to RGB Color conversion
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process hands to count
            results = hands.process(img_rgb)
            
            # Finger counter
            counter = 0

            # When record every fingers, this condition will use
            if results.multi_hand_landmarks:
                
                # Motions array for record positions of all fingers
                motions = []
                
                for handLms in results.multi_hand_landmarks:
                    # Draw 20 landmarks
                    # mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
                    
                    for id, lm in enumerate(handLms.landmark):
                        h,w,c = img.shape
                        # Convert ratios to reel positions
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        
                        # If it's head finger, calculate avg. (exceptional situation)
                        if id == 4:
                            cy = ((cy + motions[3][2]) / 2) + cap.get(4) / 30
                            
                        # Add finger landmark position [id, coordinat x, coordinat y]
                        motions.append([id,cx, cy])
            
                
                for item in calculated_distances:
                    downFingerPosY  = motions[item[0]][2]
                    upperFingerPosY = motions[item[1]][2]
                    # If down landmark of finger y position bigger than upper:
                    # The finger increases counter
                    isFingerOpen = downFingerPosY > upperFingerPosY
                    counter += 1 if isFingerOpen else 0

            cv2.rectangle(img, (60, img.shape[0] - 85), (110, 570), (255, 255, 255), -1)        
            cv2.putText(img, str(int(counter)), (70, img.shape[0] - 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)







