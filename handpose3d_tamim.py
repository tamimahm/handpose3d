import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [1920, 1080]

def run_mp(input_stream1,img_path):#, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)

    caps = [cap0]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])
    #create hand keypoints detector object.
    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =2, min_tracking_confidence=0.5)

    #containers for detected keypoints for each camera
    kpts_cam0_l = []
    kpts_cam0_r = []
    count=0
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()


        if not ret0 : break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        # if frame0.shape[1] != 720:
            # frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            # frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False

        results0 = hands0.process(frame0)
        print(results0.multi_handedness)
        hand_count=0
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                hand_count+=1
        frame0_keypoints_l = []
        frame0_keypoints_r = []
        loop_count=0
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                loop_count+=1
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    #0 index left hand 1 index right hand
                    pxl_x_l = int(round(frame0.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y_l = int(round(frame0.shape[0]*hand_landmarks.landmark[p].y))

                    pxl_x_r = int(round(frame0.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y_r = int(round(frame0.shape[0]*hand_landmarks.landmark[p].y))
                    kpts_l = [pxl_x_l, pxl_y_l]
                    kpts_r = [pxl_x_r, pxl_y_r]
                    frame0_keypoints_l.append(kpts_l)
                    frame0_keypoints_r.append(kpts_r)

        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*21


        kpts_cam0_l.append(frame0_keypoints_l)
        kpts_cam0_r.append(frame0_keypoints_r)

        # Draw the hand annotations on the image.
        frame0.flags.writeable = True

        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)


        if results0.multi_hand_landmarks:
          for hand_landmarks in results0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #cv.imshow('cam0', frame0)
        file_name=img_path+'/'+str(count)+'.jpg'
        cv.imwrite(file_name,frame0)
        count+=1
        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()
    print()
    return kpts_cam0_l,kpts_cam0_r

if __name__ == '__main__':
    input_stream1 = 'media/test2.mp4'
    img_path='media/images/'
    kpts_cam0_l,kpts_cam0_r = run_mp(input_stream1,img_path)#, P0, P1)

