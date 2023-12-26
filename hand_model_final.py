import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from scipy.io import savemat
import os
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [1920, 1080]

def run_mp(input_stream1,img_path):#, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    # cap1 = cv.VideoCapture(input_stream2)
    # cap2 = cv.VideoCapture(input_stream3)
    caps = [cap0]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])
    #create hand keypoints detector object.
    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)
    # hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)
    # hands2 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)
    #containers for detected keypoints for each camera
    kpts_cam0 = []
    # kpts_cam1 = []
    # kpts_cam2 = [] 
    # kpts_3d = []
    count=0
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        # ret1, frame1 = cap1.read()
        # ret2, frame2 = cap2.read()

        if not ret0 : break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        # if frame0.shape[1] != 720:
            # frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            # frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        # frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        # frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        # frame1.flags.writeable = False
        # frame2.flags.writeable = False
        results0 = hands0.process(frame0)
        # results1 = hands1.process(frame1)
        # results2 = hands2.process(frame2)
        #prepare list of hand keypoints of this frame
        #frame0 kpts
        frame0_keypoints = []
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame0.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame0.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame0_keypoints.append(kpts)

        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*21

        kpts_cam0.append(frame0_keypoints)

        #frame1 kpts
        # frame1_keypoints = []
        # if results1.multi_hand_landmarks:
            # for hand_landmarks in results1.multi_hand_landmarks:
                # for p in range(21):
                    # #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    # pxl_x = int(round(frame1.shape[1]*hand_landmarks.landmark[p].x))
                    # pxl_y = int(round(frame1.shape[0]*hand_landmarks.landmark[p].y))
                    # kpts = [pxl_x, pxl_y]
                    # frame1_keypoints.append(kpts)

        # else:
            # #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            # frame1_keypoints = [[-1, -1]]*21

        # #update keypoints container
        # kpts_cam1.append(frame1_keypoints)

        # #frame2 kpts
        # frame2_keypoints = []
        # if results2.multi_hand_landmarks:
            # for hand_landmarks in results2.multi_hand_landmarks:
                # for p in range(21):
                    # #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    # pxl_x = int(round(frame2.shape[1]*hand_landmarks.landmark[p].x))
                    # pxl_y = int(round(frame2.shape[0]*hand_landmarks.landmark[p].y))
                    # kpts = [pxl_x, pxl_y]
                    # frame2_keypoints.append(kpts)

        # else:
            # #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            # frame2_keypoints = [[-1, -1]]*21

        # #update keypoints container
        # kpts_cam2.append(frame2_keypoints)
        #calculate 3d position
        # frame_p3ds = []
        # for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            # if uv1[0] == -1 or uv2[0] == -1:
                # _p3d = [-1, -1, -1]
            # else:
                # _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            # frame_p3ds.append(_p3d)

        # '''
        # This contains the 3d position of each keypoint in current frame.
        # For real time application, this is what you want.
        # '''
        # frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        # kpts_3d.append(frame_p3ds)

        # Draw the hand annotations on the image.
        frame0.flags.writeable = True
        # frame1.flags.writeable = True
        # frame2.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        # frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
        # frame2= cv.cvtColor(frame2, cv.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
          for hand_landmarks in results0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # if results1.multi_hand_landmarks:
          # for hand_landmarks in results1.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        # if results2.multi_hand_landmarks:
          # for hand_landmarks in results2.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS)            
        #cv.imshow('cam0', frame0)
        file_name=img_path+'/'+str(count)+'.jpg'
        cv.imwrite(file_name,frame0)
        raw_img=cv.imread(file_name)
        resized_img = cv.resize(raw_img, (320, 180),interpolation = cv.INTER_LINEAR)
        cv.imwrite(file_name,resized_img)

        count+=1
        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0)#, np.array(kpts_3d)

if __name__ == '__main__':
    rootdir = 'D:/Chicago_study/all_ARAT_videos/'
    mmpose_dir='D:/Chicago_study/final_hand_all_ARAT/'
    folder_dir=[]
    folder_name=[]
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            folder_dir.append(d)
            folder_name.append(file)
    for i in range(90,len(folder_name)):
        open_pt=mmpose_dir+folder_name[i]
        if not os.path.exists(open_pt):
            os.mkdir(open_pt)    
        for videos in os.listdir(folder_dir[i]):
            #parsing the filename 
            videos=videos.replace(" ","")
            video_name=videos.split('.')[0]
            fold_split=video_name.split('_')
            p_id=fold_split[1]
            h_id=fold_split[2]
            imp_id=fold_split[3]
            cam_id=int(fold_split[4].replace('cam',""))
            activity_id=int(fold_split[5].replace('activity',""))        
            if cam_id==3:
                cam_name='top'
            if cam_id==1:
                cam_name='right'
            if cam_id==2:
                cam_name='back'
            if cam_id==4:
                cam_name='left'  
            if cam_id!=2 and imp_id=="Impaired":    
                if activity_id!=17 or activity_id!=18 or activity_id!=19:
                    print(i)
                    print(videos)
                    #create  openpose directories
                    video_path=rootdir+folder_name[i]+'/'+videos
                    kp_dir=open_pt+'/ARAT_'+p_id+'_'+h_id+'_'+imp_id+'_activity'+str(activity_id)
                    if not os.path.exists(kp_dir):
                        os.mkdir(kp_dir)   
                    key_dir=kp_dir+'/keypoints'
                    if not os.path.exists(key_dir):
                        os.mkdir(key_dir)  
                    img_path=kp_dir+'/images_CAM'+str(cam_id) 
                    if not os.path.exists(img_path):
                        os.mkdir(img_path)   
                    kpts_cam0 = run_mp(video_path,img_path)#, P0, P1)
                    if cam_id==3:
                        file_name=key_dir+'/'+'top.mat'
                        savemat(file_name, {'top': kpts_cam0})
                    if cam_id==1:
                        file_name=key_dir+'/'+'right.mat'
                        savemat(file_name, {'right': kpts_cam0}) 
                    if cam_id==4:
                        file_name=key_dir+'/'+'left.mat'
                        savemat(file_name, {'left': kpts_cam0}) 

            # if cam_id==2 :    
            #     if activity_id==17 or activity_id==18 or activity_id==19:
            #         print(i)
            #         print(videos)
            #         #create  openpose directories
            #         video_path=rootdir+folder_name[i]+'/'+videos
            #         kp_dir=open_pt+'/ARAT_'+p_id+'_'+h_id+'_'+imp_id+'_activity'+str(activity_id)
            #         if not os.path.exists(kp_dir):
            #             os.mkdir(kp_dir)   
            #         key_dir=kp_dir+'/keypoints'
            #         if not os.path.exists(key_dir):
            #             os.mkdir(key_dir)  
            #         img_path=kp_dir+'/images_CAM'+str(cam_id) 
            #         if not os.path.exists(img_path):
            #             os.mkdir(img_path)   
            #         kpts_cam0 = run_mp(video_path,img_path)#, P0, P1)
            #         if cam_id==2:
            #             file_name=key_dir+'/'+'back.mat'
            #             savemat(file_name, {'back': kpts_cam0})
