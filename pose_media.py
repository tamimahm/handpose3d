import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from scipy.io import savemat
import os
from utils import DLT, get_projection_matrix, write_keypoints_to_disk
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

frame_shape = [1920, 1080]

def run_mp(input_stream1,img_path):#, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    # cap1 = cv.VideoCapture(input_stream2)
    # cap2 = cv.VideoCapture(input_stream3)
    caps = [cap0]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for pose detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])
    #create pose keypoints detector object.
    pose0 = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    #containers for detected keypoints for each camera
    kpts_cam0 = []
    count=0
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()

        if not ret0 : break
        
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame0.flags.writeable = False
        results0 = pose0.process(frame0)
        #frame0 kpts
        frame0_keypoints = []
        if results0.pose_landmarks:
            for p in range(33):
                pxl_x = int(round(frame0.shape[1]*results0.pose_landmarks.landmark[p].x))
                pxl_y = int(round(frame0.shape[0]*results0.pose_landmarks.landmark[p].y))
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)

        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*33

        kpts_cam0.append(frame0_keypoints)
        # Draw the pose annotations on the image.
        frame0.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        annotated_image = frame0.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results0.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #cv.imshow('cam0', annotated_image)
        # if results0.pose_landmarks:
          # for pose_landmarks in results0.pose_landmarks:
            # mp_drawing.draw_landmarks(frame0, pose_landmarks, mp_pose.POSE_CONNECTIONS)         
        #cv.imshow('cam0', frame0)
        file_name=img_path+'/'+str(count)+'.jpg'
        cv.imwrite(file_name,annotated_image)
        count+=1
        # k = cv.waitKey(1)
        # if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()
    print(len(kpts_cam0[0]))
    return np.array(kpts_cam0)#, np.array(kpts_3d)

if __name__ == '__main__':
    rootdir = 'D:/Chicago_study/impaired/'
    mmpose_dir='D:/Chicago_study/NSF_DARE_imp/'
    folder_dir=[]
    folder_name=[]
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            folder_dir.append(d)
            folder_name.append(file)
    for i in range(0,len(folder_name)):
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
                if activity_id==1 or activity_id==7 or activity_id==11:
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