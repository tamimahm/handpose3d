import cv2
import math
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# with mp_hands.Hands(
    # static_image_mode=True,
    # max_num_hands=2,
    # min_detection_confidence=0.7) as hands:
    # # Convert the BGR image to RGB, flip the image around y-axis for correct 
    # # handedness output and process it with MediaPipe Hands.
    # results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

    # # Print handedness (left v.s. right hand).
    # print(f'Handedness of:')
    # print(results.multi_handedness)

    # # if not results.multi_hand_landmarks:
      # # continue
    # # Draw hand landmarks of each hand.
    # print(f'Hand landmarks of:')
    # image_hight, image_width, _ = image.shape
    # annotated_image = cv2.flip(image.copy(), 1)
    # for hand_landmarks in results.multi_hand_landmarks:
      # # Print index finger tip coordinates.
      # print(
          # f'Index finger tip coordinate: (',
          # f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          # f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
      # )
      # mp_drawing.draw_landmarks(
          # annotated_image,
          # hand_landmarks,
          # mp_hands.HAND_CONNECTIONS,
          # mp_drawing_styles.get_default_hand_landmarks_style(),
          # mp_drawing_styles.get_default_hand_connections_style())
    # resize_and_show(cv2.flip(annotated_image, 1))
frame_shape = [1920, 1080]
def run_mp(input_stream1,img_path):#, P0, P1):
    #input video stream
    cap0 = cv2.VideoCapture(input_stream1)
    caps = [cap0]
    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])
    #create hand keypoints detector object.
    hands0 = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands =2, min_tracking_confidence=0.7)
    #containers for detected keypoints for each camera
    kpts_cam0 = []
    count=0
    while True:
        #read frames from stream
        ret0, frame0 = cap0.read()
        if not ret0 : break
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        image_hight, image_width, _ = frame0.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        results = hands0.process(frame0)
        #prepare list of hand keypoints of this frame
        #frame0 kpts
        frame0_keypoints = []
        for hand_world_landmarks in results.multi_hand_world_landmarks:
          mp_drawing.plot_landmarks(
            hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
        frame0_keypoints = []
        if results.multi_hand_world_landmarks:
            for hand_world_landmarks in results.multi_hand_world_landmarks:
                for p in range(21):
                    pxl_x = int(round(image_width*hand_world_landmarks.landmark[p].x))
                    pxl_y = int(round(image_hight*hand_world_landmarks.landmark[p].y))
                    pxl_z = int(round(2880*hand_world_landmarks.landmark[p].z))
                    kpts = [pxl_x, pxl_y, pxl_z]
                    frame0_keypoints.append(kpts) 
        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1,-1]]*21
        kpts_cam0.append(frame0_keypoints)


        # Draw the hand annotations on the image.
        frame0.flags.writeable = True

        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)

        #cv.imshow('cam0', frame0)
        file_name=img_path+'/'+str(count)+'.jpg'
        cv2.imwrite(file_name,frame0)
        count+=1
        k = cv2.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0)

if __name__ == '__main__':
    input_stream1 = 'media/test2.mp4'
    img_path='media/images/'
    kpts_cam0 = run_mp(input_stream1,img_path)#, P0, P1)
    print(kpts_cam0.shape)

# Run MediaPipe Hands and plot 3d hands world landmarks.
# with mp_hands.Hands(
    # static_image_mode=True,
    # max_num_hands=2,
    # min_detection_confidence=0.7) as hands:
    # # Convert the BGR image to RGB and process it with MediaPipe Hands.
    # results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # image_hight, image_width, _ = image.shape
    # # Draw hand world landmarks.
    # print(f'Hand world landmarks of :')
    # # if not results.multi_hand_world_landmarks:
      # # continue
    # for hand_world_landmarks in results.multi_hand_world_landmarks:
      # mp_drawing.plot_landmarks(
        # hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
    # frame0_keypoints = []
    # if results.multi_hand_world_landmarks:
        # for hand_world_landmarks in results.multi_hand_world_landmarks:
            # print(hand_world_landmarks.landmark[0].x)
            # print(int(round(image_width*hand_world_landmarks.landmark[0].x)))
            # for p in range(21):
                # pxl_x = int(round(image_width*hand_world_landmarks.landmark[p].x))
                # pxl_y = int(round(image_hight*hand_world_landmarks.landmark[p].y))
                # #pxl_z = int(round(frame0.shape[0]*hand_world_landmarks.landmark[p].z))
                # kpts = [pxl_x, pxl_y]
                # frame0_keypoints.append(kpts)    