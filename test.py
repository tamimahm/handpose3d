import os
import torch
import json
import re
import csv
from pypots.optim import Adam
from scipy.signal import savgol_filter
#print(torch.cuda.current_device())
import scipy.io
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
import torch
from pypots.utils.metrics import cal_mae
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyts.preprocessing import InterpolationImputer
from matplotlib.backends.backend_pdf import PdfPages
from pypots.imputation import Transformer
from sklearn.model_selection import train_test_split
# Define the test set size (e.g., 20%)
test_size = 0.2
# directory_path = r"D:\Chicago_study\Study_videos"
# data_dict = {}

# # Walk through the directory and its subdirectories
# for root, dirs, files in os.walk(directory_path):
#     for folder in dirs:
#         folder_path = os.path.join(root, folder)
#         calibration_folder = os.path.join(folder_path, "calibration")
#         # Check if the 'calibration' folder exists in the directory
#         if os.path.exists(calibration_folder):
#             for filename in os.listdir(calibration_folder):
#                 if filename.endswith(".csv") and "activityData" in filename:
#                     csv_path = os.path.join(calibration_folder, filename)
#                     # Read the CSV file and extract data
#                     with open(csv_path, 'r') as csv_file:
#                         csv_reader = csv.reader(csv_file)
#                         next(csv_reader)  # Skip the header row
#                         for row in csv_reader:
#                             activity = int(row[1])  # Second column
#                             activity_time = (int(row[4])/1000)*30  # Fifth column
#                             # Use regular expression to find numbers
#                             numbers = re.findall(r'\d+', folder)
#                             folders='ARAT_'+numbers[0]
#                             data_dict[folders] = {'Activity': activity, 'Activity Time': activity_time}



# Specify the filename to save the dictionary to
filename = 'D:/Chicago_study/files/label_dict.json'
excluded_names = {
    "ARAT_074_right_Impaired_activity3",
    "ARAT_074_right_Impaired_activity2",
    "ARAT_074_right_Impaired_activity1",
    "ARAT_066_left_Impaired_activity2",
    "ARAT_06_left_Impaired_activity6",
    "ARAT_06_left_Impaired_activity4",
    "ARAT_06_left_Impaired_activity3",
    "ARAT_06_left_Impaired_activity2",
    "ARAT_052_left_Impaired_activity1",
    "ARAT_04_left_Impaired_activity3",
    "ARAT_023_right_Impaired_activity6",
    "ARAT_023_right_Impaired_activity5"
}
# Load the dictionary from the file
with open(filename, 'r') as json_file:
    label_dict = json.load(json_file)

def process_time_series(time_series, target_length=238):
    current_length = len(time_series)

    if current_length > target_length:
        # If the length is greater than the target, interpolate to reduce it to target_length.
        interpolation_indices = np.linspace(0, current_length - 1, target_length).astype(int)
        interpolated_series = time_series[interpolation_indices]
        return interpolated_series
    elif current_length < target_length:
        # If the length is less than the target, zero-pad to reach target_length.
        zero_padding = np.zeros(target_length - current_length)
        padded_series = np.concatenate((time_series, zero_padding))
        return padded_series
    else:
        # If the length is already equal to the target, no need to change anything.
        return time_series

def find_change_indices(arr):
    change_indices = []
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            change_indices.append(i)
    return change_indices

        

set_key=['top','right','left']

# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
# Define the directory path
directory_path = r'D:\Chicago_study\final_hand_all_ARAT'
label_title=[]
object_dir_top=r'D:\Chicago_study\data_res\alternative\top'
object_dir_ipsi=r'D:\Chicago_study\data_res\alternative\ipsilateral'
object_dir_contra=r'D:\Chicago_study\data_res\alternative\contralateral'
for keys in set_key:
    pdf_path='D:/Chicago_study/'+keys+'plots.pdf'
    pdf_pages = PdfPages(pdf_path)
    
    # Initialize an empty list to store the data
    data = []
    data_label=[]
    test_data=[]
    title=[]
    target_samples = 238
    # Example usage:
    target_length = 238
    norm_dimension=1080
    original_length=[]
    test_label=[]
    task_id_train=[]
    hand_train=[]
    hand_test=[]
    task_id_test=[]
    obj_data_train=[]
    obj_data_test=[]
    impute_test=[]
    k=0
    ip= np.array([])
    t=np.array([])
    mtr=np.array([])
    pr=np.array([])

    ip_t= np.array([])
    t_t=np.array([])
    mtr_t=np.array([])
    pr_t=np.array([])
    # Iterate through patient folders
    for patient_folder in os.listdir(directory_path):
        patient_folder_path = os.path.join(directory_path, patient_folder)
        # Check if it's a directory
        if os.path.isdir(patient_folder_path):
            # Iterate through activity folders
            for activity_folder in os.listdir(patient_folder_path):
                if activity_folder not in excluded_names:
                    for pth, label_data in label_dict.items():
                        label_key=label_data['patient_id']+'_'+label_data['hand_id']+'_'+label_data['impaired_status']+'_'+label_data['activity']
                        label_title.append(label_key)
                        if label_key==activity_folder:
                            activity_folder_path = os.path.join(patient_folder_path, activity_folder)
                            if os.path.isdir(activity_folder_path) and 'keypoints' in os.listdir(activity_folder_path):
                                keypoints_folder_path = os.path.join(activity_folder_path, 'keypoints')
                                if any(file.endswith('.mat') for file in os.listdir(keypoints_folder_path)):
                                    # Iterate through .mat files
                                    for file in os.listdir(keypoints_folder_path):
                                        if file.endswith('.mat'):
                                            mat_file_path = os.path.join(keypoints_folder_path, file)
                                            key=file.split('.')[0]
                                            if key==keys:
                                                if activity_folder.split('_')[4]=='activity1' or activity_folder.split('_')[4]=='activity2' or activity_folder.split('_')[4]=='activity3'or activity_folder.split('_')[4]=='activity4' or activity_folder.split('_')[4]=='activity5' or activity_folder.split('_')[4]=='activity6':
                                                    # Load the .mat file
                                                    mat_data_ = scipy.io.loadmat(mat_file_path)
                                                    if mat_data_[key].shape[0]<450 and np.abs(np.min(mat_data_[key])-np.max(mat_data_[key]))>0.1:
                                                        # for jj in range(21):
                                                        # original_length.append(mat_data_[key].shape[0])
                                                        label_padded=np.zeros(mat_data_[key].shape[0])
                                                        indices_ch=find_change_indices(label_data['sequence'])
                                                        if len(indices_ch)==4:
                                                            if label_data['hand_id']=="right" and key=='left':
                                                                temp=mat_data_[key][indices_ch[0]-10:indices_ch[2]+10,0,1]
                                                            elif label_data['hand_id']=="left" and key=='right':
                                                                temp=mat_data_[key][indices_ch[0]-10:indices_ch[2]+10,0,1]
                                                            else:
                                                                temp=mat_data_[key][indices_ch[0]-10:indices_ch[3]+20,0,1]
                                                            temp=temp/norm_dimension
                                                            chunk=[]
                                                            start=0
                                                            for ch in range (3):
                                                                if ch==0:
                                                                    diff=indices_ch[ch+1]-(indices_ch[ch]-10)
                                                                else:
                                                                    diff=indices_ch[ch+1]-(indices_ch[ch])
                                                                chunk.append(temp[start:start+diff])
                                                                start=start+diff
                                                            chunk.append(temp[start:len(temp)])
                                                            ip = np.concatenate((ip, chunk[0][~np.isnan(chunk[0])]))
                                                            t = np.concatenate((t, chunk[1][~np.isnan(chunk[1])]))
                                                            mtr = np.concatenate((mtr,chunk[2][~np.isnan(chunk[2])]))
                                                            pr = np.concatenate((pr,chunk[3][~np.isnan(chunk[3])]))
                                                            ipt=chunk[0][~np.isnan(chunk[0])]
                                                            tt=chunk[1][~np.isnan(chunk[1])]
                                                            mtrt=chunk[2][~np.isnan(chunk[2])]
                                                            prt=chunk[3][~np.isnan(chunk[3])]
                                                            ip_t = np.concatenate((ip_t, [len(ipt)]))
                                                            t_t = np.concatenate((t_t, [len(tt)]))
                                                            mtr_t = np.concatenate((mtr_t,[len(mtrt)]))
                                                            pr_t = np.concatenate((pr_t,[len(prt)]))

                                                            if len(label_data['sequence'])<mat_data_[key].shape[0]:
                                                                label_padded[0:len(label_data['sequence'])]=label_data['sequence']
                                                            if len(label_data['sequence'])>mat_data_[key].shape[0]:
                                                                label_padded[0:len(label_data['sequence'])]=label_data['sequence'][0:mat_data_[key].shape[0]]
                                                            
                                                            #lab_data=label_padded
                                                            lab_data = process_time_series(label_padded, target_length)
                                                            nan_count=np.count_nonzero(np.isnan(temp))
                                                            if nan_count<0.2*len(temp) and nan_count!=0:#consider missing 10% of the values then do manual interpolation
                                                                temp[temp<0.05]=np.nan
                                                                imputer = InterpolationImputer(strategy='nearest')
                                                                temp = imputer.transform(temp[np.newaxis,:])[0,:]
                                                            temp[temp<0.05]=np.nan
                                                            #mat_data = process_time_series(temp, target_length)
                                                            nan_count=np.count_nonzero(np.isnan(temp))
                                                            if len(temp)!=0 and nan_count==0:
                                                                temp = savgol_filter(temp, window_length=5, polyorder=2)

                                                                mat_data=temp
                                                                
                                                                data.append(temp)
                                                                data_label.append(lab_data)
                                                                original_length.append(len(temp))
                                                                    
                                                                # Create a figure and axis
                                                                fig, ax = plt.subplots()

                                                                # Plot the original samples
                                                                ax.plot(range(len(temp)), temp)
                                                                # Set labels and legend
                                                                plt.xlabel('Sample Index')
                                                                plt.ylabel('Value')  
                                                                plt.title(activity_folder)  
                                                                
                                                                # Add the current figure to the PDF file
                                                                pdf_pages.savefig(fig)
                                                                
                                                                # Close the figure to free up memory
                                                                plt.close(fig)

                                                                test_data.append(temp)
                                                                test_label.append(lab_data)
                                                                title.append(activity_folder)
    # Close the PDF file
    pdf_pages.close()
    # Assuming you have a variable 'chunks' with 4 arrays
    chunks = [ip,t,mtr,pr]
    chunks_t=[ip_t,t_t,mtr_t,pr_t]
    def check_dist(chunks_t):
        name=[]
        param=[]
        # List of candidate distributions to consider
        distributions = [stats.norm, stats.expon, stats.uniform]
        for i, chunk in enumerate(chunks_t, 1):
            print(f'Chunk {i}:')
            best_fit_name, best_fit_params, best_aic = None, None, np.inf
            
            for distribution in distributions:
                # Fit the distribution to the data
                params = distribution.fit(chunk)
                
                # Calculate the Akaike Information Criterion (AIC) as a goodness-of-fit measure
                aic = stats.kstest(chunk, distribution.cdf, args=params)[0]
                
                if aic < best_aic:
                    best_fit_name, best_fit_params, best_aic = distribution.name, params, aic
            name.append(best_fit_name)
            param.append(best_fit_params)
        return name,param
    name_t,param_t=check_dist(chunks_t)
    name_v,param_v=check_dist(chunks)
    # Now 'data' contains the data from all three .mat files from all patient and activity folders
    n_d=len(data)
    n_t=len(test_data)
    
    # data=np.asarray(data)
    # test_data=np.asarray(test_data)
    # 
    # feature_test=np.zeros([n_t,target_samples,1])
    # feature_data[:,:,0]=data[:,:]
    # feature_test[:,:,0]=test_data[:,:]
    patch_length_min = 10
    patch_length_max = 40
    def create_mask(observations, feature_data):
        # Create a random mask for each observation
        masked_data = []

        for i in range(observations):
            time_steps=feature_data[i].shape[0]
            patch_length_min=time_steps*0.05
            patch_length_max=time_steps*0.2
            # Randomly determine the patch length and location
            patch_length = np.random.randint(patch_length_min, patch_length_max + 1)
            start_time = np.random.randint(0, time_steps - patch_length + 1)

            # Create a mask for the current observation
            mask = np.ones((time_steps))
            mask[start_time:start_time + patch_length] = np.nan

            # # Apply the mask to the data
            # masked_data[i, :, :] = mask
            # Apply the mask to your original data
            masked_data.append(mask * feature_data[i])
            # plt.figure(figsize=(12, 5))
            # plt.plot(feature_data[i], color='blue', marker='o')
            # plt.plot(masked_data[i], color='red', marker='x')
            # plt.show()
        return masked_data,feature_data
    
    masked,intact=create_mask(n_d,test_data)
    masked_data=np.zeros([n_d,target_samples,1])
    feature_data=np.zeros([n_d,target_samples,1])
    for i in range (n_d):
        masked_data[i,:,0] = process_time_series(masked[i], target_length)
        feature_data[i,:,0] = process_time_series(intact[i], target_length)
    #masked_data=create_mask(n_d,target_samples,feature_data)
    length=np.zeros([masked_data.shape[0],masked_data.shape[1],1])
    length[:,0,0]=original_length
    full_dataset=np.concatenate((masked_data,feature_data,length),axis=2)
    train, test = train_test_split(full_dataset, test_size=test_size, random_state=42)

    masked_data_train=train[:,:,0]
    masked_data_train=masked_data_train[:,:,np.newaxis]
    feature_data_train=train[:,:,1]
    feature_data_train=feature_data_train[:,:,np.newaxis]
    original_length_train=train[:,0,2]

    features = 1

    X_intact_train, X_train, missing_mask_train, indicating_mask_train = mcar(masked_data_train, 0.0) # hold out 10% observed values as ground truth
    # plt.figure(figsize=(12, 5))
    # #plt.plot(feature_data_train[0,:int(original_length_train[0]),0], color='blue', marker='o')
    # plt.plot(X_intact_train[0,:,0], color='red', marker='o')
    # plt.show()
    
    X_train_filled = masked_fill(X_train, 1 - missing_mask_train, np.nan)
    ##################remove padding from calculation##########################
    # Find the length without padding along dimension 1
    # length_without_padding = np.argmax(X_train[:, ::-1] != 0, axis=1)[:, ::-1] + 1

    # Find indices where missing_mask_train has zeros
    indices = np.where(missing_mask_train == 0)
    # Replace corresponding values in indicating_mask_train with 1 at those indices
    indicating_mask_train[indices] = 1
    leny=X_train.shape[0]
    for i in range(leny):
        indi=int(original_length_train[i])
        missing_mask_train[i,indi+1:target_length,0]=0
    # print(X_train_filled[10,:,0])
    # print(missing_mask_train[10,:,0])
    # print(indicating_mask_train[10,:,0])
    # print(feature_data_train[10,:,0])
    # Set the values of variable2 to zero where there is padding in variable1
    #missing_mask[np.arange(leny), :length_without_padding, :] = 0
    masked_data_test=test[:,:,0]
    masked_data_test=masked_data_test[:,:,np.newaxis]
    feature_data_test=test[:,:,1]
    feature_data_test=feature_data_test[:,:,np.newaxis]
    original_length_test=test[:,0,2]

    X_intact_test, X_test, missing_mask_test, indicating_mask_test = mcar(masked_data_test, 0.0) # hold out 10% observed values as ground truth
    
    X_test = masked_fill(X_test, 1 - missing_mask_test, np.nan)
    # Find indices where missing_mask_train has zeros
    indices = np.where(missing_mask_test == 0)
    # Replace corresponding values in indicating_mask_train with 1 at those indices
    indicating_mask_test[indices] = 1
    ######################################################remove padding from calculation###########
    # Find the length without padding along dimension 1
    # length_without_padding_test = np.argmax(X_test[:, ::-1] != 0, axis=1)[:, ::-1] + 1
    leny=X_test.shape[0]
    for i in range(leny):
        indi=int(original_length_test[i])
        missing_mask_test[i,indi+1:target_length,0]=0
    # plt.figure(figsize=(12, 5))
    # #plt.plot(feature_data_train[0,:int(original_length_train[0]),0], color='blue', marker='o')
    # plt.plot(X_intact_test[0,:,0], color='red', marker='o')
    # plt.show()
    
    # for i in range(masked_data.shape[0]):
    #     plt.figure(figsize=(12, 5))
    #     plt.plot(masked_data[i,:,0], color='blue', marker='o')
    #     plt.show()
    #     print('lala')

    # def train_test_splitter(dataset):
    #     # Define the split ratios
    #     train_ratio = 0.6
    #     validation_ratio = 0.2
    #     test_ratio = 0.2

    #     # Calculate the number of samples for each split
    #     num_samples = len(dataset)
    #     num_train = int(train_ratio * num_samples)
    #     num_validation = int(validation_ratio * num_samples)
    #     num_test = num_samples - num_train - num_validation

    #     # Use NumPy's array slicing to split the dataset
    #     train_set = dataset[:num_train]
    #     validation_set = dataset[num_train:num_train + num_validation]
    #     test_set = dataset[num_train + num_validation:]
    #     return train_set,validation_set,test_set


    #train dataset
    # x_train,x_validation,x_test=train_test_splitter(X)
    # x_train_intact,x_validation_intact,x_test_intact=train_test_splitter(feature_data)
    # x_train_missing,x_validation_missing,x_test_missing=train_test_splitter(missing_mask)
    # x_train_indicating,x_validation_indicating,x_test_indicating=train_test_splitter(indicating_mask)
    #ori_train,ori_val,ori_test=train_test_splitter(original_length)
    dataset = {"X": X_train,
               "X_intact":feature_data_train,
               "missing_mask":missing_mask_train,
               "indicating_mask":indicating_mask_train}
    # validation_dataset={"X": x_validation,
    #         "X_intact":x_validation_intact,
    #        "missing_mask":x_validation_missing,
    #        "indicating_mask":x_validation_indicating}
    test_dataset={"X": X_test,
                "X_intact":feature_data_test,
               "missing_mask":missing_mask_test,
               "indicating_mask":indicating_mask_test}
    print(dataset["X"].shape)
    print(test_dataset["X"].shape)
    # print(validation_dataset["X"].shape)
    root="D:/Chicago_study/impute/model_base_tv_likeli_final_nonan_smooth"
    imputed_dataset_save=root+keys+"/"+"imputed.npy"
    test_dataset_save=root+keys+"/"+"test_set.npy"
    dataset_save=root+keys+"/"+"dataset.npy"
    saving_path=root+keys+"/"
    mae_save=root+keys+"/"+"mae.txt"
    psnr_save=root+keys+"/"+"psnr.txt"
    pdf_path=saving_path+'plots.pdf'
    # initialize the model
    saits = SAITS(
        device=[torch.device('cuda:0')],#,torch.device('cuda:1')],
        n_steps=target_samples,
        n_features=features,
        n_layers=2,
        d_model=128,
        d_inner=64,
        n_heads=16,
        d_k=8,
        d_v=8,
        dropout=0.1,
        batch_size=8,
        optimizer=Adam(lr=1e-3),
        epochs=1500,
        patience=50,
        saving_path=saving_path, # set the path for saving tensorboard logging file and model checkpoint
        model_saving_strategy="best", # only save the model with the best validation performance
    )
    # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
    saits.fit(train_set=dataset)
    # # impute the originally-missing values and artificially-missing values
    #model_path="D:/Chicago_study/model_sait/20230926_T171553/SAITS.pypots"
    #saits.load_model("D:/Chicago_study/model_sait/20230922_T073128/SAITS.pypots")
    imputation = saits.impute(test_dataset)
    # Save the imputed data as a NumPy array
    np.save(imputed_dataset_save, imputation)
    np.save(test_dataset_save, test_dataset['X'])
    np.save(dataset_save, dataset['X'])
    # Create a PdfPages object to save the plots to a PDF file
    pdf_pages = PdfPages(pdf_path)
    # Choose a specific sample (e.g., the first one)
    for i in range(len(imputation)):
        imputed = imputation[i][:,0]#plot wrist
        not_imputed = test_dataset['X'][i][:,0]#plot wrist
        intact_value=test_dataset['X_intact'][i][:,0]

        if len(imputed)<int(original_length_test[i]):
            imputed= process_time_series(imputed, int(original_length_test[i]))
            not_imputed= process_time_series(not_imputed, int(original_length_test[i]))
            intact_value= process_time_series(intact_value, int(original_length_test[i]))
        else:
            imputed= imputed[:int(original_length_test[i])]
            not_imputed=  not_imputed[:int(original_length_test[i])]
            intact_value=  intact_value[:int(original_length_test[i])]
        # Create an array of indices for the NaN values
        nan_indices = np.isnan(not_imputed)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the original samples
        ax.plot(range(len(not_imputed)), not_imputed, marker='o', label='missing', color='blue',markersize=2)

        # Overlay the predicted missing values
        ax.scatter(np.where(nan_indices)[0], imputed[nan_indices], marker='x', label='Predicted', color='red',s=3)

        # # Overlay the predicted missing values
        ax.scatter(np.where(nan_indices)[0], intact_value[nan_indices], marker='x', label='gt', color='green',s=3)

        # Set labels and legend
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()   
        plt.title(title[i])  
        
        # Add the current figure to the PDF file
        pdf_pages.savefig(fig)
        
        # Close the figure to free up memory
        plt.close(fig)
    # Close the PDF file
    pdf_pages.close()
    print('done')
    # # # calculate mean absolute error on the ground truth (artificially-missing values)
    # mae = cal_mae(imputation, X_intact_test, indicating_mask_test)
    # print(mae)
    # np.save(mae_save, mae)
    # dynamic_range=255
    # # Convert MAE to MSE
    # mse = (mae / dynamic_range) ** 2

    # # Calculate PSNR
    # psnr = 10 * np.log10((dynamic_range ** 2) / mse)
    # print(psnr)
    # np.save(psnr_save, psnr)
