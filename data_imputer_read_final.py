import os
import torch
import json
import re
import csv
from pypots.optim import Adam
#print(torch.cuda.current_device())
import scipy.io
import numpy as np
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
    k=0
    # Iterate through patient folders
    for patient_folder in os.listdir(directory_path):
        patient_folder_path = os.path.join(directory_path, patient_folder)
        # Check if it's a directory
        if os.path.isdir(patient_folder_path):
            # Iterate through activity folders
            for activity_folder in os.listdir(patient_folder_path):
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
                                            original_length.append(mat_data_[key].shape[0])
                                            mat_data = process_time_series(mat_data_[key][:,0,1], target_length)/norm_dimension
                                            mat_data[mat_data<0.05]=np.nan
                                            nan_count=np.count_nonzero(np.isnan(mat_data))
                                            if nan_count<0.1*mat_data_[key].shape[0]:#consider missing 10% of the values then do manual interpolation
                                                imputer = InterpolationImputer(strategy='nearest')
                                                mat_data = imputer.transform(mat_data[np.newaxis,:])[0,:]
                                                data.append(mat_data)
                                                # plt.figure(figsize=(12, 5))
                                                # plt.plot(mat_data, color='blue', marker='o')
                                                # plt.show()
                                                # print('lala')
                                            test_data.append(mat_data)
                                            title.append(activity_folder)



    # Now 'data' contains the data from all three .mat files from all patient and activity folders
    n_d=len(data)
    n_t=len(test_data)
    data=np.asarray(data)
    test_data=np.asarray(test_data)
    feature_data=np.zeros([n_d,target_samples,1])
    feature_test=np.zeros([n_t,target_samples,1])
    feature_data[:,:,0]=data[:,:]
    feature_test[:,:,0]=test_data[:,:]
    patch_length_min = 10
    patch_length_max = 50
    def create_mask(observations, time_steps,feature_data):
        # Create a random mask for each observation
        masked_data = np.zeros((observations, time_steps, 1))

        for i in range(observations):
            # Randomly determine the patch length and location
            patch_length = np.random.randint(patch_length_min, patch_length_max + 1)
            start_time = np.random.randint(0, time_steps - patch_length + 1)

            # Create a mask for the current observation
            mask = np.ones((time_steps, 1))
            mask[start_time:start_time + patch_length] = np.nan

            # Apply the mask to the data
            masked_data[i, :, :] = mask

        # Apply the mask to your original data
        masked_data = masked_data * feature_data
        return masked_data
    masked_data=create_mask(n_d,target_samples,feature_data)
    test_masked_data=create_mask(n_t,target_samples,feature_test)
    features = 1
    X_intact, X, missing_mask, indicating_mask = mcar(masked_data, 0.1) # hold out 10% observed values as ground truth
    X = masked_fill(X, 1 - missing_mask, np.nan)
    y_intact, y, missing_mask_test, indicating_mask_test = mcar(test_masked_data, 0.1) # hold out 10% observed values as ground truth
    y = masked_fill(y, 1 - missing_mask_test, np.nan)
    # for i in range(masked_data.shape[0]):
    #     plt.figure(figsize=(12, 5))
    #     plt.plot(masked_data[i,:,0], color='blue', marker='o')
    #     plt.show()
    #     print('lala')
    dataset = {"X": X,
               "X_intact":feature_data,
               "missing_mask":missing_mask,
               "indicating_mask":indicating_mask}
    test_dataset={"X": y}#,
            #     "X_intact":y_intact,
            #    "missing_mask":missing_mask_test,
            #    "indicating_mask":indicating_mask_test}
    print(dataset["X"].shape)
    print(test_dataset["X"].shape)
    root="D:/Chicago_study/impute/model_test"
    imputed_dataset_save=root+keys+"/"+"imputed.npy"
    test_dataset_save=root+keys+"/"+"test_set.npy"
    dataset_save=root+keys+"/"+"dataset.npy"
    saving_path=root+keys+"/"
    mae_save=root+keys+"/"+"mae.txt"
    pdf_path=saving_path+'plots.pdf'
    # initialize the model
    saits = SAITS(
        device=[torch.device('cuda:0'),torch.device('cuda:1')],
        n_steps=target_samples,
        n_features=features,
        n_layers=2,
        d_model=512,
        d_inner=256,
        n_heads=32,
        d_k=16,
        d_v=16,
        dropout=0.1,
        batch_size=8,
        optimizer=Adam(lr=1e-3),
        epochs=1000,
        patience=20,
        saving_path=saving_path, # set the path for saving tensorboard logging file and model checkpoint
        model_saving_strategy="best", # only save the model with the best validation performance
    )
    # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
    saits.fit(dataset)
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
        if len(imputed)>original_length[i]:
            imputed= process_time_series(imputed, original_length[i])
            not_imputed= process_time_series(not_imputed, original_length[i])
        # Create an array of indices for the NaN values
        nan_indices = np.isnan(not_imputed)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the original samples
        ax.plot(range(len(not_imputed)), not_imputed, marker='o', label='Original', color='blue',markersize=2)

        # Overlay the predicted missing values
        ax.scatter(np.where(nan_indices)[0], imputed[nan_indices], marker='x', label='Predicted', color='red',s=3)

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
    # # calculate mean absolute error on the ground truth (artificially-missing values)
    # mae = cal_mae(imputation, y_intact, indicating_mask_test)
    # print(mae)
    # np.save(mae_save, mae)
