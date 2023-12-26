import os
import torch
import json
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
# initialize the model
from pypots.optim import Adam
from pypots.classification import BRITS
# Assuming you have a variable 'data' with shape (760, 238, 21)


# Now, 'normalized_data' contains the normalized values between 0 and 1
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

for keys in set_key:
    # Initialize an empty list to store the data
    data = []
    data_label=[]
    test_data=[]
    title=[]
    target_samples = 200
    # Example usage:
    target_length = target_samples
    norm_dimension=1080
    original_length=[]
    test_label=[]
    k=0
    # Iterate through patient folders
    for patient_folder in os.listdir(directory_path):
        patient_folder_path = os.path.join(directory_path, patient_folder)
        
        # Check if it's a directory
        if os.path.isdir(patient_folder_path):
            # Iterate through activity folders
            for activity_folder in os.listdir(patient_folder_path):
                for pth, label_data in label_dict.items():
                    label_key=label_data['patient_id']+'_'+label_data['hand_id']+'_'+label_data['impaired_status']+'_'+label_data['activity']
                    label_title.append(label_key)
                    if label_key==activity_folder:
                        activity_folder_path = os.path.join(patient_folder_path, activity_folder)
                        # mat_data=np.zeros([target_length,21])
                        # Check if it's a directory and contains 'keypoints' folder
                        if os.path.isdir(activity_folder_path) and 'keypoints' in os.listdir(activity_folder_path):
                            keypoints_folder_path = os.path.join(activity_folder_path, 'keypoints')
                            # Check if 'keypoints' folder contains .mat files
                            if any(file.endswith('.mat') for file in os.listdir(keypoints_folder_path)):
                                # Iterate through .mat files
                                for file in os.listdir(keypoints_folder_path):
                                    if file.endswith('.mat'):
                                        mat_file_path = os.path.join(keypoints_folder_path, file)
                                        key=file.split('.')[0]
                                        if key==keys and activity_folder.split('_')[4]=='activity1' or activity_folder.split('_')[4]=='activity2' or activity_folder.split('_')[4]=='activity3'or activity_folder.split('_')[4]=='activity4' or activity_folder.split('_')[4]=='activity5' or activity_folder.split('_')[4]=='activity6':
                                            # Load the .mat file
                                            mat_data_ = scipy.io.loadmat(mat_file_path)
                                            if mat_data_[key].shape[0]<450:
                                                # for jj in range(21):
                                                original_length.append(mat_data_[key].shape[0])
                                                label_padded=np.zeros(mat_data_[key].shape[0])
                                                if len(label_data['sequence'])<mat_data_[key].shape[0]:
                                                    label_padded[0:len(label_data['sequence'])]=label_data['sequence']
                                                if len(label_data['sequence'])>mat_data_[key].shape[0]:
                                                    label_padded[0:len(label_data['sequence'])]=label_data['sequence'][0:mat_data_[key].shape[0]]
                                                mat_data = process_time_series(mat_data_[key][:,0,1], target_length)/norm_dimension
                                                lab_data = process_time_series(label_padded, target_length)


                                                mat_data[mat_data<0.05]=np.nan
                                                nan_count=np.count_nonzero(np.isnan(mat_data))
                                                if nan_count<0.2*mat_data_[key].shape[0]:#consider missing 10% of the values then do manual interpolation
                                                    data.append(mat_data)
                                                    data_label.append(lab_data)
                                                    # plt.figure(figsize=(12, 5))
                                                    # plt.plot(mat_data, color='blue', marker='o')
                                                    # plt.show()
                                                    # print('lala')
                                                test_data.append(mat_data)
                                                title.append(activity_folder)
                                                test_label.append(lab_data)


    # Now 'data' contains the data from all three .mat files from all patient and activity folders
    data=np.asarray(data)
    data_label=np.asarray(data_label)
    segments=[]
    seg_labels=[]
    ######################
    for k in range(data_label.shape[0]):
        observation=data_label[k]
        data_ob=data[k]
        # Calculate the differences between adjacent elements
        changes = np.diff(observation)
        # Find the indices where the values change
        change_indices = np.where(changes != 0)[0]
        labels=observation[change_indices-1]
        start=0
        for i in range (len(change_indices)):
            ind=int(change_indices[i])
            segments.append(data_ob[start:ind])
            start=ind
            seg_labels.append(labels[i])
        segments.append(data_ob[ind:])
        seg_labels.append(0.0)

    ######################################
    segments=np.asarray(segments)
    seg_labels=np.asarray(seg_labels)
    segments=segments[:, :, np.newaxis]
    test_data=np.asarray(test_data)
    test_data=test_data[:, :, np.newaxis]
    test_label=np.asarray(test_label)
    print(data.shape)
    # Define the dimensions of your data
    observations = data.shape[0]
    time_steps = data.shape[1]
    patch_length_min = 10
    patch_length_max = 50

    # Create a random mask for each observation
    masked_data = np.zeros((segments, time_steps, 1))

    for i in range(segments):
        # Randomly determine the patch length and location
        patch_length = np.random.randint(patch_length_min, patch_length_max + 1)
        start_time = np.random.randint(0, time_steps - patch_length + 1)

        # Create a mask for the current observation
        mask = np.ones((time_steps, 1))
        mask[start_time:start_time + patch_length] = np.nan

        # Apply the mask to the data
        masked_data[i, :, :] = mask

    # Apply the mask to your original data
    masked_data = masked_data * segments
    features = 1
    # X_intact, X, missing_mask, indicating_mask = mcar(data, 0.1) # hold out 10% observed values as ground truth
    # X = masked_fill(X, 1 - missing_mask, np.nan)
    # for i in range(masked_data.shape[0]):
    #     plt.figure(figsize=(12, 5))
    #     plt.plot(masked_data[i,:,0], color='blue', marker='o')
    #     plt.show()
    #     print('lala')
    dataset = {"X": masked_data,
               "Y":seg_labels}
    test_dataset={"X": test_data,
                  "Y":test_label}
    print(dataset["X"].shape)
    print(test_dataset["X"].shape)
    root="D:/Chicago_study/model_brit_"
    predicted_dataset_save=root+keys+"/"+"imputed.npy"
    test_dataset_save=root+keys+"/"+"test_set.npy"
    dataset_save=root+keys+"/"+"dataset.npy"
    saving_path=root+keys+"/"
    pdf_path=saving_path+'plots.pdf'
    # initialize the model
    brits = BRITS(
        n_steps=target_samples,
        n_features=features,
        n_classes=5,
        rnn_hidden_size=256,
        batch_size=32,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=10,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=3,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr=1e-3),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # Set it to None to use the default device (will use CPU if you don't have CUDA devices).
        # You can also set it to 'cpu' or 'cuda' explicitly, or ['cuda:0', 'cuda:1'] if you have multiple CUDA devices.
        device=[torch.device('cuda:0'),torch.device('cuda:1')],
        # set the path for saving tensorboard and trained model files
        saving_path=saving_path,
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )

    # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
    brits.fit(train_set=dataset, val_set=test_dataset)
    # # Assemble the datasets for training, validating, and testing.
    # dataset_for_training = {
    #     "X": X,
    # }
    # dataset_for_validating = {
    #     "X": X,
    #     "X_intact": X_intact,
    #     "indicating_mask": indicating_mask,
    # }
    # print(dataset_for_training["X"].shape)
    # print(dataset_for_validating["X"].shape)
    # # impute the originally-missing values and artificially-missing values
    #model_path="D:/Chicago_study/model_sait/20230926_T171553/SAITS.pypots"
    #saits.load_model("D:/Chicago_study/model_sait/20230922_T073128/SAITS.pypots")
# the testing stage, impute the originally-missing values and artificially-missing values in the test set
    brits_prediction = brits.classify(test_dataset)
    # Save the imputed data as a NumPy array
    np.save(predicted_dataset_save, brits_prediction)
    np.save(test_dataset_save, test_dataset['X'])
    np.save(dataset_save, dataset['X'])
    # Create a PdfPages object to save the plots to a PDF file
    pdf_pages = PdfPages(pdf_path)
    # Choose a specific sample (e.g., the first one)
    for i in range(50):#len(imputation)):
        imputed = brits_prediction[i][:,0]#plot wrist
        not_imputed = test_dataset['X'][i][:,0]#plot wrist
        if len(imputed)<original_length[i]:
            imputed=imputed[:original_length[i]]
            not_imputed=not_imputed[:original_length[i]]
        else:
            imputed= process_time_series(imputed, original_length[i])
            not_imputed= process_time_series(not_imputed, original_length[i])
        # Create a figure with two subplots: one for the data and one for the label
        # plt.figure(figsize=(12, 5))
        # plt.subplot(2,1,1)
        # plt.plot(imputed, color='blue', marker='o')
        # plt.title('imputed')
        # plt.subplot(2,1,2)
        # plt.plot(not_imputed, color='red', marker='o')
        # plt.title('not_imputed')
        # plt.tight_layout()
        # plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns for subplots
        
        # Customize the first subplot (imputed data)
        ax1.plot(imputed, color='blue', marker='o')
        ax1.set_title(f'Imputed Data - Figure {i}'+title[i])
        
        # Customize the second subplot (unimputed data)
        ax2.plot(not_imputed, color='red', marker='o')  # Example data for unimputed
        ax2.set_title(f'Unimputed Data - Figure {i}'+title[i])
        
        # Add the current figure to the PDF file
        pdf_pages.savefig(fig)
        
        # Close the figure to free up memory
        plt.close(fig)
    # Close the PDF file
    pdf_pages.close()
    # # calculate mean absolute error on the ground truth (artificially-missing values)
    # mae = cal_mae(imputation, X_intact, indicating_mask)
    # print(mae)
