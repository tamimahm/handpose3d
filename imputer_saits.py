from pypots.data.generating import gene_physionet2012
from pypots.utils.random import set_random_seed

set_random_seed()

# Load the PhysioNet-2012 dataset
physionet2012_dataset = gene_physionet2012(artificially_missing_rate=0.1)
 
# Take a look at the generated PhysioNet-2012 dataset, you'll find that everything has been prepared for you, 
# data splitting, normalization, additional artificially-missing values for evaluation, etc.
print(physionet2012_dataset.keys())
# Assemble the datasets for training, validating, and testing.

dataset_for_training = {
    "X": physionet2012_dataset['train_X'],
}

dataset_for_validating = {
    "X": physionet2012_dataset['val_X'],
    "X_intact": physionet2012_dataset['val_X_intact'],
    "indicating_mask": physionet2012_dataset['val_X_indicating_mask'],
}

dataset_for_testing = {
    "X": physionet2012_dataset['test_X'],
}
print(dataset_for_training["X"].shape)
print(dataset_for_validating["X"].shape)
# from pypots.optim import Adam
# from pypots.imputation import SAITS
# from pypots.utils.metrics import cal_mae

# # initialize the model
# saits = SAITS(
#     n_steps=physionet2012_dataset['n_steps'],
#     n_features=physionet2012_dataset['n_features'],
#     n_layers=2,
#     d_model=256,
#     d_inner=128,
#     n_heads=4,
#     d_k=64,
#     d_v=64,
#     dropout=0.1,
#     attn_dropout=0.1,
#     diagonal_attention_mask=True,  # otherwise the original self-attention mechanism will be applied
#     ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight
#     # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
#     MIT_weight=1,
#     batch_size=32,
#     # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
#     epochs=10,
#     # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
#     # You can leave it to defualt as None to disable early stopping.
#     patience=3,
#     # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
#     # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
#     optimizer=Adam(lr=1e-3),
#     # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
#     # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
#     # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
#     num_workers=0,
#     # Set it to None to use the default device (will use CPU if you don't have CUDA devices). 
#     # You can also set it to 'cpu' or 'cuda' explicitly, or ['cuda:0', 'cuda:1'] if you have multiple CUDA devices.
#     device=None,
#     # set the path for saving tensorboard and trained model files 
#     saving_path="tutorial_results/imputation/saits",
#     # only save the best model after training finished.
#     # You can also set it as "better" to save models performing better ever during training.
#     model_saving_strategy="best",
# )

# # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
# saits.fit(train_set=dataset_for_training, val_set=dataset_for_validating)

# # the testing stage, impute the originally-missing values and artificially-missing values in the test set
# saits_imputation = saits.impute(dataset_for_testing)

# # calculate mean absolute error on the ground truth (artificially-missing values)
# testing_mae = cal_mae(
#     saits_imputation, physionet2012_dataset['test_X_intact'], physionet2012_dataset['test_X_indicating_mask'])
# print("Testing mean absolute error: %.4f" % testing_mae)