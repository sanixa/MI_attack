import torch

state_dict = torch.load("model/Target_dataset_cifar_10_ep_-1_nm_1.1_epoch_30_param_0_dataset_custom_True_model_type_cnn_1618455243.1593597_57.4_0.5479/model_0.5.pt")
torch.save(state_dict, "model/Target_dataset_cifar_10_ep_-1_nm_1.1_epoch_30_param_0_dataset_custom_True_model_type_cnn_1618455243.1593597_57.4_0.5479/model_0.5.pt", _use_new_zipfile_serialization=False)