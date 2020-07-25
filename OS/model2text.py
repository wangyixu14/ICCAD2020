# from mpc2nn import ACC_nn
from Model import Actor
import torch
import numpy as np

trained_model = Actor(state_size=2, action_size=1, seed=0, fc1_units=25, fc2_units=100)
trained_model.load_state_dict(torch.load('actor_2900.pth'))
trained_model.eval()
bias_list = []
weight_list = []
for name, param in trained_model.named_parameters():
	if 'bias' in name:
		bias_list.append(param.detach().cpu().numpy())
		
	if 'weight' in name:
		weight_list.append(param.detach().cpu().numpy())

all_param = []

for i in range(len(bias_list)):
	for j in range(len(bias_list[i])):
		for k in range(weight_list[i].shape[1]):
			all_param.append(weight_list[i][j, k])
		all_param.append(bias_list[i][j])

# print(len(all_param),all_param[600:610], all_param[719:722])
np.savetxt('nn_actos2900_relu_tanh', np.array(all_param))
print('done')