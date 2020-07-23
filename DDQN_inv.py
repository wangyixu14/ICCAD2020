import numpy as np
import torch
import torch.nn as nn
from Model import Actor, DQN_Actor
import time
import torch.optim as optim
import random
from collections import deque, namedtuple
from torch.autograd import Variable
import math
from torch.utils.tensorboard import SummaryWriter
import sys
from env import Osillator
import scipy.io as io
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import gym
import torch.autograd as autograd
from interval import Interval

weight = float(sys.argv[2])
print(weight)

class ReplayBuffer(object):
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)
	
	def push(self, state, action, reward, next_state, done):
		state      = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)
			
		self.buffer.append((state, action, reward, next_state, done))
	
	def sample(self, batch_size):
		state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
		return np.concatenate(state), action, reward, np.concatenate(next_state), done
	
	def __len__(self):
		return len(self.buffer)

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 3000
replay_buffer = ReplayBuffer(int(5e3))
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_1 = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
model_1.load_state_dict(torch.load("actor_2800.pth"))
model_1.eval()

model_2 = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
model_2.load_state_dict(torch.load("actor_2900.pth"))
model_2.eval()

invariant_1 = io.loadmat('inv_os2800.mat')['V1']
invariant_2 = io.loadmat('inv_os2900.mat')['V']
# 150 * 150 size
# def where_inv(state):	 
# 	x_loc = state[0]
# 	y_loc = state[1]
# 	x1 = np.linspace(-2.5, 2.5, 150)
# 	y1 = np.linspace(-2.5, 2.5, 150)
# 	inv1 = interp2d(x1, y1, invariant_1, kind='linear')(x_loc, y_loc)
	
# 	x2 = np.linspace(-2.5, 2.5, 150)
# 	y2 = np.linspace(-2.5, 2.5, 150)
# 	inv2 = interp2d(x2, y2, invariant_2, kind='linear')(x_loc, y_loc)
# 	# print(inv1, inv2)
# 	return np.array([int(inv1<1e-8), int(inv2<1e-8)])

def where_inv(state):
	x = state[0]
	y = state[1]
	inv1 = -0.191853371186-0.000536911697763*(x/3)-0.000435109277384*(y/3)+0.0182690460991*(x/3)**2+0.0247561665106*(x/3)**2*(y/3)-0.346538735833*(x/3)**3+0.0131520860539*(x/3)*(y/3)+1.53816421231*(x/3)**3*(y/3)+0.00483596382816*(y/3)**2+0.0028560139637*(x/3)*(y/3)**2+0.00414190902842*(y/3)**3+2.57054493409*(x/3)**2*(y/3)**2-0.390195854557*(x/3)**2*(y/3)**3-0.0185636017707*(x/3)**3*(y/3)**2+0.516490714902*(x/3)*(y/3)**3-7.11604377572*(x/3)**3*(y/3)**3+12.9396462908*(x/3)**4-0.121917887256*(y/3)**4-1.44310891792*(x/3)**5-0.477116884587*(x/3)**4*(y/3)-0.589388746102*(x/3)*(y/3)**4+0.216986128166*(y/3)**5-48.4278049612*(x/3)**6-7.34926485315*(x/3)**5*(y/3)-12.097946216*(x/3)**4*(y/3)**2-8.70246548175*(x/3)**2*(y/3)**4-0.798247329098*(x/3)*(y/3)**5+3.83825312635*(y/3)**6+5.30760297016*(x/3)**7+1.23827036181*(x/3)**6*(y/3)+3.05363915294*(x/3)**5*(y/3)**2-1.11810870233*(x/3)**4*(y/3)**3-4.22132859866*(x/3)**3*(y/3)**4+0.113604823788*(x/3)**2*(y/3)**5+2.96611977829*(x/3)*(y/3)**6-0.653435021444*(y/3)**7+64.8081899032*(x/3)**8+11.2567183203*(x/3)**7*(y/3)+20.4555166423*(x/3)**6*(y/3)**2+14.3628217064*(x/3)**5*(y/3)**3+27.6045432791*(x/3)**4*(y/3)**4+16.9262714487*(x/3)**3*(y/3)**5-8.4495658036*(x/3)**2*(y/3)**6-2.08341959261*(x/3)*(y/3)**7-2.33364382283*(y/3)**8-3.51997118773*(x/3)**9-0.776054346523*(x/3)**8*(y/3)-3.80690425671*(x/3)**7*(y/3)**2+2.17563895289*(x/3)**6*(y/3)**3+1.98252147664*(x/3)**5*(y/3)**4+0.232152626381*(x/3)**4*(y/3)**5+4.41563574444*(x/3)**3*(y/3)**6+0.997067584801*(x/3)**2*(y/3)**7-2.63841615246*(x/3)*(y/3)**8+0.453270940035*(y/3)**9-28.7177708963*(x/3)**10-5.55260617771*(x/3)**9*(y/3)-14.1070353745*(x/3)**8*(y/3)**2-7.50127458683*(x/3)**7*(y/3)**3-11.4211866286*(x/3)**6*(y/3)**4-19.7078126731*(x/3)**5*(y/3)**5-12.9837016603*(x/3)**4*(y/3)**6-6.84664730917*(x/3)**3*(y/3)**7+13.2110716019*(x/3)**2*(y/3)**8+2.02869323441*(x/3)*(y/3)**9-0.748164306403*(y/3)**10
	inv2 = -0.287964114549-0.0699479731997*(x/3)-0.0131359115208*(y/3)+1.90551199836*(x/3)**2-0.438226702423*(x/3)**2*(y/3)+0.217913775564*(x/3)**3+0.355063635107*(x/3)**3*(y/3)+0.11529327211*(y/3)**2+0.37061953383*(x/3)**3*(y/3)**2-0.132170885371*(x/3)**3*(y/3)**3-0.0304111524299*(y/3)**3-4.58502468741*(x/3)**2*(y/3)**2+1.69857223625*(x/3)**2*(y/3)**3+0.0899390854696*(x/3)*(y/3)-0.192905556144*(x/3)*(y/3)**2+0.793275425178*(x/3)*(y/3)**3-6.04638297848*(x/3)**4-0.192980459015*(x/3)**5-1.37203470804*(x/3)**5*(y/3)+1.62601309598*(y/3)**4-0.431514701438*(x/3)**5*(y/3)**2-0.848370545701*(x/3)**5*(y/3)**3+0.149606452217*(y/3)**5+1.06847840813*(x/3)**4*(y/3)+13.4647119834*(x/3)**4*(y/3)**2-2.0203241137*(x/3)**4*(y/3)**3-8.90595369713*(x/3)**4*(y/3)**4-0.808958564731*(x/3)**3*(y/3)**4+0.761107798672*(x/3)**3*(y/3)**5+3.62576055234*(x/3)**2*(y/3)**4-1.32740942552*(x/3)**2*(y/3)**5+1.62920067052*(x/3)*(y/3)**4-1.43659854523*(x/3)*(y/3)**5+9.80986280992*(x/3)**6-0.85777499264*(y/3)**6+0.0473578939388*(x/3)**7-0.5583676141*(x/3)**6*(y/3)-1.77001862447*(x/3)*(y/3)**6+0.0037625898052*(y/3)**7-4.94232892363*(x/3)**8+0.955048128205*(x/3)**7*(y/3)-9.61197034361*(x/3)**6*(y/3)**2-1.10721810889*(x/3)**2*(y/3)**6+0.00525501602542*(x/3)*(y/3)**7-6.61546755522e-06*(y/3)**8
	return np.array([int(inv1<=0), int(inv2<=0)])

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

class DQN(nn.Module):
	def __init__(self, num_inputs, num_actions):
		super(DQN, self).__init__()
		self.num_actions = num_actions
		self.layers = nn.Sequential(
			nn.Linear(num_inputs, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, num_actions)
		)
		
	def forward(self, x):
		return self.layers(x)
	
	def act(self, state, epsilon):
		if random.random() > epsilon:
			q_value = self.forward(state)
			action  = q_value.max(0)[1].item()
		else:
			action = random.randrange(self.num_actions)
		return action

def compute_td_loss(model, target_model, batch_size, optimizer):
	state, action, reward, next_state, done = replay_buffer.sample(batch_size)

	state      = Variable(torch.FloatTensor(np.float32(state)))
	next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
	action     = Variable(torch.LongTensor(action))
	reward     = Variable(torch.FloatTensor(reward))
	done       = Variable(torch.FloatTensor(done))

	q_values      = model(state)
	next_q_values = model(next_state)
	next_q_state_values = target_model(next_state) 

	q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

	next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
	expected_q_value = reward + gamma * next_q_value * (1 - done)
	
	loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
		
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	return loss

def train():
	env = Osillator()
	model = DQN(2, 2).to(device)
	target_model = DQN(2, 2).to(device)
	optimizer = optim.Adam(model.parameters())
	EP_NUM = 2001
	frame_idx = 0
	fuel_list = []
	ep_reward = deque(maxlen=100)

	for ep in range(EP_NUM):
		state = env.reset()
		# while True:
		# 	state = env.reset()
		# 	if np.sum(where_inv(state)) != 0:
		# 		break
		ep_r = 0
		for t in range(200):
			state = torch.from_numpy(state).float().to(device)
			flag = where_inv(state.cpu().data.numpy())
			epsilon = epsilon_by_frame(frame_idx)
			action = model.act(state, epsilon)
			with torch.no_grad():
				if action == 0:
					control_action = model_1(state).cpu().data.numpy()[0]
				elif action == 1:
					control_action = model_2(state).cpu().data.numpy()[0]
			next_state, reward, done = env.step(control_action)
			reward = 2
			reward -= weight * abs(control_action) * 20
			if done and t <190:
				reward -= 100
			if flag[action] == 0:
				reward -= 20
			if flag[action] != 0:
				replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
			fuel_list.append(abs(control_action) * 20)
			state = next_state
			ep_r += reward
			frame_idx += 1
			if len(replay_buffer) > batch_size:
				loss = compute_td_loss(model, target_model, batch_size, optimizer)
			if frame_idx % 100 == 0:
				update_target(model, target_model)
			if done:
				break
		ep_reward.append(ep_r)
		print('epoch:', ep, 'reward:', ep_r, 'average reward:', np.mean(ep_reward),
					 'fuel cost:', sum(fuel_list[-t - 1:]), 'epsilon:', epsilon, len(replay_buffer)) 
		if ep >= 100 and ep % 100 == 0:
			torch.save(model.state_dict(), './ddqn_models/0428/ddqn_'+str(ep)+'_'+str(weight)+'.pth')

def find_feasible(flag, action):
	indi = False
	if flag[action] == 1:
		return action, indi
	else:
		feasible = np.where(flag>0)[0]
		if len(feasible) == 0:
			indi = True
		else:
			index = np.random.randint(len(feasible))
			action = feasible[index]
		return action, indi

def test(model_name, state_list=None, renew=False, mode='switch'):
	env = Osillator()
	model = DQN(2, 2).to(device)
	EP_NUM = 500
	if mode == 'switch':
		model.load_state_dict(torch.load(model_name))
	if renew:
		state_list = []
	fuel_list = []
	ep_reward = []
	epoch_list = []

	for ep in range(EP_NUM):
		if renew:
			while True:
				state = env.reset()
				if np.sum(where_inv(state)) != 0:
					break
			state_list.append(state)
		else:
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1])
		ep_r = 0
		fuel = 0
		action_last = np.random.randint(2)
		for t in range(env.max_iteration):
			state = torch.from_numpy(state).float().to(device)
			
			if mode == 'switch':
				flag = where_inv(state.cpu().numpy())
				action = model.act(state, epsilon=0)
				action, _ = find_feasible(flag, action)
				with torch.no_grad():
					if action == 0:
						control_action = model_1(state).cpu().data.numpy()[0]
					elif action == 1:
						control_action = model_2(state).cpu().data.numpy()[0]
				if ep == 0:
					print(t, state, action)
			
			elif mode == 'random':
				action = np.random.randint(2)
				if action == 0:
					control_action = model_1(state).cpu().data.numpy()[0]
				elif action == 1:
					control_action = model_2(state).cpu().data.numpy()[0]
			
			elif mode == 'd1':
				control_action = model_1(state).cpu().data.numpy()[0]
		
			elif mode == 'd2':
				control_action = model_2(state).cpu().data.numpy()[0]
			epoch_list.append(state.cpu().numpy())
			next_state, reward, done = env.step(control_action)
			fuel += abs(control_action) * 20
			state = next_state
			ep_r += reward
			if done:
				break
	
		ep_reward.append(ep_r)
		if t >= 190:
			fuel_list.append(fuel)
		else:
			print(ep, t, state_list[ep])
	# np.save('./discussion/' + mode + '.npy', np.array(epoch_list))
	return ep_reward, np.array(fuel_list), state_list

if __name__ == '__main__':
	# print(where_inv(np.array([1.96691097, 1.28230389])))
	# assert False
	# train()
	# assert False

	sw_reward, sw_fuel, sw_state  = test('./ddqn_models/inv/ddqn_200_1.0_good.pth', state_list=None, renew=True, mode='switch')
	print('d1')
	d1_reward, d1_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='d1')
	ran_reward, ran_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='random')
	d2_reward, d2_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='d2')
	print(np.mean(sw_fuel), np.mean(d1_fuel), np.mean(d2_fuel), np.mean(ran_fuel), 
		len(sw_fuel), len(d1_fuel), len(d2_fuel), len(ran_fuel))
	# np.save('init_state.npy', np.array(sw_state))

	# sw_reward, sw_fuel, sw_state  = test('./ddqn_models/inv/ddqn_200_1.0_good.pth', state_list=None, renew=True, mode='switch')
	# print(len(sw_fuel), np.mean(sw_fuel))
	# np.save('init_state.npy', np.array(sw_state))
	# assert False
	
	# r1_used_plot
	# sw_state = [np.load('init_state_500.npy')[6]]
	# r2_used_plot
	# sw_state = np.load('init_state_500.npy')
	# print(len(sw_state), sw_state[0])
	# sw_state = [[1, 1]]
	# sw_reward, sw_fuel, sw_state  = test('./ddqn_models/inv/ddqn_200_1.0_good.pth', state_list=sw_state, renew=False, mode='switch')
	# # print('d1')
	# d1_reward, d1_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='d1')
	# print('d2')
	# d2_reward, d2_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='d2')
	# print('ran')
	# ran_reward, ran_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='random')
	# print(len(sw_fuel), len(d1_fuel), len(d2_fuel), len(ran_fuel), 
	# 	np.mean(sw_fuel), np.mean(d1_fuel), np.mean(d2_fuel), np.mean(ran_fuel))