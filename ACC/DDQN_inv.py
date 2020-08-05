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
from env import ACC
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

model_1 = Actor(state_size=2, action_size=1, seed=0, fc1_units=20).to(device)
model_1.load_state_dict(torch.load("./smooth_0.5/model_1400.pth"))
model_1.eval()

# def lqr(state):
# 	u = 1.6139 * state[0] - 2.3036 * state[1] + 8
# 	return u/20
def lqr(state):
	u = 1.934 * state[0] - 2.6622 * state[1] + 8
	return u/20
# model_2 = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
# model_2.load_state_dict(torch.load("actor_m2_2800_good.pth"))
# model_2.eval()

# single_model = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
# single_model.load_state_dict(torch.load("actor_single_2400.pth"))
# print(model_1, model_2, single_model)

invariant_1 = io.loadmat('invrl_5_5_300.mat')['V_new']
invariant_2 = io.loadmat('invlqr_2_5_300.mat')['V_new']
# print(invariant_2)
# assert False
# single_inv = io.loadmat('data/single.mat')['V']
#SDP
def where_inv(state):
	x = state[0] + 150 
	y = state[1] + 40	 
	u_lqr = -0.101250355358+1.00177427263e-19*((x-150)/35)+5.86530024765e-20*((y-40)/35)+0.00683286674384*((x-150)/35)**2+0.00391939937341*((y-40)/35)**2-0.011572914081*((x-150)/35)*((y-40)/35)-1.97707062055e-17*((x-150)/35)**3+3.94244541857e-17*((x-150)/35)**2*((y-40)/35)+1.80648426449e-17*((x-150)/35)*((y-40)/35)**2-6.12135478119e-17*((y-40)/35)**3+0.367995621602*((x-150)/35)**4-0.774738882654*((x-150)/35)**3*((y-40)/35)+1.4942682022*((x-150)/35)**2*((y-40)/35)**2-4.36323456588*((x-150)/35)*((y-40)/35)**3+3.55918109616*((y-40)/35)**4+2.37895789517e-16*((x-150)/35)**5-3.58563135086e-16*((x-150)/35)**4*((y-40)/35)-2.16589412638e-16*((x-150)/35)**3*((y-40)/35)**2-1.33749005667e-16*((x-150)/35)**2*((y-40)/35)**3+4.87950103187e-16*((x-150)/35)*((y-40)/35)**4+6.76683498752e-16*((y-40)/35)**5+2.17814689984*((x-150)/35)**6+1.84649125941*((x-150)/35)**5*((y-40)/35)-8.37926984615*((x-150)/35)**4*((y-40)/35)**2+14.2377989288*((x-150)/35)**3*((y-40)/35)**3-7.44716898832*((x-150)/35)**2*((y-40)/35)**4+38.9940823711*((x-150)/35)*((y-40)/35)**5-0.872096731721*((y-40)/35)**6-9.07163034185e-16*((x-150)/35)**7+1.14446209557e-15*((x-150)/35)**6*((y-40)/35)+2.57626120237e-16*((x-150)/35)**5*((y-40)/35)**2+6.33567846143e-16*((x-150)/35)**4*((y-40)/35)**3-5.26976186196e-16*((x-150)/35)**3*((y-40)/35)**4-6.0335868424e-16*((x-150)/35)**2*((y-40)/35)**5-3.06502708397e-15*((x-150)/35)*((y-40)/35)**6-2.38927390069e-15*((y-40)/35)**7-6.20737139181*((x-150)/35)**8+0.163391886811*((x-150)/35)**7*((y-40)/35)+13.7878414708*((x-150)/35)**6*((y-40)/35)**2-27.8618252553*((x-150)/35)**5*((y-40)/35)**3-6.81147164202*((x-150)/35)**4*((y-40)/35)**4-80.1514028899*((x-150)/35)**3*((y-40)/35)**5+32.0406334219*((x-150)/35)**2*((y-40)/35)**6-106.364871621*((x-150)/35)*((y-40)/35)**7-15.4562058765*((y-40)/35)**8+1.30134778396e-15*((x-150)/35)**9-1.47232144508e-15*((x-150)/35)**8*((y-40)/35)+4.72478988651e-16*((x-150)/35)**7*((y-40)/35)**2-7.87053209272e-16*((x-150)/35)**6*((y-40)/35)**3-9.05757497964e-17*((x-150)/35)**5*((y-40)/35)**4+1.25270386561e-15*((x-150)/35)**4*((y-40)/35)**5+4.26107838616e-15*((x-150)/35)**3*((y-40)/35)**6+1.62331898299e-15*((x-150)/35)**2*((y-40)/35)**7+5.05902261899e-15*((x-150)/35)*((y-40)/35)**8+3.27590164784e-15*((y-40)/35)**9+5.26304372026*((x-150)/35)**10-4.39142833134*((x-150)/35)**9*((y-40)/35)-2.39939795296*((x-150)/35)**8*((y-40)/35)**2+32.0526458001*((x-150)/35)**7*((y-40)/35)**3-5.02155271449*((x-150)/35)**6*((y-40)/35)**4+69.0779867462*((x-150)/35)**5*((y-40)/35)**5+26.3845687486*((x-150)/35)**4*((y-40)/35)**6+150.749934557*((x-150)/35)**3*((y-40)/35)**7-54.3544356261*((x-150)/35)**2*((y-40)/35)**8+116.060715157*((x-150)/35)*((y-40)/35)**9+23.8833716624*((y-40)/35)**10-6.1815707121e-16*((x-150)/35)**11+6.3321611038e-16*((x-150)/35)**10*((y-40)/35)-5.92682942084e-16*((x-150)/35)**9*((y-40)/35)**2+4.56060562451e-16*((x-150)/35)**8*((y-40)/35)**3-9.21747567929e-17*((x-150)/35)**7*((y-40)/35)**4-1.83973930265e-15*((x-150)/35)**6*((y-40)/35)**5-1.16195596296e-15*((x-150)/35)**5*((y-40)/35)**6-6.13879121735e-16*((x-150)/35)**4*((y-40)/35)**7-4.07629400952e-15*((x-150)/35)**3*((y-40)/35)**8-1.24063088042e-15*((x-150)/35)**2*((y-40)/35)**9-2.49308729012e-15*((x-150)/35)*((y-40)/35)**10-1.50112337177e-15*((y-40)/35)**11-1.23944919649*((x-150)/35)**12+3.20726105369*((x-150)/35)**11*((y-40)/35)-4.58355184385*((x-150)/35)**10*((y-40)/35)**2-12.4583239391*((x-150)/35)**9*((y-40)/35)**3+11.8621439579*((x-150)/35)**8*((y-40)/35)**4-34.7360911752*((x-150)/35)**7*((y-40)/35)**5-25.8116774954*((x-150)/35)**6*((y-40)/35)**6-51.2611660551*((x-150)/35)**5*((y-40)/35)**7+0.373357657989*((x-150)/35)**4*((y-40)/35)**8-89.0736484176*((x-150)/35)**3*((y-40)/35)**9+27.4388572548*((x-150)/35)**2*((y-40)/35)**10-44.2703218533*((x-150)/35)*((y-40)/35)**11-10.5248121796*((y-40)/35)**12
	u_rl = -0.164804744113+0.107478509525*((x-150)/35)-0.10544250283*((y-40)/35)+0.315356856341*((x-150)/35)**2+0.162472135933*((x-150)/35)**3-0.639801333627*((x-150)/35)**4-0.334724976583*((x-150)/35)**5-0.489726577083*((x-150)/35)*((y-40)/35)-0.615482007843*((x-150)/35)**2*((y-40)/35)+0.036197780289*((x-150)/35)**3*((y-40)/35)+0.6662313844*((x-150)/35)**4*((y-40)/35)+0.537731411089*((x-150)/35)**5*((y-40)/35)+1.1086155283*((y-40)/35)**2+0.337894874898*((y-40)/35)**3-0.379778527356*((y-40)/35)**4+0.021847467964*((x-150)/35)*((y-40)/35)**2+1.01805334847*((x-150)/35)*((y-40)/35)**3-0.257401018895*((x-150)/35)*((y-40)/35)**4-0.266226418542*((y-40)/35)**5-0.0692817978617*((x-150)/35)**2*((y-40)/35)**2+0.616060834279*((x-150)/35)**2*((y-40)/35)**3+0.307705648212*((x-150)/35)**2*((y-40)/35)**4-0.153011763442*((x-150)/35)**3*((y-40)/35)**2-0.344488232012*((x-150)/35)**3*((y-40)/35)**3-0.830757179123*((x-150)/35)**4*((y-40)/35)**2-0.299109873499*((x-150)/35)*((y-40)/35)**5+0.807113178005*((x-150)/35)**6-0.0331804670883*((y-40)/35)**6
	# print(u_rl, u_lqr)
	return np.array([int(u_rl<=-0.015), int(u_lqr<=0)])

# value based
# def where_inv(state):
	 
# 	x_loc = state[0] + 150 
# 	y_loc = state[1] + 40
# 	x1 = np.linspace(115, 185, 300)
# 	y1 = np.linspace(20, 60, 300)
# 	inv1 = interp2d(x1, y1, invariant_1, kind='cubic')(x_loc, y_loc)
	
# 	x2 = np.linspace(118, 182, 300)
# 	y2 = np.linspace(20, 60, 300)
# 	inv2 = interp2d(x2, y2, invariant_2, kind='cubic')(x_loc, y_loc)
# 	# print(inv1, inv2)
# 	return np.array([int(inv1<1e-8 and inv1 >= 0), int(inv2<1e-8 and inv2 >= 0)])

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
			# state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
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
	env = ACC()
	model = DQN(2, 2).to(device)
	target_model = DQN(2, 2).to(device)
	optimizer = optim.Adam(model.parameters())
	EP_NUM = 2001
	frame_idx = 0
	fuel_list = []
	ep_reward = deque(maxlen=100)

	for ep in range(EP_NUM):
		state = env.reset()
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
					control_action = lqr(state.cpu().data.numpy())
			next_state, reward, done, dis = env.step(control_action)	
			
			reward += 25
			reward -= weight * abs(control_action) * 20
			if done and t < 90:
				reward -= 100
			if flag[action] == 0:
				reward -= 50
			
			fuel_list.append(abs(control_action) * 20)
			replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
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
		if ep >= 200 and ep % 100 == 0:
			torch.save(model.state_dict(), './ddqn_models/inv/ddqn_'+str(ep)+'_'+str(weight)+'.pth')

def find_feasible(flag, action):
	indi = False
	if flag[action] == 1:
		return action, indi
	else:
		feasible = np.where(flag>0)[0]
		if len(feasible) == 0:
			indi = True
			# action = np.random.randint(len(flag))
		else:
			index = np.random.randint(len(feasible))
			action = feasible[index]
		return action, indi

def test(model_name, state_list=None, renew=False, mode='switch'):
	env = ACC()
	model = DQN(2, 2).to(device)
	EP_NUM = 500
	if mode == 'switch':
		model.load_state_dict(torch.load(model_name))
	if renew:
		state_list = []
	fuel_list = []
	ep_reward = []
	history = []
	for ep in range(EP_NUM):
		if renew:
			while True:
				state = env.reset()
				if np.sum(where_inv(state)) != 0:
					break
			state_list.append(state)
		else:
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0] + 150, state_list[ep][1] + 40)
		ep_r = 0
		fuel = 0
		action_last = np.random.randint(2)
		for t in range(env.max_iteration):
			state = torch.from_numpy(state).float().to(device)
			history.append(state.cpu().numpy())
			if mode == 'switch':
				flag = where_inv(state.cpu().numpy())
				action = model.act(state, epsilon=0)
				action, _ = find_feasible(flag, action)
				
				with torch.no_grad():
					if action == 0:
						control_action = model_1(state).cpu().data.numpy()[0]
					elif action == 1:
						control_action = lqr(state.cpu().data.numpy())
				if ep == 0:
					print(t, state, action, control_action*20)
				
			elif mode == 'ddpg':
				control_action = model_1(state).cpu().data.numpy()[0]
			elif mode =='lqr':
				control_action = lqr(state.cpu().data.numpy())
			elif mode == 'random':
				action = np.random.randint(2)
				if action == 0:
					control_action = model_1(state).cpu().data.numpy()[0]
				elif action == 1:
					control_action = lqr(state.cpu().data.numpy())

			next_state, reward, done, _ = env.step(control_action)
			fuel += abs(control_action) * 20
			state = next_state
			ep_r += reward
			if done:
				break
		ep_reward.append(ep_r)
		if t > 95:
			fuel_list.append(fuel)
		else:
			print(ep, state_list[ep])
		# np.save('./discussion/' + mode + '.npy', np.array(history))	
	return ep_reward, np.array(fuel_list), state_list

if __name__ == '__main__':

	sw_state = np.load('init_state_500.npy')
	sw_reward, sw_fuel, sw_state = test('./ddqn_models/0417/ddqn_400_1.0.pth', state_list=sw_state, renew=False, mode='switch')
	print('ddpg')
	m1_reward, m1_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='ddpg')
	print('lqr')
	m2_reward, m2_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='lqr')
	ran_reward, ran_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='random')
	print(np.mean(sw_fuel),np.mean(m1_fuel), np.mean(m2_fuel), np.mean(ran_fuel), 
			len(sw_fuel),len(m1_fuel), len(m2_fuel), len(ran_fuel))

