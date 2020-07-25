import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  
  
X = [0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 8.5, 9.5]  
# ACC 
Y = [100, 98.8, 97, 99.6, 99.6, 97.4, 94.6, 99, 0]  
# OS
#Y = [100, 85.6, 92.2, 89.2, 99.2, 83, 89, 85.4, 0]  
#fig = plt.figure()  
plt.bar(X[0], Y[0], 1, label='ours', color='#1f77b4')
plt.bar(X[4], Y[4], 1, color='#1f77b4')
plt.bar(X[1], Y[1], 1, label='$\kappa_{lqr}$', color='#ff7f0e')
plt.bar(X[5], Y[5], 1, color='#ff7f0e')  

plt.bar(X[2], Y[2], 1, label='$\kappe_{ddpg}$', color='#2ca02c')
plt.bar(X[6], Y[6], 1, color='#2ca02c')
plt.bar(X[3], Y[3], 1, label='random', color='#d62728')
plt.bar(X[7], Y[7], 1, color='#d62728') 

# plt.bar(X[8], Y[8], 2,  color='white')
# plt.bar(XX,YY,1,color="yellow")  #使用不同颜色  
# plt.xlabel("X-axis")  #设置X轴Y轴名称  
plt.xticks([])
plt.ylabel("Safely control rate(%)")
plt.ylim((90, 100))  
# plt.title("Scaling disturbance in oscillator")
plt.legend(loc='upper right')
#使用text显示数值  
for a,b in zip(X,Y):  
	print(a, b)
	if b != 0:
		plt.text(a, b+0.05, '%.1f' % b, ha='center', va= 'bottom',fontsize=11)  
# for a,b in zip(XX,YY):  
# 	plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)    
# plt.ylim(0,37)    #设置Y轴上下限  
# plt.text(2, 0, 'Twice', ha='center', fontsize=12)
# plt.text(7, 0, 'Fourth', ha='center', fontsize=12)
plt.savefig('ACC_scaling.pdf', bbox='tight')