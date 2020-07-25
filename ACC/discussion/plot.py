import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

lqr_patch = mpatches.Patch(color='#1f77b4', label='$X_{I}^{1}$')
ddpg_patch = mpatches.Patch(color='#ff7f0e', label='$X_{I}^{2}$')
our = mpatches.Patch(label='ours', color='#2ca02c')
lqr_tra = mpatches.Patch(label='$\kappa_{1}$', color='#d62728')
ddpg_tra = mpatches.Patch(label='$\kappa_{2}$', color='#17becf')

#构造等高线函数
def lqr(x,y):
    return -0.101250355358+1.00177427263e-19*((x-150)/35)+5.86530024765e-20*((y-40)/35)+0.00683286674384*((x-150)/35)**2+0.00391939937341*((y-40)/35)**2-0.011572914081*((x-150)/35)*((y-40)/35)-1.97707062055e-17*((x-150)/35)**3+3.94244541857e-17*((x-150)/35)**2*((y-40)/35)+1.80648426449e-17*((x-150)/35)*((y-40)/35)**2-6.12135478119e-17*((y-40)/35)**3+0.367995621602*((x-150)/35)**4-0.774738882654*((x-150)/35)**3*((y-40)/35)+1.4942682022*((x-150)/35)**2*((y-40)/35)**2-4.36323456588*((x-150)/35)*((y-40)/35)**3+3.55918109616*((y-40)/35)**4+2.37895789517e-16*((x-150)/35)**5-3.58563135086e-16*((x-150)/35)**4*((y-40)/35)-2.16589412638e-16*((x-150)/35)**3*((y-40)/35)**2-1.33749005667e-16*((x-150)/35)**2*((y-40)/35)**3+4.87950103187e-16*((x-150)/35)*((y-40)/35)**4+6.76683498752e-16*((y-40)/35)**5+2.17814689984*((x-150)/35)**6+1.84649125941*((x-150)/35)**5*((y-40)/35)-8.37926984615*((x-150)/35)**4*((y-40)/35)**2+14.2377989288*((x-150)/35)**3*((y-40)/35)**3-7.44716898832*((x-150)/35)**2*((y-40)/35)**4+38.9940823711*((x-150)/35)*((y-40)/35)**5-0.872096731721*((y-40)/35)**6-9.07163034185e-16*((x-150)/35)**7+1.14446209557e-15*((x-150)/35)**6*((y-40)/35)+2.57626120237e-16*((x-150)/35)**5*((y-40)/35)**2+6.33567846143e-16*((x-150)/35)**4*((y-40)/35)**3-5.26976186196e-16*((x-150)/35)**3*((y-40)/35)**4-6.0335868424e-16*((x-150)/35)**2*((y-40)/35)**5-3.06502708397e-15*((x-150)/35)*((y-40)/35)**6-2.38927390069e-15*((y-40)/35)**7-6.20737139181*((x-150)/35)**8+0.163391886811*((x-150)/35)**7*((y-40)/35)+13.7878414708*((x-150)/35)**6*((y-40)/35)**2-27.8618252553*((x-150)/35)**5*((y-40)/35)**3-6.81147164202*((x-150)/35)**4*((y-40)/35)**4-80.1514028899*((x-150)/35)**3*((y-40)/35)**5+32.0406334219*((x-150)/35)**2*((y-40)/35)**6-106.364871621*((x-150)/35)*((y-40)/35)**7-15.4562058765*((y-40)/35)**8+1.30134778396e-15*((x-150)/35)**9-1.47232144508e-15*((x-150)/35)**8*((y-40)/35)+4.72478988651e-16*((x-150)/35)**7*((y-40)/35)**2-7.87053209272e-16*((x-150)/35)**6*((y-40)/35)**3-9.05757497964e-17*((x-150)/35)**5*((y-40)/35)**4+1.25270386561e-15*((x-150)/35)**4*((y-40)/35)**5+4.26107838616e-15*((x-150)/35)**3*((y-40)/35)**6+1.62331898299e-15*((x-150)/35)**2*((y-40)/35)**7+5.05902261899e-15*((x-150)/35)*((y-40)/35)**8+3.27590164784e-15*((y-40)/35)**9+5.26304372026*((x-150)/35)**10-4.39142833134*((x-150)/35)**9*((y-40)/35)-2.39939795296*((x-150)/35)**8*((y-40)/35)**2+32.0526458001*((x-150)/35)**7*((y-40)/35)**3-5.02155271449*((x-150)/35)**6*((y-40)/35)**4+69.0779867462*((x-150)/35)**5*((y-40)/35)**5+26.3845687486*((x-150)/35)**4*((y-40)/35)**6+150.749934557*((x-150)/35)**3*((y-40)/35)**7-54.3544356261*((x-150)/35)**2*((y-40)/35)**8+116.060715157*((x-150)/35)*((y-40)/35)**9+23.8833716624*((y-40)/35)**10-6.1815707121e-16*((x-150)/35)**11+6.3321611038e-16*((x-150)/35)**10*((y-40)/35)-5.92682942084e-16*((x-150)/35)**9*((y-40)/35)**2+4.56060562451e-16*((x-150)/35)**8*((y-40)/35)**3-9.21747567929e-17*((x-150)/35)**7*((y-40)/35)**4-1.83973930265e-15*((x-150)/35)**6*((y-40)/35)**5-1.16195596296e-15*((x-150)/35)**5*((y-40)/35)**6-6.13879121735e-16*((x-150)/35)**4*((y-40)/35)**7-4.07629400952e-15*((x-150)/35)**3*((y-40)/35)**8-1.24063088042e-15*((x-150)/35)**2*((y-40)/35)**9-2.49308729012e-15*((x-150)/35)*((y-40)/35)**10-1.50112337177e-15*((y-40)/35)**11-1.23944919649*((x-150)/35)**12+3.20726105369*((x-150)/35)**11*((y-40)/35)-4.58355184385*((x-150)/35)**10*((y-40)/35)**2-12.4583239391*((x-150)/35)**9*((y-40)/35)**3+11.8621439579*((x-150)/35)**8*((y-40)/35)**4-34.7360911752*((x-150)/35)**7*((y-40)/35)**5-25.8116774954*((x-150)/35)**6*((y-40)/35)**6-51.2611660551*((x-150)/35)**5*((y-40)/35)**7+0.373357657989*((x-150)/35)**4*((y-40)/35)**8-89.0736484176*((x-150)/35)**3*((y-40)/35)**9+27.4388572548*((x-150)/35)**2*((y-40)/35)**10-44.2703218533*((x-150)/35)*((y-40)/35)**11-10.5248121796*((y-40)/35)**12
def DDPG(x, y):
	return -0.164804744113+0.107478509525*((x-150)/35)-0.10544250283*((y-40)/35)+0.315356856341*((x-150)/35)**2+0.162472135933*((x-150)/35)**3-0.639801333627*((x-150)/35)**4-0.334724976583*((x-150)/35)**5-0.489726577083*((x-150)/35)*((y-40)/35)-0.615482007843*((x-150)/35)**2*((y-40)/35)+0.036197780289*((x-150)/35)**3*((y-40)/35)+0.6662313844*((x-150)/35)**4*((y-40)/35)+0.537731411089*((x-150)/35)**5*((y-40)/35)+1.1086155283*((y-40)/35)**2+0.337894874898*((y-40)/35)**3-0.379778527356*((y-40)/35)**4+0.021847467964*((x-150)/35)*((y-40)/35)**2+1.01805334847*((x-150)/35)*((y-40)/35)**3-0.257401018895*((x-150)/35)*((y-40)/35)**4-0.266226418542*((y-40)/35)**5-0.0692817978617*((x-150)/35)**2*((y-40)/35)**2+0.616060834279*((x-150)/35)**2*((y-40)/35)**3+0.307705648212*((x-150)/35)**2*((y-40)/35)**4-0.153011763442*((x-150)/35)**3*((y-40)/35)**2-0.344488232012*((x-150)/35)**3*((y-40)/35)**3-0.830757179123*((x-150)/35)**4*((y-40)/35)**2-0.299109873499*((x-150)/35)*((y-40)/35)**5+0.807113178005*((x-150)/35)**6-0.0331804670883*((y-40)/35)**6


#作点
x=np.linspace(120,180,500)
y=np.linspace(25,55,500)

#构造网格
X,Y=np.meshgrid(x,y)

#绘制等高线,8表示等高线数量加1
plt.figure(figsize=(6, 4)) 
plt.contour(X,Y,lqr(X,Y),0,colors='#1f77b4')
plt.contour(X,Y,DDPG(X,Y),0,colors='#ff7f0e')
# plt.title('Inner-approximation of maximal robust invariant set')
plt.xlabel('s/m',fontsize=12.5)
plt.ylabel('v/(m/s)', fontsize=12.5)
# plt.legend(proxy, ["LQR"])
# # plt.legend()
# plt.savefig('ACC_Inner.pdf', bbox='tight')

switch = np.load('switch.npy')
lqr = np.load('lqr.npy')
ddpg = np.load('ddpg.npy')

plt.plot(switch[:, 0]+150, switch[:, 1]+40, label='ours', color='#2ca02c')
plt.plot(lqr[:, 0]+150, lqr[:, 1]+40, label='lqr', color='#d62728')
plt.plot(ddpg[:, 0]+150, ddpg[:, 1]+40, label='ddpg', color='#17becf')
# plt.legend()
plt.legend(handles=[lqr_patch, ddpg_patch, our, lqr_tra, ddpg_tra], fontsize=11)
plt.savefig('ACC_discuss.pdf', bbox='tight')