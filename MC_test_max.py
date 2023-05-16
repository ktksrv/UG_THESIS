import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate, stats
max_alpha=0.2
max_beta=0.4
total_disp_steps=349
lower = -1.5
upper =  1.5
steps = 17
no_atoms = 270
iteration = 2
right_limit = 0.3
string='Normal_stress_'
factors_ns=np.linspace(lower,upper,steps)
max_shear_lvl = np.zeros(len(factors_ns))
max_yeild_strain = np.zeros(len(factors_ns))

stress_strain = np.loadtxt('Clusters/{}_cluster/cluster_{}/Normal_stress_0.0.txt'.format(no_atoms,iteration))
tau_o_zero = stress_strain[:,1].max()
for i in range(len(factors_ns)):
    string=string+str(factors_ns[i])
    stress_strain = np.loadtxt('Clusters/{}_cluster/cluster_{}/{}.txt'.format(no_atoms,iteration,string))
    stress_strain_useful = stress_strain[stress_strain[:,0]<right_limit]
    max_shear_lvl[i] = stress_strain_useful[:,1].max()
    max_yeild_strain[i] = stress_strain[stress_strain_useful[:,1].argmax(),0]  
    plt.figure(figsize=[6,4], dpi=300)
    plt.title('{} atoms STZ at normalised normal stress = {}'.format(no_atoms,round(factors_ns[i],4)))
    plt.plot(stress_strain[:,0],stress_strain[:,1]/10**9)
    plt.plot(max_yeild_strain[i],max_shear_lvl[i]/10**9, marker='x', markersize=10, color='r' )
    plt.xlabel('Shear strain')
    plt.ylabel('Shear Stress (GPa)')
    plt.savefig('Clusters/{}_cluster/cluster_{}/{}.png'.format(no_atoms,iteration,string), facecolor='white', transparent=False)
    plt.clf()
    string='Normal_stress_'
plt.figure(figsize=[6,4], dpi=300)
plt.scatter(factors_ns, max_shear_lvl/tau_o_zero, marker='.',s=150)
plt.title(' MC Test (Linear fit)')
plt.xlabel("Normalised normal stress")
plt.ylabel("Normalised shear stress")
slope, intercept, r, _, _ = stats.linregress(factors_ns, max_shear_lvl/tau_o_zero)
x=np.linspace(lower,upper,10)
y=slope*x+intercept
plt.plot(x,y,'r')
plt.text(lower,1,'y={}x+{}'.format(round(slope,5),round(intercept,5)))
plt.text(lower,0.9,'R\u00b2={}'.format(r**2))
plt.savefig('Clusters/{}_cluster/cluster_{}/{}_cluster_MC.png'.format(no_atoms,iteration,no_atoms), facecolor='white', transparent=False)
plt.clf()




























































# # factors=np.linspace(-1.5,1.5,41)
# tolerance=-4e+12                                      #determine from Yeild Criteria
# tau_o=7080433757.509869
# factors=np.linspace(-1.5,1.25,12)
# normal_s = factors*tau_o
# string='Normal_stress_'
# max_shear_lvl = np.zeros(len(normal_s))
# max_yeild_strain = np.zeros(len(normal_s))
# max_alpha=0.12
# max_beta=0.4
# total_disp_steps=49
# d_alpha=np.linspace(-max_alpha,max_alpha,total_disp_steps+1)
# d_beta=np.linspace(0,max_beta, num=total_disp_steps+1)
# Beta,Alpha=np.meshgrid(d_beta,d_alpha)
# no_atoms=408
# iteration=1
# ns=np.loadtxt('Clusters/{}_cluster/cluster_{}/Normal_stress.txt'.format(no_atoms,iteration))
# initial=STZ.extract_dat('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(no_atoms,iteration,no_atoms),no_atoms)
# cs = plt.contour(Beta,Alpha,ns,normal_s)
# for  i in range(0,len(normal_s)):
#     dat0=cs.allsegs[i][0]
#     ss_lvl=np.zeros(len(dat0))
#     s_strain=np.zeros(len(dat0)) 

#     for j in range(len(dat0)):
#         ns,ss_lvl[j]=STZ.stress_config_grad(dat0[j][1],dat0[j][0],initial,no_atoms,iteration)
#         s_strain[j]=dat0[j][0]
#     x_vals = np.linspace(0,0.4,100)
#     splines=interpolate.splrep(s_strain,ss_lvl)
#     y_vals_stress=interpolate.splev(x_vals, splines)
#     y_vals=interpolate.splev(x_vals, splines, der=2)
#     index = np.where(y_vals < tolerance)
#     max_shear_lvl[i]=y_vals_stress[index[0][0]]
#     max_yeild_strain[i]=x_vals[index[0][0]]
#     shear_lvl_data=np.column_stack((ss_lvl,s_strain))
#     string=string+str(round(normal_s[i]/tau_o,3))   
#     np.savetxt('Clusters/{}_cluster/cluster_{}/{}.txt'.format(no_atoms,iteration,string),shear_lvl_data)
#     plt.plot(x_vals,y_vals_stress)
#     plt.plot(max_yeild_strain[i],max_shear_lvl[i], marker='x', markersize=10, color='r' )
#     plt.title(' Normalised Normal stress={}'.format(round(factors[i],3)))
#     plt.xlabel('Shear strain')
#     plt.ylabel('Shear stress (N/m\u00b2)')
#     plt.savefig('Clusters/{}_cluster/cluster_{}/{}.png'.format(no_atoms,iteration,string), facecolor='white', transparent=False)
#     plt.clf()
#     string='Normal_stress_'
# plt.figure(figsize=[6,4], dpi=300)
# plt.scatter(factors, max_shear_lvl/tau_o, marker='.',s=150)
# plt.title(' MC Test (Linear fit)')
# plt.xlabel("Normalised normal stress")
# plt.ylabel("Normalised shear stress")
# slope, intercept, r, _, _ = stats.linregress(factors, max_shear_lvl/tau_o)
# x=np.linspace(-1.5,1.5,10)
# y=slope*x+intercept
# plt.plot(x,y,'r')
# plt.text(-1.25,1,'y={}x+{}'.format(round(slope,5),round(intercept,5)))
# plt.text(-1.25,0.9,'R\u00b2={}'.format(r**2))
# plt.savefig('Clusters/{}_cluster/cluster_{}/{}_cluster_MC.png'.format(no_atoms,iteration,no_atoms), facecolor='white', transparent=False)
# MC_data=np.column_stack((normal_s,max_shear_lvl,max_yeild_strain))
# np.savetxt('Clusters/{}_cluster/cluster_{}/MC_data'.format(no_atoms,iteration),MC_data)