import numpy as np
import matplotlib.pyplot as plt
import STZ
max_alpha = 0.2
max_beta =0.4
total_disp_steps = 349
for no_atoms in [60,270,446,608,666]:
    for iteration in [1,2,3,4]:
        d_alpha=np.linspace(-max_alpha,max_alpha,total_disp_steps+1)
        d_beta=np.linspace(0,max_beta, num=total_disp_steps+1)
        Beta,Alpha=np.meshgrid(d_beta,d_alpha)
        ns=np.loadtxt('Clusters/{}_cluster/cluster_{}/Normal_stress.txt'.format(no_atoms,iteration))
        initial=STZ.extract_dat('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(no_atoms,iteration,no_atoms),no_atoms)
        string = 'Normal_stress_0.0'
        #tau_o 
        normal_s = 0
        cs = plt.contour(Beta,Alpha,ns,normal_s)
        dat0=cs.allsegs[1][0]
        ss_lvl=np.zeros(len(dat0))
        s_strain=np.zeros(len(dat0))
        for j in range(len(dat0)):
            _,ss_lvl[j]=STZ.stress_config_grad(dat0[j][1],dat0[j][0],initial,no_atoms)
            s_strain[j]=dat0[j][0]
        plt.clf()
        plt.plot(s_strain,ss_lvl)
        # plt.plot(max_yeild_strain[i],max_shear_lvl[i], marker='x', markersize=10, color='r' )
        plt.title(' Normalised Normal stress= 0')
        plt.xlabel('Shear strain')
        plt.ylabel('Shear stress (N/m\u00b2)')
        plt.savefig('Clusters/{}_cluster/cluster_{}/{}.png'.format(no_atoms,iteration,string), facecolor='white', transparent=False)
        plt.clf()
        shear_lvl_data=np.column_stack((s_strain,ss_lvl))
        np.savetxt('Clusters/{}_cluster/cluster_{}/{}.txt'.format(no_atoms,iteration,string),shear_lvl_data)
