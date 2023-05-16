import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate, stats
from scipy.optimize import curve_fit
import STZ
####                                                                           Polynomial Approximation
# def test_4(x,a,b,c,d,e):
#     return a*x**4+b*x**3+c*x**2+d*x+e
# def test_3(x,a,b,c,d,):
#     return a*x**3+b*x**2+c*x+d
max_alpha=0.2
max_beta=0.4
total_disp_steps = 349
lower = -0.9375
upper =  1.5
steps = 14
for no_atoms in [446]:
    for iteration in [1]:                                                                               
        bs_change = STZ.yield_strain_bs(max_alpha,max_beta,total_disp_steps,no_atoms,iteration,3)
        upper_l = bs_change+0.01
        upper_l_neg = bs_change+0.05          # value added depends upon difference in strain in potential yield points (smaller sizes less value -> not much shift in yield point with normal stress )
        upper_l_pos = bs_change+0.01
        d_alpha=np.linspace(-max_alpha,max_alpha,total_disp_steps+1)
        d_beta=np.linspace(0,max_beta, num=total_disp_steps+1)
        Beta,Alpha=np.meshgrid(d_beta,d_alpha)
        ns=np.loadtxt('Clusters/{}_cluster/cluster_{}/Normal_stress.txt'.format(no_atoms,iteration))
        initial=STZ.extract_dat('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(no_atoms,iteration,no_atoms),no_atoms)

        #tau_o 
        normal_s = 0
        cs = plt.contour(Beta,Alpha,ns,normal_s)
        dat0=cs.allsegs[1][0]
        ss_lvl=np.zeros(len(dat0))
        s_strain=np.zeros(len(dat0))

        for j in range(len(dat0)):
            _,ss_lvl[j]=STZ.stress_config_grad(dat0[j][1],dat0[j][0],initial,no_atoms)
            s_strain[j]=dat0[j][0]

        ####                                                                                  Maximum 
            # strains_useful = s_strain[s_strain<upper_l]
            # y_interp = interpolate.interp1d(s_strain,ss_lvl)
            # tau_o_zero = y_interp(strains_useful).max()

        ####                                                                             First Strain drop
        # stress_diff = np.zeros(len(ss_lvl))
        # for i in range(len(ss_lvl)-1):
        #     stress_diff[i] = ss_lvl[i+1]-ss_lvl[i]
        # drop_strains = s_strain[stress_diff<0]
        # y_strain = drop_strains[0]
        # y_interp = interpolate.interp1d(s_strain,ss_lvl)
        # tau_o_zero = y_interp(y_strain)

        
        ####                                                                           Polynomial Approximation

        # param,param_cov = curve_fit(test_4,s_strain,ss_lvl)
        # x_o = np.linspace(s_strain.min(),s_strain.max(),100)
        # y_o = param[0]*x_o**4+param[1]*x_o**3+param[2]*x_o**2+param[3]*x_o+param[4]  #test_4
        # y_o = param[0]*x_o**3+param[1]*x_o**2+param[2]*x_o+param[3]  #test_3
        # tau_o_zero = y_o.max()


        ####                                                                          Stress drop in plastic zone 

        stress_diff = np.zeros(len(ss_lvl))
        for i in range(len(ss_lvl)-1):
            stress_diff[i] = ss_lvl[i+1]-ss_lvl[i]
        drop_strains = s_strain[stress_diff<0]
        drop_strains_useful = drop_strains[drop_strains<upper_l]
        if(len(drop_strains_useful)==0):
            stress_useful = ss_lvl[s_strain<upper_l]
            tau_o_zero = stress_useful.max()
        else:
            strains_useful = s_strain[s_strain<upper_l]
            y_interp = interpolate.interp1d(s_strain,ss_lvl)
            tau_o_zero = y_interp(strains_useful).max()


        factors_ns=np.linspace(lower,upper,steps)
        normal_s = factors_ns*tau_o_zero
        max_shear_lvl = np.zeros(len(normal_s))
        max_yeild_strain = np.zeros(len(normal_s))
        string='Normal_stress_'
        cs = plt.contour(Beta,Alpha,ns,normal_s)
        for  i in range(0,len(normal_s)):
            dat0=cs.allsegs[i][0]
            ss_lvl=np.zeros(len(dat0))
            s_strain=np.zeros(len(dat0)) 
            ns=np.zeros([len(normal_s),len(dat0)])
            for j in range(len(dat0)):
                _,ss_lvl[j]=STZ.stress_config_grad(dat0[j][1],dat0[j][0],initial,no_atoms)
                s_strain[j]=dat0[j][0]
            string=string+str(round(normal_s[i]/tau_o_zero,4)) 
            ss_data = np.column_stack((s_strain, ss_lvl))
        
        ####                                                                                  Maximum 
            # if(normal_s[i]>=0):
            #     upper_l_corr = upper_l_pos
            # else:
            #     upper_l_corr = upper_l_neg

            # stress_useful = ss_lvl[s_strain<upper_l_corr]
            # max_shear_lvl[i] = stress_useful.max()
            # strain_arg = stress_useful.argmax()
            # max_yeild_strain[i] = s_strain[strain_arg]
        ####                                                                           Polynomial Approximation
            # param, param_cov = curve_fit(test_4,s_strain,ss_lvl)
            # x_o = np.linspace(s_strain.min(),s_strain.max(),100)
            # y_o = param[0]*x_o**4+param[1]*x_o**3+param[2]*x_o**2+param[3]*x_o+param[4]   #test_4
            # # y_o = param[0]*x_o**3+param[1]*x_o**2+param[2]*x_o+param[3]       test_3
            # max_shear_lvl[i] = y_o.max()
            # max_yeild_strain[i] = x_o[y_o.argmax()]

        ####                                                                             First Strain drop
            # stress_diff = np.zeros(len(ss_lvl))
            # for p in range(len(ss_lvl)-1):
            #     stress_diff[p] = ss_lvl[p+1]-ss_lvl[p]
            # drop_strains = s_strain[stress_diff<0]
            # y_strain = drop_strains[0]
            # y_interp = interpolate.interp1d(s_strain,ss_lvl)
            # max_shear_lvl[i] = y_interp(y_strain)
            # max_yeild_strain[i] = y_strain
        
        


        ####                                                                          Stress drop in plastic zone                                                                             
            if(normal_s[i]>=0):
                upper_l_corr = upper_l_pos
            else:
                upper_l_corr = upper_l_neg

            stress_diff = np.zeros(len(ss_lvl))
            for p in range(len(ss_lvl)-1):
                stress_diff[p] = ss_lvl[p+1]-ss_lvl[p]
            drop_strains = s_strain[stress_diff<0]
            drop_strains_useful = drop_strains[drop_strains<upper_l_corr]
            if(len(drop_strains_useful)==0):
                stress_useful = ss_lvl[s_strain<upper_l_corr]
                max_shear_lvl[i] = stress_useful.max()
                strain_arg = stress_useful.argmax()
                max_yeild_strain[i] = s_strain[strain_arg]
            else:
                y_interp = interpolate.interp1d(s_strain,ss_lvl)
                max_shear_lvl[i] = y_interp(drop_strains_useful).max()
                index = y_interp(drop_strains_useful).argmax()
                max_yeild_strain[i] = drop_strains[index]



            ####                                                          Plotting
            plt.figure(figsize=[6,4], dpi=300)
            plt.title('{} atoms STZ at normalised normal stress = {}'.format(no_atoms,round(factors_ns[i],4)))
            plt.plot(s_strain,ss_lvl/10**9)
            # plt.plot(x_o,y_o/10**9,'r-')                 #Polynomial Approximation
            plt.plot(max_yeild_strain[i],max_shear_lvl[i]/10**9, marker='x', markersize=10, color='r' )
            plt.xlabel('Shear strain')
            plt.ylabel('Shear Stress (GPa)')
            np.savetxt('Clusters/{}_cluster/cluster_{}/{}.txt'.format(no_atoms,iteration,string),ss_data)
            plt.savefig('Clusters/{}_cluster/cluster_{}/{}.png'.format(no_atoms,iteration,string), facecolor='white', transparent=False)
            plt.clf()
            string='Normal_stress_'

        ####                                                              MC plot
        
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
        MC_data=np.column_stack((normal_s,max_shear_lvl,max_yeild_strain))
        plt.clf()
        np.savetxt('Clusters/{}_cluster/cluster_{}/MC_data'.format(no_atoms,iteration),MC_data)

