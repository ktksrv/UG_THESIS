#Function to extract coordinates from .dat file
from numpy import transpose
from sympy import convex_hull


def extract_dat(path,no_atoms):
                                                              
    import pandas as pd
    import numpy as np
    df=pd.read_csv(path,sep=" ",skiprows=8, nrows=no_atoms, header=None)
    df.columns=['ID', 'type', 'X','Y','Z']
    ID=np.linspace(1,no_atoms,no_atoms)
    x=np.array(df.X)
    y=np.array(df.Y)
    z=np.array(df.Z)
    out=np.column_stack([ID,x,y,z])

    return out

def Simulation_box_dim(path):
    import pandas as pd
    import numpy as np
    df=pd.read_csv(path,sep=" ",skiprows=3, nrows=3, header=None)
    df.columns=['lo', 'hi', 'label_1','label_2',]
    axis_lo=np.array(df.lo)
    axis_hi=np.array(df.hi)
    return axis_hi, axis_lo
    
def extract_dat(path,no_atoms):
                                                              
    import pandas as pd
    import numpy as np
    df=pd.read_csv(path,sep=" ",skiprows=8, nrows=no_atoms, header=None)
    df.columns=['ID', 'type', 'X','Y','Z']
    ID=np.linspace(1,no_atoms,no_atoms)
    x=np.array(df.X)
    y=np.array(df.Y)
    z=np.array(df.Z)
    out=np.column_stack([ID,x,y,z])

    return out
    
def extract_dump(path,no_atoms):
    import pandas as pd
    import numpy as np
    df=pd.read_csv(path,sep=" ",skiprows=9, nrows=no_atoms, header=None)
    df.columns=['ID', 'type', 'X','Y','Z']
    #df.drop(df.columns[[-1,-2]], axis=1, inplace=True)
    ID=np.linspace(1,no_atoms,no_atoms)
    x=np.array(df.X)
    y=np.array(df.Y)
    z=np.array(df.Z)
    out=np.column_stack([ID,x,y,z])
    return out

#Function to extract ID of surface atoms
def surface_atoms(initial,no_atoms):
    from ovito.data import Particles, DataCollection, SimulationCell
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import ConstructSurfaceModifier
    import numpy as np
    particles = Particles()
    data = DataCollection()
    data.objects.append(particles)
    cell = SimulationCell(pbc = (False, False, False))
    cell[...] = [[200,0,0,0],                               #use cell vectors for given congifuration from ovito
                [0,72.56,0,0],
                [0,0,145.12,0]]
    cell.vis.line_width = 0.1
    data.objects.append(cell)
    pos_prop = particles.create_property('Position', data=initial)
    id=range(1,no_atoms+1)
    pipeline = Pipeline(source = StaticSource(data = data))
    pipeline.add_to_scene()
    pipeline.modifiers.append(ConstructSurfaceModifier(method = ConstructSurfaceModifier.Method.AlphaShape, radius=2.55, select_surface_particles=True))
    data= pipeline.compute()
    selection=np.array(data.particles["Selection"])
    surface_ID_all=selection*id
    surface_ID=surface_ID_all[surface_ID_all !=0]
    
    return  surface_ID

#Function to classify upper and lower groups
def slice_atoms(initial):
    import numpy as np
    z_mean=initial[:,3].mean()
    upper=np.zeros(len(initial[:,3]))
    lower=np.zeros(len(initial[:,3]))

    upper_i=np.zeros([len(initial[:,2]),3])
    lower_i=np.zeros([len(initial[:,2]),3])
    for i in range(len(initial[:,3])):
        if initial[i,3]>z_mean:
            upper[i]=i+1
            upper_i[i,:]=initial[i,1:]
        else:
            lower[i]=i+1
            lower_i[i,:]=initial[i,1:]
    upper_id=upper[upper!=0]
    lower_id=lower[lower!=0]
    upper_i=upper_i[~np.all(upper_i==0,axis=1)]
    lower_i=lower_i[~np.all(lower_i==0,axis=1)]

    return upper_id,lower_id,upper_i,lower_i

def extract_fdump(path,no_atoms):
    import pandas as pd
    import numpy as np
    df=pd.read_csv(path,sep=" ",skiprows=9, nrows=no_atoms, header=None)
    df.columns=['ID', 'type', 'X','Y','Z', 'F_x', 'F_y', 'F_z']
    #df.drop(df.columns[[-1,-2]], axis=1, inplace=True)
    x=np.array(df.X)
    y=np.array(df.Y)
    z=np.array(df.Z)
    fx=np.array(df.F_x)
    fy=np.array(df.F_y)
    fz=np.array(df.F_z)

    out_c=np.column_stack([x,y,z])                                 
    out_f=np.column_stack([fx,fy,fz]) 

    return out_c, out_f

def stress_calc(coordinates,force):
    import numpy as np
    from scipy.spatial import ConvexHull
    sigma=np.zeros([3,3])
    for i in range(len(force)):
        F = force[i,:]
        F=F.reshape(3,1)
        R = coordinates[i,:]
        sigma = sigma + np.kron(F,R)
    hull= ConvexHull(coordinates) 
    volume=hull.volume
    sigma=-sigma/volume
    sigma= sigma*1.6021766208e+11   #eV/ang^3  to SI
    return sigma[2,2], sigma[0,2]

def stress_config(total_disp_x,total_disp_z,total_disp_steps,no_atoms):
    from lammps import lammps
    import STZ
    initial=STZ.extract_dat('L_100_STZ.dat',no_atoms)
    upper_id,_,_,_,=STZ.slice_atoms(initial)

    for j in  range(total_disp_steps):
        disp_x=total_disp_x/total_disp_steps
        disp_z=total_disp_z/total_disp_steps
        for m in upper_id:
            initial[int(m)-1,3]+=disp_z
            initial[int(m)-1,1]+=disp_x
        lmp=lammps()
        lmp_ovito=lammps()
        lmp_ovito.file('1_3.in')
        create_atoms=['create_atoms 1 single {} {} {}'.format(initial[l,1],initial[l,2],initial[l,3]) for l in range(len(initial[:,0]))]
        lmp_ovito.commands_list(create_atoms)
        lmp_ovito.command('dump dump_1 all custom 1 dump.ovito id type x y z')
        lmp_ovito.command('run 1')
        surface_id=STZ.surface_atoms('dump.ovito',no_atoms)
        lmp.file('1_3.in')
        lmp.commands_list(create_atoms)
        surface_id_str=surface_id.astype('str')
        surface_str=" "
        for m in range(len(surface_id)):
            surface_str=surface_str+surface_id_str[m]
            surface_str+=" "
        surface_group=['group surface id {}'.format(surface_str)]
        lmp.commands_list(surface_group)
        minimization_block='''
        group not_surface subtract all surface
        fix 1 surface setforce 0 0 0 
        minimize 1e-40 1e-40 10000 10000
        unfix 1
        '''
        lmp.commands_string(minimization_block)
        lmp.command('dump dump_1 all custom 1 dump.minimized id type x y z')
        lmp.command('compute force all property/atom fx fy fz')
        lmp.command('dump fcal all custom 1 dump.force id type x y z fx fy fz')    
        lmp.command('run 1')

        initial=STZ.extract_dump('dump.minimized',no_atoms)
        upper_id,_,_,_,=STZ.slice_atoms(initial)
    #lmp.command('minimize 1e-40 1e-40 10000 10000')
    coordinates,force=STZ.extract_fdump('dump.force',no_atoms)
    ns,ss=STZ.stress_calc(coordinates,force)
    return ns, ss


def stress_config_grad(alpha,beta,initial,no_atoms):
    import numpy as np
    import STZ
    from lammps import lammps
    deformation_grad=np.array([[1,0,beta],[0,1,0],[0,0,1+alpha]])
    d_initial=np.zeros([no_atoms,3])
    for k in range(no_atoms):
            d_initial[k,:]=np.matmul(deformation_grad,initial[k,1:])
    surface_id=STZ.surface_atoms(d_initial,no_atoms)
    surface_id_str=surface_id.astype('str')
    surface_str=" "
    for r in range(len(surface_id)):
        surface_str=surface_str+surface_id_str[r]
        surface_str+=" "     
    lmp=lammps()
    initialization_block='''
    dimension 3
    units metal
    boundary s s s
    atom_style atomic
    timestep 0.001
    region myregion block 0.0 200.0 0.0 72.56 0.0 145.12  units box
    create_box  1 myregion
    mass 1 63.546

    pair_style eam
    pair_coeff 1 1 Cu_u6.eam '''
    lmp.commands_string(initialization_block)
    create_atoms=['create_atoms 1 single {} {} {}'.format(d_initial[l,0],d_initial[l,1],d_initial[l,2]) for l in range(len(initial[:,0]))]
    lmp.commands_list(create_atoms)
    surface_group=['group surface id {}'.format(surface_str)]
    lmp.commands_list(surface_group)
    
    minimization_block='''
    fix freeze surface setforce 0 0 0 
    minimize 0 1e-4 100000 100000
    unfix freeze
    '''
    lmp.commands_string(minimization_block)
    lmp.command('compute force all property/atom fx fy fz')
    lmp.command('dump fcal all custom 1 dump.force id type x y z fx fy fz')    
    lmp.command('run 1')
    coordinates,force=STZ.extract_fdump('dump.force',no_atoms)
    ns,ss=STZ.stress_calc(coordinates,force)

    return ns, ss

def random_cluster_centroid(x_ul,x_ll,y_ul,y_ll,z_ul,z_ll):
    import numpy as np
    x= x_ll+(x_ul-x_ll)*np.random.random()
    y= y_ll+(y_ul-x_ll)*np.random.random()
    z= z_ll+(z_ul-z_ll)*np.random.random()
    return x, y, z


def random_cluster_generator(size,iteration):

    from ovito.data import Particles, DataCollection, SimulationCell
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier
    from lammps import lammps
    from ovito.io import export_file
    import numpy as np
    import STZ

    # Random Cluster of given size
    current_count_diff=10
    no_atoms=32800                                   #initial BMG size and parameters which define the usable region for cluster extraction
    if size>0 and size<500 :
        r_initial=80
    elif size>500 and size<1000 :
        r_initial=170
    elif size>1000 and size<1500 :
        r_initial=240
    r_updated=r_initial
    x_rand,y_rand,z_rand= STZ.random_cluster_centroid(54,20,54,20,90,54)
    initial=STZ.extract_dat('Initial_BMG.dat',no_atoms)
    iterations=0
    particles = Particles()
    data = DataCollection()
    data.objects.append(particles)
    cell = SimulationCell(pbc = (False, False, False))
    cell[...] = [[72.56,0,0,0],                               #use cell vectors for given congifuration from ovito
                [0,72.56,0,0],
                [0,0,145.12,0]]
    cell.vis.line_width = 0.1
    data.objects.append(cell)
    pos_prop = particles.create_property('Position', data=initial[:,1:])
    pipeline = Pipeline(source = StaticSource(data = data))
    pipeline.add_to_scene()
    while (current_count_diff!=0):
        if iterations==1000:
            x_rand,y_rand,z_rand= STZ.random_cluster_centroid(54,20,54,20,90,54)
            iterations=0
        pipeline.modifiers.append(ExpressionSelectionModifier(expression = '(Position.X-{})^2+(Position.Y-{})^2+(Position.Z-{})^2 <{}'.format(x_rand,y_rand,z_rand,r_updated)))
        data= pipeline.compute()
        current_count=data.attributes['ExpressionSelection.count']
        current_count_diff=size-current_count
        del pipeline.modifiers[0]
        iterations+=1
        r_updated= r_updated+0.01*current_count_diff

    pipeline.modifiers.append(ExpressionSelectionModifier(expression = '(Position.X-{})^2+(Position.Y-{})^2+(Position.Z-{})^2 > {}'.format(x_rand,y_rand,z_rand,r_updated)))
    data = pipeline.compute()
    count=data.attributes['ExpressionSelection.count']
    pipeline.modifiers.append(DeleteSelectedModifier())
    export_file(pipeline, "Clusters/{}_cluster/cluster_{}/cluster_{}.dat".format(size,iteration,size), "lammps/data")

    # Minimize the cluster
    lmp=lammps()
    minimization_block='''
    dimension 3
    units metal
    boundary s s s
    atom_style atomic
    timestep 0.001

    read_data Clusters/{}_cluster/cluster_{}/cluster_{}.dat
    mass 1 63.546
    pair_style eam
    pair_coeff 1 1 Cu_u6.eam
    minimize 0 1e-6 100000 100000


    dump dump_1 all custom 1 Clusters/{}_cluster/cluster_{}/cluster_{}.dat id type x y z

    run 1

    '''.format(size,iteration,size,size,iteration,size)
    lmp.commands_string(minimization_block)

    # Affine Transformations
    initial=STZ.extract_dump('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(size,iteration,size),size)
    particles = Particles()
    data = DataCollection()
    data.objects.append(particles)
    cell = SimulationCell(pbc = (False, False, False))
    cell[...] = [[200,0,0,0],                               #Transformed Simulation box, which should contain all the sheared atoms
                [0,72.56,0,0],
                [0,0,145.12,0]]
    cell.vis.line_width = 0.1
    data.objects.append(cell)
    pos_prop = particles.create_property('Position', data=initial[:,1:])
    pipeline = Pipeline(source = StaticSource(data = data))
    pipeline.add_to_scene()
    export_file(pipeline, "Clusters/{}_cluster/cluster_{}/cluster_{}.dat".format(size,iteration,size), "lammps/data")



def Normal_Stress_Matrix(no_atoms,max_alpha,max_beta,total_disp_steps,iteration):
    from lammps import lammps
    import matplotlib.pyplot as plt
    import numpy as np
    import STZ
    from scipy import interpolate, stats 
    ## NORMAL STRESS MATRIX GENERATION ##
    d_alpha=np.linspace(-max_alpha,max_alpha,total_disp_steps+1)
    d_beta=np.linspace(0,max_beta, num=total_disp_steps+1)
    ns=np.zeros([len(d_alpha),len(d_beta)])
    Beta,Alpha=np.meshgrid(d_beta,d_alpha)
    initial=STZ.extract_dat('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(no_atoms,iteration,no_atoms),no_atoms)
    y=-1
    for i in d_alpha:
        alpha=i
        y=y+1
        for j in range(0,total_disp_steps+1):
            beta=(max_beta/total_disp_steps)*j
            deformation_grad=np.array([[1,0,beta],[0,1,0],[0,0,1+alpha]])
            d_initial=np.zeros([no_atoms,3])
            for k in range(no_atoms):
                d_initial[k,:]=np.matmul(deformation_grad,initial[k,1:])
            surface_id=STZ.surface_atoms(d_initial,no_atoms)
            surface_id_str=surface_id.astype('str')
            surface_str=" "
            for r in range(len(surface_id)):
                surface_str=surface_str+surface_id_str[r]
                surface_str+=" "     
            lmp=lammps()
            initialization_block='''
            dimension 3
            units metal
            boundary s s s
            atom_style atomic
            timestep 0.001
            region myregion block 0.0 200.0 0.0 72.56 0.0 145.12  units box
            create_box  1 myregion
            mass 1 63.546

            pair_style eam
            pair_coeff 1 1 Cu_u6.eam
            '''
            lmp.commands_string(initialization_block)
            create_atoms=['create_atoms 1 single {} {} {}'.format(d_initial[l,0],d_initial[l,1],d_initial[l,2]) for l in range(len(initial[:,0]))]
            lmp.commands_list(create_atoms)
            surface_group=['group surface id {}'.format(surface_str)]
            lmp.commands_list(surface_group)
            minimization_block='''
            fix freeze surface setforce 0 0 0 
            minimize 0 1e-4 100000 100000
            unfix freeze
            '''
            lmp.commands_string(minimization_block)
            lmp.command('compute force all property/atom fx fy fz')
            lmp.command('dump fcal all custom 1 dump.force id type x y z fx fy fz')    
            lmp.command('run 1')
            coordinates,force=STZ.extract_fdump('dump.force',no_atoms)
            ns[y,j],_=STZ.stress_calc(coordinates,force)
        np.savetxt('Clusters/{}_cluster/cluster_{}/Normal_stress.txt'.format(no_atoms,iteration),ns)

def MC_test(tolerance,max_alpha,max_beta,total_disp_steps,no_atoms,iteration,lower,upper,steps):

    import numpy as np
    import matplotlib.pyplot as plt
    import STZ
    from scipy import interpolate, stats 

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
    

    # for j in range(len(dat0)):
    #     _,ss_lvl[j]=STZ.stress_config_grad(dat0[j][1],dat0[j][0],initial,no_atoms)
    #     s_strain[j]=dat0[j][0]
    
    # x_vals = np.linspace(0,max_beta,100)
    # splines=interpolate.splrep(s_strain,ss_lvl)
    # y_vals_stress=interpolate.splev(x_vals, splines)
    # y_vals=interpolate.splev(x_vals, splines, der=2)
    # index = np.where(y_vals < tolerance)
    # tau_o=y_vals_stress[index[0][0]]
    
    tau_o = 9.931594491454593658e+09
    #Multiples of tau_o
    factors=np.linspace(lower,upper,steps)
    normal_s = factors*tau_o
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
            ns[i,j],ss_lvl[j]=STZ.stress_config_grad(dat0[j][1],dat0[j][0],initial,no_atoms)
            s_strain[j]=dat0[j][0]
        # x_vals = np.linspace(0,max_beta,100)
        # splines=interpolate.splrep(s_strain,ss_lvl)
        # y_vals_stress=interpolate.splev(x_vals, splines)
        # y_vals=interpolate.splev(x_vals, splines, der=2)
        # index = np.where(y_vals < tolerance)
        # max_shear_lvl[i]=y_vals_stress[index[0][0]]
        # max_yeild_strain[i]=x_vals[index[0][0]]
        shear_lvl_data=np.column_stack((ss_lvl,s_strain))
        string=string+str(round(normal_s[i]/tau_o,3))   
        np.savetxt('Clusters/{}_cluster/cluster_{}/{}.txt'.format(no_atoms,iteration,string),shear_lvl_data)
        # np.savetxt('Clusters/{}_cluster/cluster_{}/ns_{}.txt'.format(no_atoms,iteration,string),ns)
        # plt.plot(x_vals,y_vals_stress)
        # plt.plot(max_yeild_strain[i],max_shear_lvl[i], marker='x', markersize=10, color='r' )
        # plt.title(' Normalised Normal stress={}'.format(round(factors[i],3)))
        # plt.xlabel('Shear strain')
        # plt.ylabel('Shear stress (N/m\u00b2)')
        # plt.savefig('Clusters/{}_cluster/cluster_{}/{}.png'.format(no_atoms,iteration,string), facecolor='white', transparent=False)
        # plt.clf()
        string='Normal_stress_'
    # plt.figure(figsize=[6,4], dpi=300)
    # plt.scatter(factors, max_shear_lvl/tau_o, marker='.',s=150)
    # plt.title(' MC Test (Linear fit)')
    # plt.xlabel("Normalised normal stress")
    # plt.ylabel("Normalised shear stress")
    # slope, intercept, r, _, _ = stats.linregress(factors, max_shear_lvl/tau_o)
    # x=np.linspace(lower,upper,10)
    # y=slope*x+intercept
    # plt.plot(x,y,'r')
    # plt.text(lower,1,'y={}x+{}'.format(round(slope,5),round(intercept,5)))
    # plt.text(lower,0.9,'R\u00b2={}'.format(r**2))
    # plt.savefig('Clusters/{}_cluster/cluster_{}/{}_cluster_MC.png'.format(no_atoms,iteration,no_atoms), facecolor='white', transparent=False)
    # MC_data=np.column_stack((normal_s,max_shear_lvl,max_yeild_strain))
    plt.clf()
    # np.savetxt('Clusters/{}_cluster/cluster_{}/MC_data'.format(no_atoms,iteration),MC_data)
    


def convex_hull_relaxed_config(alpha,beta,initial,no_atoms):
    import numpy as np
    import STZ
    from lammps import lammps
    from scipy.spatial import ConvexHull
    deformation_grad=np.array([[1,0,beta],[0,1,0],[0,0,1+alpha]])
    d_initial=np.zeros([no_atoms,3])
    for k in range(no_atoms):
            d_initial[k,:]=np.matmul(deformation_grad,initial[k,1:])
    surface_id=STZ.surface_atoms(d_initial,no_atoms)
    surface_id_str=surface_id.astype('str')
    surface_str=" "
    for r in range(len(surface_id)):
        surface_str=surface_str+surface_id_str[r]
        surface_str+=" "     
    lmp=lammps()
    initialization_block='''
    dimension 3
    units metal
    boundary s s s
    atom_style atomic
    timestep 0.001
    region myregion block 0.0 200.0 0.0 72.56 0.0 145.12  units box
    create_box  1 myregion
    mass 1 63.546

    pair_style eam
    pair_coeff 1 1 Cu_u6.eam '''
    lmp.commands_string(initialization_block)
    create_atoms=['create_atoms 1 single {} {} {}'.format(d_initial[l,0],d_initial[l,1],d_initial[l,2]) for l in range(len(initial[:,0]))]
    lmp.commands_list(create_atoms)
    surface_group=['group surface id {}'.format(surface_str)]
    lmp.commands_list(surface_group)
    
    minimization_block='''
    fix freeze surface setforce 0 0 0 
    minimize 0 1e-4 100000 100000
    unfix freeze
    '''
    lmp.commands_string(minimization_block)
    lmp.command('compute force all property/atom fx fy fz')
    lmp.command('dump fcal all custom 1 dump.force id type x y z fx fy fz')    
    lmp.command('run 1')
    coordinates,force=STZ.extract_fdump('dump.force',no_atoms)
    hull= ConvexHull(coordinates) 
    volume=hull.volume

    return volume


def Convex_hull__volume_zero_ns(max_alpha,max_beta,total_disp_steps,no_atoms,iteration):

    import numpy as np
    import matplotlib.pyplot as plt
    import STZ
    from scipy import interpolate, stats 

    d_alpha=np.linspace(-max_alpha,max_alpha,total_disp_steps+1)
    d_beta=np.linspace(0,max_beta, num=total_disp_steps+1)
    Beta,Alpha=np.meshgrid(d_beta,d_alpha)
    ns=np.loadtxt('Clusters/{}_cluster/cluster_{}/Normal_stress.txt'.format(no_atoms,iteration))
    initial=STZ.extract_dat('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(no_atoms,iteration,no_atoms),no_atoms)
    
    #tau_o 
    normal_s = 0
    cs = plt.contour(Beta,Alpha,ns,normal_s)
    dat0=cs.allsegs[1][0]
    convex_hull_vol=np.zeros(len(dat0))
    s_strain=np.zeros(len(dat0)) 

    for j in range(len(dat0)):
        convex_hull_vol[j]=STZ.convex_hull_relaxed_config(dat0[j][1],dat0[j][0],initial,no_atoms)
        s_strain[j]=dat0[j][0]
    
    return convex_hull_vol, s_strain


def RDF_plots_normal_stress(initial,no_atoms,beta,iteration,i):
    from ovito.data import Particles, DataCollection, SimulationCell
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import CoordinationAnalysisModifier
    import matplotlib.pyplot as plt    
    particles = Particles()
    data = DataCollection()
    data.objects.append(particles)
    cell = SimulationCell(pbc = (False, False, False))
    cell[...] = [[200,0,0,0],                               #use cell vectors for given congifuration from ovito
                [0,72.56,0,0],
                [0,0,145.12,0]]
    cell.vis.line_width = 0.1
    data.objects.append(cell)
    pos_prop = particles.create_property('Position', data=initial)
    id=range(1,no_atoms+1)
    pipeline = Pipeline(source = StaticSource(data = data))
    pipeline.add_to_scene()
    modifier = CoordinationAnalysisModifier(cutoff = 10.0 , number_of_bins=100)
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()
    rdf_alpha_beta=data.tables['coordination-rdf'].xy()
    max_rdf = rdf_alpha_beta[:,1].max()
    plt.plot(rdf_alpha_beta[:,0],rdf_alpha_beta[:,1])
    plt.title('RDF plot for Normal Stress = 0 and shear strain= {}'.format(beta))
    plt.xlabel('Pair Separation Distance')
    plt.ylabel('g(r)')
    plt.savefig('Clusters/{}_cluster/cluster_{}/RDF_step_{}'.format(no_atoms,iteration,i), facecolor='white', transparent=False)
    plt.clf()
    return max_rdf

    
def interatomic_dist_matrix(initial,no_atoms):
    import STZ
    import numpy as np
    import math
    interatomic_dist_mat = np.zeros([no_atoms,no_atoms])
    for i in range(no_atoms):
        for j in range(no_atoms):
            atom_i_coord = initial[i,:]
            atom_j_coord = initial[j,:]
            interatomic_dist_mat[i,j]= math.dist(atom_i_coord,atom_j_coord)
    return interatomic_dist_mat

def bond_matrix_config(alpha,beta,initial,thresh,no_atoms):
    import numpy as np
    import STZ
    from lammps import lammps
    deformation_grad=np.array([[1,0,beta],[0,1,0],[0,0,1+alpha]])
    d_initial=np.zeros([no_atoms,3])
    for k in range(no_atoms):
            d_initial[k,:]=np.matmul(deformation_grad,initial[k,1:])
    surface_id=STZ.surface_atoms(d_initial,no_atoms)
    surface_id_str=surface_id.astype('str')
    surface_str=" "
    for r in range(len(surface_id)):
        surface_str=surface_str+surface_id_str[r]
        surface_str+=" "     
    lmp=lammps()
    initialization_block='''
    dimension 3
    units metal
    boundary s s s
    atom_style atomic
    timestep 0.001
    region myregion block 0.0 200.0 0.0 72.56 0.0 145.12  units box
    create_box  1 myregion
    mass 1 63.546

    pair_style eam
    pair_coeff 1 1 Cu_u6.eam '''
    lmp.commands_string(initialization_block)
    create_atoms=['create_atoms 1 single {} {} {}'.format(d_initial[l,0],d_initial[l,1],d_initial[l,2]) for l in range(len(initial[:,0]))]
    lmp.commands_list(create_atoms)
    surface_group=['group surface id {}'.format(surface_str)]
    lmp.commands_list(surface_group)

    minimization_block='''
    fix freeze surface setforce 0 0 0 
    minimize 0 1e-4 100000 100000
    unfix freeze
    '''
    lmp.commands_string(minimization_block)
    lmp.command('compute force all property/atom fx fy fz')
    lmp.command('dump fcal all custom 1 dump.force id type x y z fx fy fz')    
    lmp.command('run 1')
    coordinates,_=STZ.extract_fdump('dump.force',no_atoms)
    interatomic_dist_mat = STZ.interatomic_dist_matrix(coordinates,no_atoms)
    bonded_matrix = np.where(interatomic_dist_mat<thresh,1,0)
    return bonded_matrix


def Normal_Stress_Matrix_p(no_atoms,max_alpha,max_beta,total_disp_steps,iteration):
    from lammps import lammps
    import matplotlib.pyplot as plt
    import numpy as np
    import STZ
    # from mpi4py import MPI
    from scipy import interpolate, stats 
    # comm = MPI.COMM_WORLD
    # rank = comm.rank
    ## NORMAL STRESS MATRIX GENERATION ##
    d_alpha=np.linspace(-max_alpha,max_alpha,total_disp_steps+1)
    d_beta=np.linspace(0,max_beta, num=total_disp_steps+1)
    ns=np.zeros([len(d_alpha),len(d_beta)])
    initial=STZ.extract_dat('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(no_atoms,iteration,no_atoms),no_atoms)
    y=-1
    for i in d_alpha:
        alpha=i
        y=y+1
        for j in range(0,total_disp_steps+1):
            # if (rank == 0):
            beta=(max_beta/total_disp_steps)*j
            deformation_grad=np.array([[1,0,beta],[0,1,0],[0,0,1+alpha]])
            d_initial=np.zeros([no_atoms,3])
            for k in range(no_atoms):
                d_initial[k,:]=np.matmul(deformation_grad,initial[k,1:])
            x_min,x_max,y_min,y_max,z_min,z_max = STZ.box_coordinates(d_initial)
            surface_id=STZ.surface_atoms(d_initial,no_atoms)
            surface_id_str=surface_id.astype('str')
            surface_str=" "
            for r in range(len(surface_id)):
                surface_str=surface_str+surface_id_str[r]
                surface_str+=" "     
            initialization_block='''
            dimension 3
            units metal
            boundary s s s
            atom_style atomic
            timestep 0.001
            region myregion block {} {} {} {} {} {}  units box
            create_box  1 myregion
            mass 1 63.546

            pair_style eam
            pair_coeff 1 1 Cu_u6.eam
            '''.format(x_min,x_max,y_min,y_max,z_min,z_max)

            create_atoms=['create_atoms 1 single {} {} {}'.format(d_initial[l,0],d_initial[l,1],d_initial[l,2]) for l in range(len(initial[:,0]))]
            create_atoms_str = '\n'.join(str(e) for e in create_atoms)


            surface_id=STZ.surface_atoms(d_initial,no_atoms)
            surface_id_str=surface_id.astype('str')
            surface_str=" "
            for r in range(len(surface_id)):
                surface_str=surface_str+surface_id_str[r]
                surface_str+=" "    

            surface_group='\ngroup surface id {}'.format(surface_str)

            minimization_block='''
            fix freeze surface setforce 0 0 0 
            minimize 0 1e-4 100000 100000
            unfix freeze
            compute force all property/atom fx fy fz
            dump fcal all custom 1 dump.force id type x y z fx fy fz
            run 1 
            '''
            lammps_input_script = initialization_block+create_atoms_str+surface_group+minimization_block
            # lammps_input_script = comm.bcast(lammps_input_script,root=0)
            # comm.Barrier()
            lmp = lammps()
            lmp.commands_string(lammps_input_script)
            # if (rank == 0):
            coordinates,force=STZ.extract_fdump('dump.force',no_atoms)
            ns[y,j],_=STZ.stress_calc(coordinates,force)
            # comm.Barrier()
    # if(rank ==0):
        np.savetxt('Clusters/{}_cluster/cluster_{}/Normal_stress.txt'.format(no_atoms,iteration),ns)

def box_coordinates (def_initial):
    margin = 2 # based on type of atom and its unit cell
    x_min = def_initial[:,0].min()-margin
    x_max = def_initial[:,0].max()+margin
    y_min = def_initial[:,1].min()-margin
    y_max = def_initial[:,1].max()+margin
    z_min = def_initial[:,2].min()-margin
    z_max = def_initial[:,2].max()+margin
    return x_min,x_max,y_min,y_max,z_min,z_max

def moving_average(a, n):
    import numpy as np
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def segments_fit(X, Y, count):
    from scipy import optimize
    import numpy as np
    import pylab as pl
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)



def yield_strain_bs(max_alpha,max_beta,total_disp_steps,no_atoms,iteration,no_segs):
    import STZ
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate
    d_alpha=np.linspace(-max_alpha,max_alpha,total_disp_steps+1)
    d_beta=np.linspace(0,max_beta, num=total_disp_steps+1)
    Beta,Alpha=np.meshgrid(d_beta,d_alpha)
    ns=np.loadtxt('Clusters/{}_cluster/cluster_{}/Normal_stress.txt'.format(no_atoms,iteration))
    initial=STZ.extract_dat('Clusters/{}_cluster/cluster_{}/cluster_{}.dat'.format(no_atoms,iteration,no_atoms),no_atoms)
    initial_interatomic_mat = STZ.interatomic_dist_matrix(initial,no_atoms)
    cs = plt.contour(Beta,Alpha,ns,0)
    dat0=cs.allsegs[1][0]
    ss_lvl=np.zeros(len(dat0))
    s_strain=np.zeros(len(dat0))
    distance= 2.6  #Angstroms
    factor = 1
    bond_length_thresh = factor*distance
    initial_bonded_matrix = np.where(initial_interatomic_mat<bond_length_thresh,1,0)
    bond_change_frequency = np.zeros(len(dat0))
    for j in range(len(dat0)):
        current_bonded_matrix,_,ss_lvl[j] = STZ.bond_matrix_stress_config(dat0[j][1],dat0[j][0],initial,bond_length_thresh,no_atoms)
        s_strain[j]=dat0[j][0]
        bond_status_change_matrix = current_bonded_matrix-initial_bonded_matrix
        initial_bonded_matrix = current_bonded_matrix
        bond_change_frequency[j] = np.count_nonzero(bond_status_change_matrix)

    bond_change_frequency_trunc = np.delete(bond_change_frequency,0)
    cummulative_sum = 0
    bond_change_cumulative = np.zeros(len(bond_change_frequency_trunc))
    for i in range(len(bond_change_frequency_trunc)):
        cummulative_sum = cummulative_sum+bond_change_frequency[i+1]
        bond_change_cumulative[i] = cummulative_sum

    bond_change_cumulative_averaged= STZ.moving_average(bond_change_cumulative,4)
    bond_status_data =np.column_stack((dat0[0:-4,0],bond_change_cumulative_averaged))                                                               
    X, Y = bond_status_data[0:360,0], bond_status_data[0:360,1]
    px, py = STZ.segments_fit(X, Y, no_segs)
    thresh_zero = px[1]+0.025
    stress_diff = np.zeros(len(ss_lvl))
    for i in range(len(ss_lvl)-1):
        stress_diff[i] = ss_lvl[i+1]-ss_lvl[i]
    drop_strains = s_strain[stress_diff<0]
    drop_strains_useful = drop_strains[drop_strains< thresh_zero]
    if(len(drop_strains_useful)!=0):
        y_interp = interpolate.interp1d(s_strain,ss_lvl)
        tau_o_zero = y_interp(drop_strains_useful).max()
        strain_index = y_interp(drop_strains_useful).argmax()
        yield_strain = drop_strains_useful[strain_index]
    else:
        strains_useful = s_strain[s_strain<thresh_zero]
        y_interp = interpolate.interp1d(s_strain,ss_lvl)
        tau_o_zero = y_interp(strains_useful).max()
        strain_index = y_interp(strains_useful).argmax()
        yield_strain = strains_useful[strain_index]

    plt.figure(figsize=[6,4], dpi=300)
    plt.title('{} atoms STZ at normalised normal stress = 0.0'.format(no_atoms))
    ax1 = plt.subplot()
    color = 'tab:red'
    ax1.set_xlabel('Shear Strain')
    ax1.set_ylabel('Shear Stress (GPa)', color=color) 
    l1, = ax1.plot(s_strain,ss_lvl/10**9, color=color)
    ax1.plot(yield_strain, tau_o_zero/10**9, marker='x', markersize=10, color='r' )
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Cummulative Bond Status Changes', color=color)
    l2, = ax2.plot(bond_status_data[0:360,0], bond_status_data[0:360,1], ".",color=color,)
    ax2.plot(px, py, "-or")
    ax2.tick_params(axis='y', labelcolor=color)
    plt.legend([l2, l1], ["Bond status changes", "Stress vs Strain"])
    plt.savefig('Clusters/{}_cluster/cluster_{}/bond_freq_yield_0.0.png'.format(no_atoms,iteration), facecolor='white', transparent=False)
    plt.clf()
    return yield_strain


def bond_matrix_stress_config(alpha,beta,initial,thresh,no_atoms):
    import numpy as np
    import STZ
    from lammps import lammps
    deformation_grad=np.array([[1,0,beta],[0,1,0],[0,0,1+alpha]])
    d_initial=np.zeros([no_atoms,3])
    for k in range(no_atoms):
            d_initial[k,:]=np.matmul(deformation_grad,initial[k,1:])
    x_min,x_max,y_min,y_max,z_min,z_max = STZ.box_coordinates(d_initial)
  
    initialization_block='''
    dimension 3
    units metal
    boundary s s s
    atom_style atomic
    timestep 0.001
    region myregion block {} {} {} {} {} {}  units box
    create_box  1 myregion
    mass 1 63.546

    pair_style eam
    pair_coeff 1 1 Cu_u6.eam
    '''.format(x_min,x_max,y_min,y_max,z_min,z_max)
    
    create_atoms=['create_atoms 1 single {} {} {}'.format(d_initial[l,0],d_initial[l,1],d_initial[l,2]) for l in range(len(initial[:,0]))]
    create_atoms_str = '\n'.join(str(e) for e in create_atoms)

    surface_id=STZ.surface_atoms(d_initial,no_atoms)
    surface_id_str=surface_id.astype('str')
    surface_str=" "
    for r in range(len(surface_id)):
        surface_str=surface_str+surface_id_str[r]
        surface_str+=" "    

    surface_group='\ngroup surface id {}'.format(surface_str)

    minimization_block='''
    fix freeze surface setforce 0 0 0 
    minimize 0 1e-4 100000 100000
    unfix freeze
    compute force all property/atom fx fy fz
    dump fcal all custom 1 dump.force id type x y z fx fy fz
    run 1 
    '''
    lammps_input_script = initialization_block+create_atoms_str+surface_group+minimization_block
    lmp = lammps()
    lmp.commands_string(lammps_input_script)
    coordinates,force=STZ.extract_fdump('dump.force',no_atoms)
    interatomic_dist_mat = STZ.interatomic_dist_matrix(coordinates,no_atoms)
    bonded_matrix = np.where(interatomic_dist_mat<thresh,1,0)
    ns,ss=STZ.stress_calc(coordinates,force)
    return bonded_matrix, ns ,ss