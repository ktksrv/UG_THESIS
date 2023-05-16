import STZ
import os
number_of_cluster=1
max_alpha=0.2
max_beta=0.4
total_disp_steps=349

for size_of_cluster in [608]:
    path='Clusters/{}_cluster'.format(size_of_cluster)
    for i in range(1,number_of_cluster+1):
        os.makedirs('Clusters/{}_cluster/cluster_{}'.format(size_of_cluster,i))
        STZ.random_cluster_generator(size_of_cluster,i)
        STZ.Normal_Stress_Matrix(size_of_cluster,max_alpha,max_beta,total_disp_steps,i)
