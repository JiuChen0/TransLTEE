import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
'''
The np.expand_dims operation was performed on each sample to change its shape from (3477,) to (1, 3477).
Thus, after stacking all samples, the final shape obtained is (5000, 1, 3477), which matches the model input.
The treatment, Y, and YC were all subjected to the same deformation operation, since the model may use this information simultaneously.
The dataset is divided into a training set and a test set. This step was not performed in the original code, so I assume that all generated data is used for training. If a test set is needed, some additional modifications may be required.
There is no pre-processing or normalization performed. In practice, appropriate pre-processing may be required depending on the situation.
'''
'''Data generation follow learning representations for counterfactual inference'''
for i in range(1,11):
    TY=np.loadtxt('data/NEWS/topic_doc_mean_n5000_k3477_seed_'+str(i)+'.csv.y',delimiter=',')
    treatment=TY[:,0]
    treated=np.where(treatment>0)[0]
    controlled=np.where(treatment<1)[0]
    y=TY[:,1]
    yc=TY[:,2]

    matrix = np.zeros((5000, 1, 3477))

    with open('data/NEWS/topic_doc_mean_n5000_k3477_seed_'+str(i)+'.csv.x', 'r') as fin:
        for line in fin.readlines():
            line = line.strip().split(',')
            matrix[int(line[0]) - 1, 0, int(line[1]) - 1] = int(line[2])

    #run LDA
    no_topics=50
    z = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit_transform(matrix[:, 0, :])
    z1=z[np.random.randint(5000,size=1),:]
    z0=np.mean(z,axis=0)

    T=100
    Y=np.zeros((5000, T, 1))
    YC=np.zeros((5000, T, 1))

    Y[:,0,0]=y
    YC[:,0,0]=yc
    for t in range(1,T):
        noise=np.random.normal(0,1,1)
        Y[:,t,0]=50*(np.matmul(z,z0.reshape(-1))+treatment*np.matmul(z,z1.reshape(-1)))+0.03*np.sum(Y[:,0:t-1,0],axis=1)+noise
        YC[:, t,0] = 50 * (np.matmul(z, z0.reshape(-1)) + (1-treatment) * np.matmul(z, z1.reshape(-1))) + 0.03 * np.sum(
            YC[:, 0:t - 1,0], axis=1) + noise

    treatment=np.reshape(treatment,(5000,1,1))
    data=np.concatenate((treatment,Y),axis=2)

    y_treated=np.concatenate((YC[controlled],Y[treated]),axis=0)
    y_controlled=np.concatenate((Y[controlled],YC[treated]),axis=0)
    causal_effects = np.mean(y_treated - y_controlled, axis=0)
    np.savetxt('data/NEWS/Series_groundtruth_'+str(i)+'.txt', causal_effects, delimiter=',', fmt='%.2f')
    np.savetxt('data/NEWS/Series_y_'+str(i)+'.txt', data, delimiter=',', fmt='%.2f')