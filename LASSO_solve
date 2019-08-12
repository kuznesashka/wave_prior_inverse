import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, enet_path

data = np.loadtxt('spike_cluster5.csv', delimiter=",")
waves = np.loadtxt('waves_cluster5.csv', delimiter=",")
ndir = np.loadtxt('ndir_cluster5.csv', delimiter=",")

#print(waves.shape) # waves (90(sum of all directions) x 25704(12S*102ch*20ms))
#print(data.shape) # data (2040(102channels*20ms) x 273(nspikes*R))

# speeds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
nspikes = len(ndir) # number of spikes in the cluster
R = int(data.shape[1]/nspikes) # number of shifts in sliding window
S = int(waves.shape[1]/data.shape[0]) # number of propagation speeds
T = data.shape[0]

regression = ElasticNetCV(positive=True, normalize=True, cv=5, max_iter=10000) # elastic net regression
cumdir = 0

bestind = np.zeros([nspikes,], dtype=int) # indices of best speeds
numdir = np.zeros(nspikes,) # number of nonzero directions in optimum
#bestcoef = np.zeros([nspikes, int(ndir.T.max())])
finalscore = np.zeros(nspikes,)

for ind_sp in range(0,nspikes):
    Ndir = int(ndir[ind_sp]) # number of propagation directions
    datasp = data[:, ind_sp*R:(ind_sp + 1)*R]  # spike 21x2040
    wavessp = waves[cumdir:cumdir + Ndir]  # waves for this spike Ndir x 12S*102ch*20ms

    coefs = np.zeros([R,S,Ndir]) # regression coefficients
    intercept = np.zeros([R,S]) # regression intercept
    score = np.zeros([R,S]) # R-squared scores
    nzdir = np.zeros([R,S]) # number of nonzero directions
    y_pred = np.zeros([R,S,data.shape[0]]) # predicted spikes
    cumdir = cumdir+Ndir
    for r in range(0,R):
        DataLin = datasp[:,r] # for each time slice
        for s in range(0,int(S)):
            wavesspeed = wavessp[:,(s*T):((s+1)*T)]
            regression.fit(wavesspeed.T, DataLin)
            coefs[r,s,:] = regression.coef_
            intercept[r,s] = regression.intercept_
            score[r,s] = regression.score(wavesspeed.T, DataLin)
            y_pred[r,s,:] = regression.predict(wavesspeed.T)
            nzdir[r,s] = np.sum(coefs[r,s,:]!=0)

    bestshifts = score.argmax(axis = 0) # best shifts for each speed
    bestscore = score.max(axis = 0) # corresponding scores

    bestdir = np.zeros(S,) # corresponding number of nonzero directions (without intercept)
    for s in range(0,S):
        bestdir[s] = nzdir[bestshifts[s],s]

    score_sort_ind = (-bestscore).argsort() # indices of sorted scores for all speeds
    dir_sort_ind = (bestdir[score_sort_ind[0:3]]).argsort() # indices of sorted number of nonzero directions for top-3 scores

    bestind[ind_sp] = score_sort_ind[dir_sort_ind[0]] # index of best speed
    numdir[ind_sp] = bestdir[bestind[ind_sp]] # number of nonzero directions in optimum
    # bestcoef[ind_sp, 0:Ndir] = coefs[bestshifts[bestind[ind_sp]], bestind[ind_sp], :]
    finalscore[ind_sp] = bestscore[bestind[ind_sp]] # R-squared value in optimum
    print(ind_sp)

    #plt.figure()
    #plt.plot(DataLin)
    #plt.plot(y_pred[bestshifts[bestind[ind_sp]], bestind[ind_sp], :])
    #plt.title(['R-squared = ', str(finalscore[ind_sp])])