def LASSO_inverse_solve(data, waves):
    """Function to compute the inverse solution with LASSO with positive coefficients
    Parameters
    ----------
    data : numpy.ndarray
        Input data for the inverse problem (N_channels x T)
    waves : numpy.ndarray
       Basis waves to fit (directions_number x number_of_speeds x channels_number x timepoints_number)
    Returns
    -------
    best_score : R-squared in optimum
    best_coefs : coefficients in optimum
    best_shift : starting time point in optimum
    best_speed_ind : index of the best speed
    """

    import numpy as np
    from sklearn.linear_model import ElasticNetCV

    Ndir = waves.shape[0] # number of propagation directions
    R = data.shape[1]-waves.shape[3] + 1 # number of sliding window shifts
    S = waves.shape[1] # number of propagation speeds
    Tw = waves.shape[3]

    regression = ElasticNetCV(l1_ratio=1, positive=True, cv=5, max_iter=100000) # elastic net regression

    coefs = np.zeros([R, S, Ndir]) # regression coefficients
    intercept = np.zeros([R,S]) # regression intercept
    score = np.zeros([R, S]) # R-squared scores
    nzdir = np.zeros([R, S]) # number of nonzero directions
    y_pred = np.zeros([R, S, data.shape[0]*data.shape[1]]) # predicted spikes

    for r in range(0, R):
        data_vec = data[:,r:(Tw+r)].flatten()
        for s in range(0, S):
            wavesspeed = waves[:, s, :, :]
            wavesspeed_vec = np.zeros([Ndir, data_vec.shape[0]])
            for d in range(0, Ndir):
                wavesspeed_vec[d] = wavesspeed[d,:,:].flatten()
            regression.fit(wavesspeed_vec.T, data_vec)
            coefs[r, s, :] = regression.coef_
            intercept[r, s] = regression.intercept_
            score[r, s] = regression.score(wavesspeed_vec.T, data_vec)
            y_pred[r, s, :] = regression.predict(wavesspeed_vec.T)
            nzdir[r, s] = np.sum(coefs[r, s, :] != 0)

    shifts_s = score.argmax(axis = 0) # best shifts for each speed
    score_s = score.max(axis = 0) # corresponding scores

    # best solution according to the number of nonzero directions
    # nzdir_s = np.zeros(S) # corresponding number of nonzero directions (without intercept)
    # for s in range(0, S):
    #     nzdir_s[s] = nzdir[shifts_s[s], s]
    #
    # score_sort_ind = (-score_s).argsort() # indices of sorted scores for all speeds
    # dir_sort_ind = (nzdir_s[score_sort_ind[0:3]]).argsort() # indices of sorted number of nonzero directions for top-3 scores
    #
    # best_speed_ind = score_sort_ind[dir_sort_ind[0]] # index of best speed

    score_sort_ind = (-score_s).argsort()
    best_speed_ind = score_sort_ind[0]
    best_intercept = intercept[shifts_s[best_speed_ind], best_speed_ind]
    best_coefs = coefs[shifts_s[best_speed_ind], best_speed_ind]
    best_score = score_s[best_speed_ind] # R-squared value in optimum
    best_shift = shifts_s[best_speed_ind]

    # plt.figure()
    # plt.plot(data_vec)
    # plt.plot(y_pred[bestshifts[bestind], bestind, :])
    # plt.title(['R-squared = ', str(finalscore)])

    return [best_score, best_intercept, best_coefs, best_shift, best_speed_ind]