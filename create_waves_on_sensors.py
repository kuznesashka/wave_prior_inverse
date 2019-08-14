def create_waves_on_sensors(cortex, params, G, start_point):

    import numpy as np

    speeds = params['speeds']
    duration = params['duration']
    Fs = params['Fs']

    vertices = cortex[0][1]
    VertConn = cortex[0][3]
    VertNormals = cortex[0][4]

    # Create matrix with template paths in different directions from the starting point
    neighbour_step_1 = VertConn[start_point, :].nonzero()[1]  # nearest neighbours of the starting vertex
    max_step = 100
    num_dir = len(neighbour_step_1)  # number of propagation directions
    path_indices = np.zeros([num_dir, max_step], dtype=int)  # vertices forming the path
    for n in range(0, num_dir):
        path_indices[n, 0] = start_point
        neighbour_ind = neighbour_step_1[n]
        path_indices[n, 1] = neighbour_ind

        norm_start = np.mean(VertNormals[neighbour_step_1], axis=0)  # average normal to all of the nearest neighbours
        norm_start = norm_start[:, np.newaxis]
        norm_start = norm_start / np.linalg.norm(norm_start)
        P_norm = np.identity(3) - norm_start @ norm_start.T  # projection away from average normal

        direction_0 = vertices[neighbour_ind] - vertices[start_point]
        direction_0 = direction_0 @ P_norm.T
        direction_0 = direction_0 / np.linalg.norm(direction_0)
        d = 2
        while d <= max_step-1:
            neighbour_step_2 = VertConn[neighbour_ind, :].nonzero()[1]
            cs = np.zeros(len(neighbour_step_2))
            for p in range(0, len(neighbour_step_2)):
                direction = vertices[neighbour_step_2[p]] - vertices[neighbour_ind]
                direction = direction @ P_norm.T
                direction = direction / np.linalg.norm(direction)
                cs[p] = direction @ direction_0.T

            csmax_ind = np.argmax(cs)
            neighbour_ind = neighbour_step_2[csmax_ind]
            path_indices[n, d] = neighbour_ind
            d += 1

    # for all directions compute distance to the following point

    dist = np.zeros([num_dir, max_step-1])
    for d in range(0, num_dir):
        for i in range(0, max_step-1):
            dist[d, i] = np.linalg.norm(vertices[path_indices[d,i+1]]-
                                        vertices[path_indices[d,i]])

    ntpoints = int(Fs*duration)+1 # number of time points to generate
    path_final = np.zeros([num_dir, len(speeds), ntpoints, 3])
    forward_model = np.zeros([num_dir, len(speeds), ntpoints, 102])
    tstep = 1/Fs

    for s in range(0, len(speeds)):
        l = speeds[s]*tstep
        for d in range(0, num_dir):
            path_final[d,s,0,:] = vertices[start_point]
            forward_model[d,s,0,:] = G[:, start_point]
            res = 0
            v1 = 0
            v2 = 1
            for t in range(1, ntpoints):
                if l < res:
                    alpha = 1-l/res
                    path_final[d, s, t, :] = alpha*path_final[d, s, (t-1), :] + (1-alpha)*vertices[path_indices[d, v2]]
                    forward_model[d, s, t, :] = alpha*forward_model[d, s, (t-1), :] + (1-alpha)*G[:, path_indices[d, v2]]
                    res = res - l
                elif l > res:
                    if res == 0:
                        if l < dist[d, (v2-1)]:
                            alpha = 1-l/dist[d, (v2-1)]
                            path_final[d, s, t, :] = alpha*vertices[path_indices[d, v1]] + (1-alpha)*vertices[path_indices[d, v2]]
                            forward_model[d, s, t, :] = alpha*G[:, path_indices[d, v1]] + (1-alpha)*G[:, path_indices[d, v2]]
                            res = dist[d, (v2-1)] - l
                        elif l == dist[d, (v2-1)]:
                            path_final[d, s, t, :] = vertices[path_indices[d, v2]]
                            forward_model[d, s, t, :] = G[:, path_indices[d, v2]]
                            v1 += 1
                            v2 += 1
                        else:
                            l2 = l - dist[d, (v2-1)]
                            v1 += 1
                            v2 += 1
                            while l2 > dist[d, (v2-1)]:
                                l2 = l2-dist[d, (v2-1)]
                                v1 += 1
                                v2 += 1
                            alpha = 1 - l2/dist[d, (v2-1)]
                            path_final[d, s, t, :] = alpha*vertices[path_indices[d, v1]] + (1-alpha)*vertices[path_indices[d, v2]]
                            forward_model[d, s, t, :] = alpha*G[:, path_indices[d, v1]] + (1-alpha)*G[:, path_indices[d, v2]]
                            res = dist[d, (v2-1)] - l2
                    else:
                        l2 = l - res
                        v1 += 1
                        v2 += 1
                        while l2 > dist[d, (v2-1)]:
                            l2 = l2 - dist[d, (v2-1)]
                            v1 += 1
                            v2 += 1
                        alpha = 1 - l2/dist[d, (v2-1)]
                        path_final[d, s, t, :] = alpha*vertices[path_indices[d, v1]] + (1-alpha)*vertices[path_indices[d, v2]]
                        forward_model[d, s, t, :] = alpha*G[:, path_indices[d, v1]] + (1-alpha)*G[:, path_indices[d, v2]]
                        res = dist[d, (v2-1)] - l2
                else:
                    path_final[d, s, t, :] = vertices[path_indices[d, v2]]
                    forward_model[d, s, t, :] = G[:, path_indices[d, v2]]
                    v1 += 1
                    v2 += 1

    wave = np.zeros([ntpoints, ntpoints])
    t = np.arange(0, ntpoints)
    n = np.arange(1, ntpoints+1)
    for i in t:
        wave[i, :] = (1 + np.cos(2 * np.pi * (n - i) / ntpoints))

    sensor_waves = np.zeros([num_dir+1, len(speeds), ntpoints, 102])
    for s in range(0, len(speeds)):
        for i in range(0, num_dir):
            for t in range(0, ntpoints):
                fm_s = np.zeros([102, ntpoints])
                for k in range(0, ntpoints):
                    fm_s[:, k] = forward_model[i, s, k, :]
                sensor_waves[i, s, t, :] = fm_s@wave[t].T

    for s in range(0, len(speeds)):
        for t in range(0, ntpoints):
            for i in range(0, num_dir):
                sensor_waves[num_dir, s, t, :] = sensor_waves[num_dir, s, t, :] + sensor_waves[i, s, t, :]
            sensor_waves[num_dir, s, t, :] = sensor_waves[num_dir, s, t, :]/num_dir

    return sensor_waves