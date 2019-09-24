import numpy as np


def horn_method(p, q):
    """Function to compute the transformation between set of points in 3d space using Horn
    method (Horn, 1987)
       Parameters
       ----------
       p : nd.array
           coordinates of first set of point in 3d
       q : nd.array
           coordinates of the second set of points in 3d
       Returns
       -------
       R: rotation matrix
       s: the overall scaling factor
       t: translation vector
       """

    ss = np.sum(p*q, axis=0)
    sxy = np.sum(p[:, 0]*q[:, 1])
    syx = np.sum(p[:, 1]*q[:, 0])
    sxz = np.sum(p[:, 0]*q[:, 2])
    szx = np.sum(p[:, 2]*q[:, 0])
    syz = np.sum(p[:, 1]*q[:, 2])
    szy = np.sum(p[:, 2]*q[:, 1])

    N = np.array([[ss[0]+ss[1]+ss[2], syz-szy, szx-sxz, sxy-syx],
                 [syz-szy, ss[0]-ss[1]-ss[2], sxy+syx, szx+sxz],
                 [szx-sxz, sxy+syx, -ss[0]+ss[1]-ss[2], syx+szx],
                 [sxy-syx, szx+sxz, syx+szx, -ss[0]-ss[1]+ss[2]]])

    [d, v] = np.linalg.eig(N)
    v_sorted = v[:, np.argsort(-d)]

    v0 = v_sorted[0, 0]
    vx = v_sorted[1, 0]
    vy = v_sorted[2, 0]
    vz = v_sorted[3, 0]

    R = np.array([[v0**2+vx**2-vy**2-vz**2, 2*(vx*vy - v0*vz), 2*(vx*vz + v0*vy)],
                  [2*(vy*vx + v0*vz), v0**2 - vx**2 + vy**2 -vz**2, 2*(vy*vz - v0*vx)],
                  [2*(vz*vx + v0*vy), 2*(vz*vy + v0*vx), v0**2 - vx**2 - vy**2 + vz**2]])

    aq = np.mean(q, axis=0)
    ap = np.mean(p, axis=0)
    s = np.sum(np.diag((q-aq) @ R @ (p-ap).T)) / np.sum((p-ap)**2)
    t = aq - s * R @ ap

    return R, s, t
