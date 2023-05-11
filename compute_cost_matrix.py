import numpy as np

Pose = np.array([[3,0,0],
                 [1,0,0],
                 [2,1,0]])

goal = np.array([1,0])

P = goal - Pose[:,:2]
cos = np.cos(Pose[:,2])
sin = np.sin(Pose[:,2])
R = np.stack([cos,sin,-sin,cos], axis =1).reshape(cos.size,2,2)
# print(R)

batch = np.einsum('nij,nj->ni', R, P)
alpha = np.arctan2(batch[:,-1], batch[:,0])
cost_M_normalized = np.abs(alpha)/np.max(np.abs(alpha))
best_traj = Pose[np.argmin(cost_M_normalized)]
