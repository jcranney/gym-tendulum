import numpy as np
import control
import gym
import gym_tendulum

env = gym.make('tendulum-v0')
s = env.reset()

def get_A(s,u,delta=1e-6):
    A = np.zeros(((env.env._n_order+1)*2,(env.env._n_order+1)*2))
    a_0 = np.linalg.solve(env.env._mass_matrix(s),env.env._f_vector(s,u))
    for i in range((env.env._n_order+1)*2):
        s_ = s.copy()
        s_[i] += delta
        A[:,i] = np.linalg.solve(env.env._mass_matrix(s_),env.env._f_vector(s_,u)) - a_0
    A /= delta
    return A

def get_B(s,u,delta=1e-6):
    B = np.zeros(((env.env._n_order+1)*2,1))
    b_0 = np.linalg.solve(env.env._mass_matrix(s),env.env._f_vector(s,u))
    u += delta
    B[:,0] = np.linalg.solve(env.env._mass_matrix(s),env.env._f_vector(s,u)) - b_0
    B /= delta
    return B

n_state = env.observation_space.shape[0]
n_actu  = env.action_space.shape[0]

Rd_mat = np.array([[1e-5]])
Qd_mat = np.eye(n_state)
Qd_mat[0,0] = 1.0

def get_K(s):
    A_mat = get_A(s,0)
    B_mat = get_B(s,0)
    C_mat = np.eye(n_state)
    D_mat = np.zeros((n_state,n_actu))
    ss = control.ss(A_mat,B_mat,C_mat,D_mat)
    dss = control.c2d(ss,env.env.tau)
    K,_,_ = control.dlqr(dss.A,dss.B,Qd_mat,Rd_mat)
    return K

done = False
u = 0.0
K = get_K(s*0.0)
score = 0.0
while not done:
    env.render()
    #K = get_K(s)
    u = -K @ s
    s,r,done,_ = env.step(u.astype(np.float32))
    score += r
    if score > 500:
        done = True
        print(f"Survived. Convergence quality: {r:.5f}")
print(f"Score: {score:.1f}")