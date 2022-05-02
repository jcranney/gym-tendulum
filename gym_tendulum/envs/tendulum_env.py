import gym
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.integrate import solve_ivp
import math
from typing import Optional, Union
import os

import numpy as np

import gym
from gym import logger, spaces
from gym.utils import seeding

class TendulumEnv(gym.Env):
    """Environment for the 10th order ndulum.
    
    Observations:
    ----------------------------------------
      index  | observation
    ----------------------------------------
        0    | Cart position
        1    | Angle of 1st rod (wrt vertical)
        2    | Angle of 2nd rod (wrt vertical)
       ...   | ...
        10   | Angle of 10th rod (wrt vertical)
    ----------------------------------------
     10+0    | Cart velocity
     10+1    | Angle velocity of 1st rod
     10+2    | Angle velocity of 2nd rod
       ...   | ...
     10+10   | Angle velocity of 10th rod
    ----------------------------------------
    
    Actions:
    ----------------------------------------
     action | description
    ----------------------------------------
       0    | Push cart to the left
       1    | No push
       2    | Push cart to the right
    ----------------------------------------
    
    Rewards:
    ----------------------------------------
     reward | description
    ----------------------------------------
       1    | for all non-terminal states  
       0    | for terminal states
    ----------------------------------------
    
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self):
        # order of the ndulum
        self._n_order = 10

        # length of each segment
        self._ell = np.ones(self._n_order+1)*3.0/self._n_order
        self._ell[0] = np.NaN
        self._mass = np.ones(self._n_order+1)*0.1/self._n_order
        self._mass[0] = 0.1 # mass of the cart
        self._gravity = 9.8
        self.u_max = 10.0
        self._max_episode_steps = 500
        self._theta_threshold_radians = 12 * np.pi / 180

        self.tau = 0.02  # seconds between state updates
         
        # Angle at which to fail the episode
        self._theta_threshold_radians = 0.5 * math.pi
        self._x_threshold = 4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.r_[
                self._x_threshold * 2,
                [self._theta_threshold_radians * 2] * self._n_order,
                [np.finfo(np.float32).max]*(self._n_order+1),
            ].astype(np.float32)

        self.action_space = spaces.Box(low=-self.u_max,high=self.u_max,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.steps_beyond_done = None
    
    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        if type(action) is not np.ndarray or action.dtype != np.float32 or action.shape != (1,):
            action = np.r_[action].astype(np.float32)
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        force = action[0]
            
        # integrate forward by self.tau seconds, applying the newest force
        self.state = solve_ivp(lambda t,x: np.linalg.solve(self._mass_matrix(x),self._f_vector(x,force)), 
                      [0,self.tau], self.state, method="RK45").y[:,-1]
        
        done = np.any(np.abs(self.state[0]) > self._x_threshold) \
            or np.any(np.abs(self.state[1:(self._n_order+1)]) > self._theta_threshold_radians)

        self._episode_step += 1
        done = done or (self._episode_step > self._max_episode_steps)

        if not done:
            reward = -(self.state[:2]**2).sum()
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -(self.state[:2]**2).sum()
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
        options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-1e-1,high=1e-1,size=((1+self._n_order)*2,))
        self.state[0] = -1.0
        self.state[self._n_order+1:] = 0.0
        self.steps_beyond_done = None
        self._episode_step = 0
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
    
    def render(self, mode='human',render_text=""):
        import pygame
        pygame.font.init()
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(render_text, False, (0, 0, 0))
        # create a surface on screen that has the size of 240 x 180
        width = 400
        height = 400
        x0 = width/2
        y0 = height/2
        black = (0,0,0)
        grey = (100,100,100)
        red = (100,0,0)
        background = pygame.surfarray.make_surface(np.ones([width, height, 3])*255)

        world_width = self._x_threshold * 2
        scale = width / world_width / 2 
        polewidth = 5.0
        cartwidth = 80.0
        cartheight = 60.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            # initialize the pygame module
            pygame.init()
            pygame.display.set_caption(f"N-Dulum: Order {self._n_order:d}")
            self.screen = pygame.display.set_mode((width,height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.screen.blit(background,(0,0))
        self.screen.blit(text_surface,(10,10))
        pygame.draw.line(self.screen,black,(0,y0+cartheight/2),(width,y0+cartheight/2),1)
        pygame.draw.line(self.screen,black,(x0,0),(x0,height),1)
        pygame.draw.rect(
            self.screen, grey, 
            pygame.Rect(x0+x[0]*scale-cartwidth/2,y0,cartwidth,cartheight)
        )
        verts = self._get_vertices(x,scale=scale,offset=(x0,y0),height=height)
        pygame.draw.lines(self.screen, black, False,verts,width=int(polewidth))
        for vert in verts:
            pygame.draw.circle(self.screen, red, vert, 5)
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    
        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _mass_matrix(self,x):
        """
        Mass matrix for the ndulum.
        """
        m = np.zeros((2*(1+self._n_order),2*(1+self._n_order)))
        
        # block 1
        m[:1+self._n_order,:1+self._n_order] = np.eye(1+self._n_order)
        
        # block 2
        block_2 = np.zeros((1+self._n_order,1+self._n_order))
        for i in range(1+self._n_order):
            for j in range(i,1+self._n_order):
                if i == 0:
                    if j == 0:
                        block_2[i,j] = self._mass.sum()
                    else:
                        block_2[i,j] = self._mass[j:].sum()*self._ell[j]*np.cos(x[j])
                else:
                    block_2[i,j] = self._ell[i]*self._ell[j]*np.cos(x[i]-x[j])*self._mass[np.max([i,j]):].sum()
                block_2[j,i] = block_2[i,j]
        m[1+self._n_order:,1+self._n_order:] = block_2
        return m

    def _f_vector(self,x,u):
        """
        RHS of the ndulum.
        
        Args:
            x (np.ndarray): state vector
            u (float): input
        """
        f = np.zeros(2*(1+self._n_order))
        
        # vec 1
        f[:1+self._n_order] = x[1+self._n_order:]
        
        # vec 2
        vec_2 = np.zeros(1+self._n_order)
        vec_2[0] = u + np.sum([np.sum([
            self._mass[j]*self._ell[i]*x[1+self._n_order+i]**2*np.sin(x[i]) 
            for i in range(1,j+1)]) for j in range(1,self._n_order+1)])
        for k in range(1,1+self._n_order):
            vec_2[k] = self._gravity*np.sin(x[k])*self._ell[k]*self._mass[k:].sum() - np.sum([np.sum([
                self._ell[k]*self._mass[j]*self._ell[i]*x[1+self._n_order+i]**2*np.sin(x[k]-x[i])
                for i in range(1,j+1)]) for j in range(k,self._n_order+1)])
        #vec_2 -= 0.001*(x[1+self._n_order:])
        f[1+self._n_order:] = vec_2
        return f

    def _get_vertices(self,x,offset,scale,height):
        """
        Get the vertices of the ndulum.
        
        Args:
            x (np.ndarray): state vector
        """
        vertices = np.zeros(((1+self._n_order),2))
        vertices[0,:] = [x[0], 0]
        for i in range(1,1+self._n_order):
            vertices[i,:] = [
                vertices[i-1,0] + self._ell[i]*np.sin(x[i]),
                vertices[i-1,1] + self._ell[i]*np.cos(x[i])]
        vertices[:,0] = vertices[:,0]*scale + offset[0]
        vertices[:,1] = height-(vertices[:,1]*scale + offset[1])
        return vertices