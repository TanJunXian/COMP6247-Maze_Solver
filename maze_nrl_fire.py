import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from read_maze import *

class MazeRunner():

    #Initialisation
    def __init__(self,runner=(1,1)):

        self.world = np.ones((201,201,2))*5 

        nrows, ncols = self.world.shape[0], self.world.shape[1]
        self.lr = 1
        self.epoch = 1

        self.win_reward = 1
        self.default_reward = -0.0
        self.visited_penalty = -0.02
        self.wall_penalty = -0.1

        self.discount_factor = .5

        self.step = 0
        self.step_limit = 200000
        self.target = (nrows-2,ncols-2)

        self.actions = [self.up, self.left, self.right, self.down]
        self.action_labels = ['UP', 'LEFT', 'RIGHT', 'DOWN']

        self.q_values = np.zeros([self.world.shape[0], self.world.shape[1], len(self.actions)])

        self.reset(runner)

    def check_terminal_state(self,action_idx):
        row, col = self.state
        nrow, ncol = self.nstate

        if (nrow, ncol) == self.target:
            
            self.q_values[row, col, action_idx] = self.q_values[row, col,action_idx] + self.lr*(self.win_reward + self.discount_factor * self.q_values[nrow, ncol].max() - self.q_values[row, col,action_idx])
            self.state = np.copy(self.nstate)
            self.plot()
            self.epoch += 1
            self.reset()

        if self.step >= self.step_limit:
            self.plot()
            self.epoch += 1
            self.reset()

    def reset(self,runner=(1,1)):

        self.runner = runner
        self.world = np.copy(self.world)
    
        self.state = runner
        self.nstate = runner

        self.visited = set()
        self.visited.add(runner)

        self.fire = set()
        self.step = 0

    def up(self):
        action_idx = 0
        row, col = self.state
        new_r = row - 1

        if self.world[new_r, col, 1] > 0:
            self.fire.add((new_r,col))
            reward = self.default_reward
            pass

        elif self.world[new_r, col, 0] == 0:
            reward = self.wall_penalty

        elif (new_r,col) in self.visited:
            row = new_r
            reward = self.visited_penalty

        else:
            row = new_r
            reward = self.default_reward

        self.nstate = (row,col)

        return reward, action_idx

    def left(self):
        action_idx = 1
        row, col = self.state
        new_c = col - 1

        if self.world[row, new_c, 1] > 0:
            self.fire.add((row,new_c))
            reward = self.default_reward
            pass

        elif self.world[row, new_c, 0] == 0:
            reward = self.wall_penalty

        elif (row, new_c) in self.visited:
            col = new_c
            reward = self.visited_penalty

        else:
            col = new_c
            reward = self.default_reward
            
        self.nstate = (row,col)

        return reward, action_idx

    def right(self):
        action_idx = 2
        row, col = self.state
        new_c = col + 1

        if self.world[row, new_c, 1] > 0:
            self.fire.add((row,new_c))
            reward = self.default_reward
            pass

        elif self.world[row, new_c, 0] == 0:
            reward = self.wall_penalty
        
        elif (row, new_c) in self.visited:
            col = new_c
            reward = self.visited_penalty
        
        else:
            col = new_c
            reward = self.default_reward

        self.nstate = (row,col)

        return reward, action_idx

    def down(self):
        action_idx = 3
        row, col = self.state
        new_r = row + 1

        if self.world[new_r, col, 1] > 0:
            self.fire.add((new_r, col))
            reward = self.default_reward
            pass
        
        elif self.world[new_r, col, 0] == 0:
            reward = self.wall_penalty
 
        elif (new_r,col) in self.visited:
            row = new_r
            reward = self.visited_penalty

        else:
            row = new_r
            reward = self.default_reward

        self.nstate = (row,col)

        return reward, action_idx
    
    def pre_back(self):

        row, col = self.state
        nrow, ncol = self.nstate

        if nrow == row -1:
            q_vals = self.q_values[nrow, ncol]
            qv = q_vals[3] + self.lr*(self.visited_penalty)
            self.q_values[nrow, ncol, 3] = qv            

        if ncol == col -1:
            q_vals = self.q_values[nrow, ncol]
            qv = q_vals[2] + self.lr*(self.visited_penalty)
            self.q_values[nrow, ncol, 2] = qv  

        if nrow == row +1:
            q_vals = self.q_values[nrow, ncol]
            qv = q_vals[0] + self.lr*(self.visited_penalty)
            self.q_values[nrow, ncol, 0] = qv  

        if ncol == col +1:
            q_vals = self.q_values[nrow, ncol]
            qv = q_vals[1] + self.lr*(self.visited_penalty)
            self.q_values[nrow, ncol, 1] = qv

    def map(self,around):
        row, col = self.state

        self.world[row-1,col-1] = around[0, 0]
        self.world[row-1,col]   = around[0, 1]
        self.world[row-1,col+1] = around[0, 2]
        self.world[row,col-1]   = around[1, 0]
        self.world[row,col]     = around[1, 1]
        self.world[row,col+1]   = around[1, 2]
        self.world[row+1,col-1] = around[2, 0]
        self.world[row+1,col]   = around[2, 1]
        self.world[row+1,col+1] = around[2, 2]

    def action(self):

        row, col = self.state
        # self.map(get_info(row,col))
        self.map(get_local_maze_information(row,col))

        q_vals = self.q_values[row, col]
        if q_vals.sum() != 0:
            reward, action_idx = self.actions[np.argmax(q_vals)]()
        else:
            reward, action_idx = self.actions[random.randint(0,3)]()

        nrow, ncol = self.nstate

        self.step += 1
        qv = q_vals[action_idx] + self.lr*(reward + (self.discount_factor * self.q_values[nrow, ncol].max()) - q_vals[action_idx])
        self.q_values[row, col, action_idx] = qv

        self.pre_back()
        # print("Epoch: {0} Step: {1} Coordinate: {2} ---> {3} Action: {4}".format(self.epoch,self.step,self.state,self.nstate,self.action_labels[action_idx]))
        with open("/Users/junxian/Desktop/Maze_CW//Output.txt", 'a') as f:
            f.write("Epoch: {0} Step: {1} Coordinate: {2} ---> {3} Action: {4} \n".format(self.epoch,self.step,self.state,self.nstate,self.action_labels[action_idx]))

        self.check_terminal_state(action_idx)

        self.state = np.copy(self.nstate)

        if self.world[row, col, 0] > 0.0 and ((row,col) not in self.visited):
            self.visited.add((row, col))

    def plot(self):
            File="Pics"
            plt.figure(figsize=(9,9))
            plt.grid('on')
            nrows, ncols = self.world.shape[0],self.world.shape[1]
            ax = plt.gca()
            ax.tick_params(axis='both', which='both', top=False, bottom=False,left=False,right=False,grid_alpha=0,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            canvas = np.copy(self.world)

            cmap = colors.ListedColormap(['black','white','tan', 'deepskyblue','gold','grey','crimson'])
            bounds = [0,1,2,3,4,5,6,7]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            for row,col in self.visited:
                canvas[row,col,0] = 3
            rat_row, rat_col = self.state
            canvas[nrows-2, ncols-2, 0] = 4
            canvas[rat_row, rat_col, 0] = 2

            for row,col in self.fire:
                canvas[row,col,0] = 6

            plt.imshow(canvas[:,:,0], interpolation='none', cmap=cmap,norm=norm)
            if self.step<self.step_limit:
                plt.title("Epochs: "+str(self.epoch)+"  Steps: "+str(self.step)+"  Status: Escaped")
                plt.savefig(File+"/"+str(self.epoch)+"_"+str(self.step)+"_ESCAPED")
            else:
                plt.title("Epochs: "+str(self.epoch)+"  Steps: "+str(self.step)+"  Status: Trapped")
                plt.savefig(File+"/"+str(self.epoch)+"_"+str(self.step)+"_TRAPPED")
