import gym
from gym import error, spaces, utils
import numpy as np
import itertools

## environemnt parameters


#how probabilistic the environment is, the bigger the number, the more garbage collection and harder to improve WA
gc_prob = .75 
# number of write points (keep small because it is tabular)
write_points = 2
#number of spots per write point, this is can be small because we assume that the drive is close to full already when we start measuring WA
space = 4

##learning hyper parameters

#how much to value future returns compared to instant returns
discount_factor = 0.9 
# adjustment rate for q-values
learning_rate = 0.2 
# total learning episodes
episodes = 100 
#lower exploration by 4% after each learning episode in favor of exploitation
eps_decay = .96 

#number of writes per episode, this should be about 5 X number of states = 5*2*((space+1)**write_points)*(write_points**2)*4 to ensure learning
episode_length = 5*2*((space+1)**write_points)*write_points**2*4
print("length of each training episode: %s" %episode_length)


class SimpleFlash(gym.Env):
    metadata = {'render.modes': ['human']}
    
    #state: features from drive and user requests
    #counter: counts up to episode length
    #done: is true when counter == episode length
    #write_ampllification: total WA observed so far
    #actions_space: choice of write point
    def __init__(self):
        #make a little write-points array (better than using numpy because of typing problems)
        wp = []
        for i in range(write_points):
            wp.append(0) 
        self.write_point_quality = np.random.randint(0,write_points)
        self.prev_write_point = np.random.randint(0,write_points)
        self.cluster = np.random.randint(2)
        self.gc = not self.cluster
        self.state = [1,self.prev_write_point,wp,self.write_point_quality,self.cluster,self.gc] #we assume we start with a small request, to keep things simple
        self.counter = 0
        self.done = False
        self.write_amplification = 0
        self.action_space = spaces.Discrete(write_points)
        self.secret = []
        self.total_discounts = 0 #diagnostic tool

    
    
    #user generates write requests. It is a markov chain
    #if the  previous request is small, it is likely
    #produce a small request, if the previous request
    #is large, then it is very likely to produce a small
    #request
    def user(self):
        if np.random.random() < .8:
            return 1
        else:
            return 2

    #helper to change the state of a point and calculate instant reward by accounting for GC
    def inst_reward(self, a: int, b:int, c:int):
        #a is starting fullness of a write point, b is the request size, c is the ending fullness after GC
        numerator, denominator = 0,0
        final_fullness = a
        while b > 0:
            final_fullness = min(space,final_fullness + numerator)
            if final_fullness < space:
                denominator+= 1
            else:
                denominator += 2
            numerator += 1
            b -= 1
        final_fullness = min(space,final_fullness + numerator)
        return numerator/(denominator + (final_fullness - c))
    
    def write_data(self, k: int):
        branch = np.random.random()
        total_load = min(self.state[0]+k,space) #current request + fullness
        #encodes the randomness in the environment, either you pay for what you write
        if branch < 1 - gc_prob:
            fullness = total_load
            reward = self.inst_reward(k,  self.state[0],fullness)
        #or you collect even more garbage and then pay for that too
        elif branch >= 1 - gc_prob:
            fullness = np.random.randint(0,total_load+1)
            reward = self.inst_reward(k,  self.state[0],fullness)
        return fullness, reward


                
    def step(self,action):
        if self.counter < episode_length:
            #get a new request
            self.state[0] = self.user()
            #write the data and get the instantaneous reward
            self.state[2][action], reward = self.write_data(self.state[2][action])
            #update write amplification

            #check whether clustering and GC discounts apply
            if action == self.write_point_quality and self.gc:
                reward = min(1, (1.5)*reward)
                self.total_discounts += 1
            if action == self.prev_write_point and self.cluster:
                reward = min(1, (1.5)*reward)
                self.total_discounts += 1
            
            #accumulate long term write amplification
            self.write_amplification = (self.counter*self.write_amplification + (1/reward))/(self.counter + 1)
            #add one to counter
            self.counter += 1
            #make a new best GC write point
            self.write_point_quality = np.random.randint(0,write_points)
            self.state[3] = self.write_point_quality
            #new prev write point
            self.prev_write_point = np.random.randint(0,write_points)
            self.state[1] = self.prev_write_point

            #update whether discounts are available
            self.cluster = np.random.randint(2)
            self.gc = not self.cluster
            self.state[4] = self.cluster
            self.state[5] = self.gc
            return self.state, reward, False, [self.write_amplification,action]
        else:
            return self.state, 0, True, [self.write_amplification,action]


    def reset(self):
        self.cluster = np.random.randint(2)
        self.gc = not self.cluster
        self.write_point_quality = np.random.randint(0,write_points)
        self.prev_write_point = np.random.randint(0,write_points)
        wp = []
        for i in range(write_points):
            wp.append(0) 
        self.state = [1,self.prev_write_point,wp,self.write_point_quality,self.cluster,self.gc]
        self.counter = 0
        self.done = False
        self.write_amplification = 0
        self.total_discounts = 0
        return self.state


#begin some RL

env = SimpleFlash()

#pick actions randomly
def rand_act():
    return env.action_space.sample()

#pick actions greedily according to oracle
def greedy_oracle_policy():
    return env.write_point_quality

#greedy policy given a q-table
def policy_greedy(q_table, state):
    s = state_to_int(state)
    return np.argmax(q_table[s])

#balance exploration and exploitation given a q-table
def policy_eps_greedy(q_table, state, epsilon):
    if(np.random.random() < epsilon):
        action = rand_act()
    else:
        action = policy_greedy(q_table, state)
    return action


#baseline for random actions
for i in range(episode_length):
    env.step(rand_act())

print("random actions WA: %f " % env.write_amplification)
print("number of discounted steps %d/%d" %(env.total_discounts, episode_length))
env.reset()

#baseline for oracle guided actions
for i in range(episode_length):
    env.step(greedy_oracle_policy())

print("greedy oracle WA: %f " % env.write_amplification)
print("number of discounted steps %d/%d" %(env.total_discounts, episode_length))
env.reset()


#initialize q-table

#helper function to format q-table with mutable lists rather than immutable tuples
def tuple_to_list(t):
    ret = []
    for i in range(len(t)):
        ret.append(t[i])
    return ret

#insert every possible state into a list in lexicographical order
state_to_int_list = [] 

for i in range(1,3):
    for x in itertools.product(range(space+1), repeat=write_points):
        for y in itertools.product(range(write_points), repeat=2):
            for z in itertools.product(range(2), repeat=2):
                state_to_int_list.append([i,y[0],tuple_to_list(x),y[1],z[0],z[1]])


#utilize the lexicographical ordering in order to be able to recall a state
def state_to_int(state):
    return state_to_int_list.index(state)

env.state = env.reset()

def bellman_update(q_table, learning_rate, discount_factor, reward, state, state_next, action):
    s = state_to_int(state)
    s_next = state_to_int(state_next)
    q_table[s][action] = q_table[s][action] +learning_rate*(reward + discount_factor*np.max(q_table[s_next]) - q_table[s][action])
    return q_table

def run_random_episode(env, q_table, epsilon,learning_rate, discount_factor):
    state = env.reset()
    done = False
    while(not done):
        action = policy_eps_greedy(q_table,state,epsilon)
        wrapper_prev_state = state_to_int(state) #need to save previous state otherwise state_next and state end up pointing to the same thing
        state_next, reward, done, _ = env.step(action)
        q_table = bellman_update(q_table, learning_rate, discount_factor, reward, state_to_int_list[wrapper_prev_state], state_next, action)
  
    return(q_table)


# Normalize Q-values for readability
def normalize_q_values(q_table):
  q_table_max = np.max(q_table)
  if q_table_max > 0.0:  
    q_table = (q_table/q_table_max)*100.0
  return q_table


#let's define a geenric training agent function, which will run a bunch of episodes and report the progress
def train_agent(env, epsiodes, learning_rate, discount_factor, eps_decay):
    reward_history = np.array([])
    size = 2*((space+1)**write_points)*(3**write_points)*4

    q_table = np.zeros((size,write_points))
    epsilon = .99
  
    for episode in range(episodes):
        if(epsilon > .01):
            epsilon *= eps_decay
        q_table = run_random_episode(env, q_table, epsilon, learning_rate, discount_factor)
        reward_history = np.append(reward_history, env.write_amplification)
        print("%d/100 episodes completed: current trained WA with %f exploration is %f" % ((episode + 1) ,epsilon,reward_history[episode]))
    
    return(reward_history, normalize_q_values(q_table))

reward_curve, trained_q = train_agent(env,episodes,learning_rate,discount_factor,eps_decay)

num = 1
#let's test our trained q-table!
for episode in range(num):
  state = env.reset()
  done = False
  while(not done):
    action = policy_greedy(trained_q, state)
    state, reward, done,_ = env.step(action)
  print("Q-trained WA: %f " % env.write_amplification)
  print("number of discounted steps %d/%d" %(env.total_discounts, episode_length))

