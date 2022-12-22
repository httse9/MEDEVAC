from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *
from scipy.stats import beta

from itertools import product

def Myopic(env):
    _, zone = env.request

    if zone == -1:
        print('weird! This zone resquest sahould not have occured')
        action = env.M_n

    service_rate = env.mu_zm[zone, :]
    valid_actions = env.valid_actions[:-1]

    best_m, best_t = env.M_n, -1
    for idx in range(env.M_n):
        if valid_actions[idx]:
            if service_rate[idx] > best_t:
                best_t = service_rate[idx]
                best_m = idx
    
    # print(valid_actions, best_m)
    return best_m

class MedEvac(Environment):
    """
    environment described in paper: https://www.sciencedirect.com/science/article/pii/S0305048318304699
    """

    def __init__(self, Z_n=34):
        # termination parameters
        self.gamma_term = 0.95
        self.max_horizon = 100

        self.M_n = 4
        self.K_n = 3
        self.Z_n = Z_n
        self.state = np.zeros((self.M_n + self.K_n, self.Z_n))
        self.n_actions = self.M_n + 1
        self.gamma = 1.0 # for discounting, different from gamma_term

        # load MedEvac environment specifications
        if self.Z_n == 12:
            # For arrival rates
            self.partial_lmd = np.genfromtxt("./medevac_env_spec/AR_12.csv", delimiter=",")
            self.pZ = self.partial_lmd.sum(axis=1) / self.partial_lmd.sum() # does not sum to 1, hack
            self.pK = np.array([0.16, 0.16, 0.68])  # does not sum to 1, use values for 34 zone
            self.base_lmd = 1.0/30
            assert np.shape(self.partial_lmd) == (self.Z_n, self.K_n)

            # For completion times. ST: service times
            self.ST = np.genfromtxt('./medevac_env_spec/ST_12.csv', delimiter=',')
            self.ST_max = np.max(self.ST)
            self.ST_min = np.min(self.ST)
            self.mu_zm = 1.0 / self.ST
            self.mu = np.sum(np.max(self.mu_zm[:, idx]) for idx in range(self.M_n))
            assert np.shape(self.mu_zm) == (self.Z_n, self.M_n)

            # For reward function. RT: response times
            self.RT = np.genfromtxt('./medevac_env_spec/RT_12.csv', delimiter=',')
            self.RT_max = np.max(self.RT)
            self.RT_min = np.min(self.RT)
            assert np.shape(self.RT) == (self.Z_n, self.M_n)

        elif self.Z_n == 34:
            # For arrival rates
            self.pZ = np.genfromtxt('./medevac_env_spec/AR_zone_34.csv', delimiter=' ')
            self.pK = np.genfromtxt('./medevac_env_spec/AR_priority_34.csv', delimiter=' ')
            self.partial_lmd = self.pK * self.pZ.reshape(-1,1)
            self.base_lmd = 1.0/30
            assert np.shape(self.partial_lmd) == (self.Z_n, self.K_n)

            # For completion times
            self.ST = np.genfromtxt('./medevac_env_spec/ST_34.csv', delimiter=' ')
            self.ST_max = np.max(self.ST)
            self.ST_min = np.min(self.ST)
            self.mu_zm = 1.0 / self.ST
            self.mu = np.sum(np.max(self.mu_zm[:, idx]) for idx in range(self.M_n))
            assert np.shape(self.mu_zm) == (self.Z_n, self.M_n)

            # For reward function. RT: response times
            self.RT = np.genfromtxt('./medevac_env_spec/RT_34.csv', delimiter=' ')
            self.RT_max = np.max(self.RT)
            self.RT_min = np.min(self.RT)
            assert np.shape(self.RT) == (self.Z_n, self.M_n)

        # environment parameters
        self.lmd_zk = self.base_lmd * self.partial_lmd
        self.varphi = self.base_lmd + self.mu
        
        # weight of requests of different priorities
        self.w_k = np.array([100.0, 10.0, 1.0])
        self.w_k /= np.sum(self.w_k)

        # reward function
        self.R_zm = {}
        self.R_zm[0] = self.w_k[0] * beta(3,5).cdf((150 - self.RT)/150.0)
        self.R_zm[1] = self.w_k[1] * beta(1,1).cdf((400 - self.RT)/400.0)
        self.R_zm[2] = self.w_k[2] * beta(1,1).cdf((2400 - self.RT)/2400.0)

        # Various variables to keep track of how the interaction evolves
        self.request = np.zeros(2, dtype=int) -1                        # Initialized to (-1, -1)
        self.active = np.zeros((self.M_n, 2))                           # No MEDEVACs active initially
        self.current_ST = np.zeros(self.M_n)
        self.valid_actions = np.ones(self.n_actions)     # All actions valid initially

        self.RBF_centers_initialized = False
        self.reset()
        self.num_features = len(self.get_observation())
        self._create_env_des()

    def _six_numbers(self):
        state_numbers = []
        state = self.state.copy()

        for m in range(self.M_n):
            if state[m].sum() == 0: # unit m not dispatched
                state_numbers.append(0)
            else: # find where unit m is dispatched
                state_numbers.append(np.argmax(state[m]) + 1)

        for k in range(self.K_n):
            for z in range(self.Z_n):
                if state[self.M_n + k, z] == 1:
                    state_numbers += [k + 1, z + 1]

        if len(state_numbers) < env.M_n + 2:
            state_numbers += [0, 0]

        return state_numbers


    def _myopic_action(self,):
        _, zone = self.request

        if zone == -1:
            print('weird! This zone resquest sahould not have occured')
            raise ValueError()
            action = env.M_n

        service_rate = self.mu_zm[zone, :]
        valid_actions = self.valid_actions.copy()

        best_m, best_t = self.M_n, -1
        for idx in range(self.M_n):
            if valid_actions[idx]:
                if service_rate[idx] > best_t:
                    best_t = service_rate[idx]
                    best_m = idx
        
        # print(valid_actions, best_m)
        return best_m


    def _create_env_des(self,):
        each_dim_bound = np.array([[0.0, 1.0]])
        observation_space_bounds = np.repeat(each_dim_bound, self.num_features, axis=0)
        observation_space = Continuous_Space(observation_space_bounds)

        action_space = Discrete_Space(0, 4) # 0123: units, 4: no-op. Different from paper
        self.env_description =  Env_Description(observation_space, action_space)

    def get_observation(self, features="RBF"):
        """
        return state features directly
        state: 
        """
        if features == "polynomial":
            return self._polynomial_state_features()
        elif features == "RBF":
            return self._RBF_state_features()
        else:
            raise ValueError(f"Feature {features} not recognized.")

    def _initialize_RBF_centers(self, n, type='tile'):
        # random
        """
        No RT and Valid, No Normalization at least 500 ; 10, 20, 50, 200, 500 (2.2), 1000 (3.13), 2000 (3.32)
        No RT and Valid ; 50 (1.7), 200 (2.94), 500 (2.54), 
        No RT ;  200 (1.78), 500 (2.8), 1000 (2.43)
        All ; 200 (2.07), 500 (2.54)


        200 All 3.355691523122539
        """
        if type == 'random':
            n_centers = 200
            np.random.seed(0)   # each time generate same centers
            self.centers = np.random.uniform(size=(n_centers, n))
            np.random.seed()
        
        elif type == 'tile':
            num_bins = 2
            bin_vals = np.array([((i/n)) for i in range(0, num_bins)])
            raw_centers = np.tile(bin_vals, (n, 1)).tolist()
            cart_centers = list(product(*raw_centers))
            
            self.centers = np.array(cart_centers)

        
        self.RBF_centers_initialized = True
        
    def _RBF_state_features(self):
        # print(self.active)
        # print(self.valid_actions)

        k, z = self.request

        request_ST = (self.ST[z].copy() - self.ST_min) / self.ST_max
        request_RT = (np.copy(self.RT[z]) - self.RT_min) / self.RT_max
        # print(request_ST)

        # print(self.w_k[k])

        state_features = np.concatenate([
            self.current_ST, request_ST, [self.w_k[k]], request_RT,
        ])

        if not self.RBF_centers_initialized:
            self._initialize_RBF_centers(len(state_features))

        state_features = state_features.reshape(1, -1)
        feats = np.exp(-(((state_features - self.centers)*2) ** 2).sum(axis=1))

        feats /= feats.sum() # normalization

        val = self.valid_actions.copy()[:-1]
        val -= (val == 0).astype(int)
        feats = np.concatenate([feats, val])

        
        # print(feats.shape)
        # print(feats)
        return feats

    def _polynomial_state_features(self, order=2):
        # current_service_rates = np.copy(self.active[:, 1]) # service rates of all units
        # current_service_rates += (current_service_rates == 0).astype(int) # set to 1 if unit available

        current_ST = self.active[:, 1].copy()
        current_ST += (current_ST == 0).astype(int)
        current_ST = 1 / current_ST
        current_ST = (current_ST - self.ST_min) / self.ST_max
        current_ST *= (1 - self.valid_actions[:self.M_n])

        k, z = self.request

        request_ST = (self.ST[z].copy() - self.ST_min) / self.ST_max

        # request_service_time_normalized = np.copy(self.ST[z]) / self.ST_max
        # request_service_rates = np.copy(self.mu_zm[z])
        # request_response_time_normalized = np.copy(self.RT[z]) / self.RT_max
        # request_response_rates = 1 / np.copy(self.RT[z])
        
        # valid_actions = self.valid_actions.copy()
        # rewards = self.R_zm[k][z]

        # feats = np.concatenate([
        #     valid_actions, current_service_rates,
        #     request_service_rates, request_response_rates,
        #     # request_service_time_normalized, request_response_time_normalized,
        #     # request_response_time_normalized ** 2, request_response_time_normalized ** 2,
        #     [self.w_k[k]],  
        #     #1 / rewards # 1 / to bound between 0 and 1
        # ])

        # feats += np.random.normal(loc=0.1, size=(len(feats)))


        feats = np.concatenate([
            current_ST, request_ST, [self.w_k[k]]
        ])
        polyfit = PolynomialFeatures(degree=order)
        poly_feats = polyfit.fit_transform(feats)

        return poly_feats


    def _step(self, action):

        reward = 0

        # assert self.valid_actions[action] == 1
        # if action is invalid, map to Myopic
        if self.valid_actions[action] == 0:
            # reward -= 1     # punish for taking non-valid action
            action = self.M_n
            # action = self._myopic_action()

        # Whether any action was taken or not,
        # previous request needs to be removed from the state
        k, z = self.request
        if k>-1 and z>-1:
            self.state[self.M_n + k, z] = 0
            self.request.fill(-1)

        # If a valid MEDEVAC action is taken
        # then allocate MEDEVAC to that zone
        # Note: action = M_n is for no-op, action in [0, Z_n - 1] corresponds to the Z_n zones
        if action < self.M_n:
            self.valid_actions[action] = 0
            self.state[action, z] = 1
            self.active[action, :] = [z, self.mu_zm[z, action]]
            self.current_ST[action] = (self.ST[z, action] - self.ST_min) / self.ST_max
            reward += self.R_zm[k][z, action]
        

        flag = True
        skip = True # default True
        while flag:
            flag = skip

            # sum of service completion rates for the active MEDEVACs
            mu = np.sum(self.active[:,1])

            # Some event occurs
            # Note: Can get rid of this loop when skip is enabled
            if skip or np.random.rand() < (self.base_lmd + mu) / self.varphi:
                if np.random.rand() < mu / (self.base_lmd + mu):
                    # A service finishes
                    probs = self.active[:,1] / mu
                    M = np.random.choice(self.M_n, p=probs)     # sample the unit to finish service

                    # Update the state variables
                    self.state[M, int(self.active[M,0])] = 0
                    self.valid_actions[M] = 1

                    # Note: This marks zone 0 as active for M, but with completion rate 0. SHould be fine
                    self.active[M, :] = 0 
                    self.current_ST[M] = 0

                else:
                    # A new request arrives 
                    # Following is equivalent to sampling from λ_zk/λ
                    # Note: works for λ in paper, does not work for custom λ
                    z = np.random.choice(self.Z_n, p=self.pZ)
                    k = np.random.choice(self.K_n, p=self.pK)

                    # Update the state variables
                    self.state[self.M_n + k, z] = 1
                    self.request[:] = [k, z] 

                    # Break out of loop; An action is needed
                    flag = False

        

        return reward

    def transition(self, action):
        reward = self._step(action)
        self.ctr += 1
        self._check_terminal()
        return reward

    def _check_terminal(self):
        self.terminal_state = np.random.rand() > self.gamma_term or self.ctr >= self.max_horizon

    def reset(self):
        self.state.fill(0)
        self.request.fill(-1)
        self.active.fill(0)
        self.current_ST.fill(0)
        self.valid_actions.fill(1)
        self.ctr = 0

        self.terminal_state = False # indicate whether episode has terminated, used by self.terminated() in runner
        self._step(self.M_n)   # force to start with a request

    def visualize(self):
        raise NotImplementedError()

    