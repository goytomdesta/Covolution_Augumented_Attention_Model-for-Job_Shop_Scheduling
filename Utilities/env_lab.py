import numpy as np
from Utilities.JSSP_Env import SJSSP
from Utilities.uniform_instance_gen import uni_instance_gen
from Utilities.Params import configs
import time

import torch

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


n_j = 200
n_m = 50
low = 1
high = 99
SEED = 11
np.random.seed(SEED)
env = SJSSP(n_j=n_j, n_m=n_m)


# rollout env random action
t1 = time.time()
data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
dur = np.array([[83, 65,  3],
               [69, 42, 64],
               [27, 27, 18]])
mch = np.array([[3, 2, 1],
                [1, 2, 3],
                [2, 1, 3]])
# data = (dur, mch)
print('Dur')
print(data[0])
print('Mach')
print(data[-1])
print()
_, _, omega, mask = env.reset(data)
# print('Init end time')
# print(env.LBs)
# print()
rewards = [- env.initQuality]
while True:
    action = np.random.choice(omega[~mask])
    # print('action:', action)
    adj, _, reward, done, omega, mask = env.step(action)
    rewards.append(reward)
    # print('ET after action:\n', env.LBs)
    # print(fea)
    # print()
    if env.done():
        break
t2 = time.time()
makespan = sum(rewards) - env.posRewards
# print(makespan)
# print(env.LBs)
print(t2 - t1)

# np.save('sol', env.opIDsOnMchs // n_m)
# np.save('jobSequence', env.opIDsOnMchs)
# np.save('testData', data)
# print(env.opIDsOnMchs // n_m + 1)
# print(env.step_count)
# print(t)
# print(np.concatenate((fea, data[1].reshape(-1, 1)), axis=1))
# print()
# print(env.adj)
