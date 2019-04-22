import time
from DDPG_algorithm import DDPGAlgorithm
from DDPG_agent import DDPGAgent
import paddle.fluid.profiler as profiler
import pickle
import threading as th
from parl.utils import logger


ACTION_DIM = 19
VEL_OBS_DIM = 4
OBS_DIM = 185 + VEL_OBS_DIM
GAMMA = 0.96
TAU = 0.001
actor_lr = 3e-5
critic_lr = 3e-5
train_times = 100
BATCH_SIZE = 128

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_num', type=int)
parser.add_argument('--test_times', type=int, default=10)
args = parser.parse_args()
ensemble_num = args.ensemble_num

alg = DDPGAlgorithm(OBS_DIM, VEL_OBS_DIM, ACTION_DIM, GAMMA, TAU, gpu_id=0, ensemble_num=ensemble_num)
agent = DDPGAgent(alg, no_mem_allocation=False)

with open('batch_data.pickle', 'rb') as f:
    batch_data = pickle.load(f)

def train_single_model(model_id):
    global alg, agent, batch_data
    [states,actions,rewards,dones,new_states] = batch_data
    for idx in range(100):
        need_fetch = True
        agent.learn(states, actions, rewards, new_states, dones, actor_lr, critic_lr, model_id, need_fetch=need_fetch)


if __name__ == '__main__':
    start_time = time.time()
    for i in range(args.test_times):
        th_list = []
        for j in range(ensemble_num):
            t = th.Thread(target=train_single_model, args=(j, ))
            t.start()
            th_list.append(t)

        for t in th_list:
          t.join()
    logger.info("[learn] {} heads, time consuming: {}".format(ensemble_num, (time.time() - start_time) / args.test_times))
