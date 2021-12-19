import pickle as pk
import os
from utils_psgail import sampling
import sys
import numpy as np
from multiagent_traffic_simulator import MATrafficSim
from smarts.env.wrappers.parallel_env import ParallelEnv



# {
#     'model': psgail,
#     'epoch': i_episode,
#     'rewards_log': rewards_log,
#     'episodes_log': episodes_log,
#     'agent_num': agent_num,
#     'stage': stage,
#     'dis_ag_losses': dis_ag_losses,
#     'dis_ex_losses': dis_ex_losses,
#     'dis_gp_losses': dis_gp_losses,
#     'pol_losses': pol_losses,
#     'val_losses': val_losses,
# },

# Increase system recursion limit
sys.setrecursionlimit(25000)
device1 = "cuda:0"
device0 = "cpu"

def load_model(model2test):
    with open('./models/' + model2test, "rb") as f:
        models = pk.load(f)
    return models


def eval(psgail, batch_size=10000, agent_num=10):
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
    vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
    states, next_states, actions, probs, dones, rewards, total_agent_num = sampling(psgail, vector_env,
                                                                                    batch_size=batch_size)
    print('avg reward per agent: {}\navg survival time per agent: {}'.format(np.sum(rewards) / total_agent_num,
                                                                             batch_size / total_agent_num))
    vector_env.close()
    return None

if __name__ == "__main__":
    current_path = 'models'
    filename_list = os.listdir(current_path)
    # filename = 'psgail_3_tuning.model'
    # print(filename)
    # modesls = load_model(filename)
    # psgail = modesls['model']
    # eval(psgail, 10000, 10)
    for filename in filename_list:
        filename = 'psgail_15_train.model'
        print(filename)
        modesls = load_model(filename)
        psgail = modesls['model']
        eval(psgail, 10000, 10)
