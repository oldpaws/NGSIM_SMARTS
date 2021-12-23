import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time
import math
from collections import defaultdict
import cv2

# Increase system recursion limit
sys.setrecursionlimit(25000)
device1 = "cuda:0"
device0 = "cpu"

headings = {'gnE05b_0': [-1.5708518823503947, -0.1, 180, 2.87],
            'gneE01': [-1.5708606520231623, -0.2, 310.92, 8.43, 3.69],
            'gneE51_0': [-1.5708418823503947, -0.1, 130.92, 3.50]}


def im_show(image):
    cv2.imshow('husky', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def getlist(list_, idx):
    if idx < 0 or idx >= len(list_) or len(list_) == 0:
        return None
    else:
        return list_[idx]


def smooth_curve(y, smooth):
    r = smooth
    length = int(np.prod(y.shape))
    for i in range(length):
        if i > 0:
            if (not np.isinf(y[i - 1])) and (not np.isnan(y[i - 1])):
                y[i] = y[i - 1] * r + y[i] * (1 - r)
    return y


def moving_average(y, x=None, total_steps=100, smooth=0.9, move_max=False):
    if isinstance(y, list):
        y = np.array(y)
    length = int(np.prod(y.shape))
    if x is None:
        x = list(range(1, length + 1))
    if isinstance(x, list):
        x = np.array(x)
    if length > total_steps:
        block_size = length // total_steps
        select_list = list(range(0, length, block_size))
        select_list = select_list[:-1]
        y = y[:len(select_list) * block_size].reshape(-1, block_size)
        if move_max:
            y = np.max(y, -1)
        else:
            y = np.mean(y, -1)
        x = x[select_list]
    y = smooth_curve(y, smooth)
    return y, x


def plotReward(infos):
    x, y = infos["episodes"], infos["rewards"]
    y, x = moving_average(y, x)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(x, y)
    plt.show()


class trajectory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []


class samples_agents():
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []


def dump_trajectory(expert_trajectory, agent_id, batch_samples):
    batch_samples.states += expert_trajectory[agent_id].states
    batch_samples.probs += expert_trajectory[agent_id].probs
    batch_samples.actions += expert_trajectory[agent_id].actions
    batch_samples.next_states += expert_trajectory[agent_id].next_states
    batch_samples.rewards += expert_trajectory[agent_id].rewards
    batch_samples.dones += expert_trajectory[agent_id].dones
    if expert_trajectory[agent_id].states[-1][56] > 300:
        end = 1
    else:
        end = 0
    return len(expert_trajectory[agent_id].states), end, expert_trajectory[agent_id].states[-1][56]


def dump_all(expert_trajectory, agent_traj, total_agent_num, counter, ends, final_xs):
    for expert_traject in expert_trajectory.values():
        for agent_id in expert_traject.keys():
            total_agent_num += 1
            counter += len(expert_traject[agent_id].states)
            agent_traj.states += expert_traject[agent_id].states
            agent_traj.probs += expert_traject[agent_id].probs
            agent_traj.actions += expert_traject[agent_id].actions
            agent_traj.next_states += expert_traject[agent_id].next_states
            agent_traj.rewards += expert_traject[agent_id].rewards
            agent_traj.dones += expert_traject[agent_id].dones
            if expert_traject[agent_id].states[-1][56] > 300:
                ends += 1
            final_xs += expert_traject[agent_id].states[-1][56]
    return total_agent_num, counter, ends, final_xs


def trans2tensor(batch):
    for k in batch:
        # if k == 'action' or k == 'probs':
        #     batch[k] = torch.cat(batch[k], dim=0).to(device1)
        # else:
        batch[k] = torch.tensor(batch[k], device=device1, dtype=torch.float32)
    return batch


def obs_extractor_new(obs_agent):
    if obs_agent is None:
        return np.zeros(62)
    obs_vector = cal_obs(obs_agent, 8)
    return obs_vector


def cal_obs(env_obs, closest_neighbor_num):
    ego = env_obs.ego_vehicle_state
    neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
    features = np.zeros((closest_neighbor_num, 7))
    surrounding_vehicles = _get_closest_vehicles(
        ego, neighbor_vehicle_states, n=closest_neighbor_num
    )
    husky = {'gnE05b_0': [-1.5708518823503947, -0.1, 180, 2.87],
             'gneE01': [-1.5708606520231623, -0.2, 310.92, 8.43, 3.69],
             'gneE51_0': [-1.5708418823503947, -0.1, 130.92, 3.50]}
    husky_idx = {'gneE01': 1, 'gnE05b_0': 2, 'gneE51_0': 3, 'gneE05a_0': 4}
    ego_pos = ego.position[:2]
    ego_heading = np.asarray(float(ego.heading))
    iddx = 0
    for i, v in surrounding_vehicles.items():
        if v[0] is None:
            features[i, :] = np.asarray([0, 0, 0, 0, 0, 0, 0])
            continue
        else:
            v = v[0]
        if husky_idx.get(v.lane_id) is not None:
            iddx = husky_idx.get(v.lane_id)
        elif v.lane_id[:6] in husky_idx:
            iddx = husky_idx['gneE01']
        pos = v.position[:2]
        heading = np.asarray(float(v.heading))
        speed = np.asarray(v.speed)
        rel0 = pos[0] - ego_pos[0]
        rel1 = pos[1] - ego_pos[1]
        if rel0 > 0:
            rel_b0 = rel0 - v[2].length / 2
        else:
            rel_b0 = rel0 + v[2].length / 2
        if rel1 > 0:
            rel_b1 = rel1 - v[2].width / 2
        else:
            rel_b1 = rel1 + v[2].width / 2
        features[i, :] = np.asarray([rel0, rel1, heading - ego_heading, speed, rel_b0, rel_b1, iddx])
    features = features.reshape((-1,))
    ego_pos = np.zeros(20)
    ego_pos[:2] = ego.position[:2]
    ego_pos[2] = ego.heading
    ego_pos[3] = ego.speed
    ego_pos[4:6] = ego.bounding_box.as_lwh[:2]
    ego_pos[6:8] = ego.angular_velocity[:2]
    ego_pos[8:10] = ego.angular_acceleration[:2]
    ego_pos[10:12] = ego.linear_velocity[:2]
    ego_pos[12:14] = ego.linear_acceleration[:2]
    if husky_idx.get(ego.lane_id) is not None:
        if husky_idx.get(ego.lane_id) == 4:
            ego_pos[14:18] = np.array([1, 0, 0, 0])
            ego_pos[18] = 0
            ego_pos[19] = 0
        elif husky_idx.get(ego.lane_id) == 2:
            ego_pos[14:18] = np.array([0, 1, 0, 0])
            ego_pos[18] = (ego.heading - husky['gnE05b_0'][0]) * 100
            ego_pos[19] = ego.position[0] * husky['gnE05b_0'][1] / husky['gnE05b_0'][2] + husky['gnE05b_0'][
                3] - ego.position[1]
        elif husky_idx.get(ego.lane_id) == 3:
            ego_pos[14:18] = np.array([0, 0, 1, 0])
            ego_pos[18] = (ego.heading - husky['gneE51_0'][0]) * 100
            ego_pos[19] = (ego.position[0] - 180) * husky['gneE51_0'][1] / husky['gneE51_0'][2] + \
                          husky['gneE51_0'][3] - ego.position[1]
    elif ego.lane_id[:6] in husky_idx:
        ego_pos[14:18] = np.array([0, 0, 0, 1])
        ego_pos[18] = (ego.heading - husky['gneE01'][0]) * 100
        ego_pos[19] = ego.position[0] * husky['gneE01'][1] / husky['gneE01'][2] + \
                      husky['gneE01'][3] + ego.lane_index * husky['gneE01'][4] - ego.position[1]
    vecs = np.concatenate((features, ego_pos), axis=0)
    return vecs


def sampling(psgail, vector_env, batch_size):
    vector_env.seed(random.randint(1, 500))
    vec_obs = vector_env.reset()
    vec_done = []
    expert_trajectory = {}
    total_agent_num = 0
    for i in range(12):
        expert_trajectory[i] = {}
    agent_traj = samples_agents()
    counter = 0
    ends = 0
    final_xs = 0
    while True:
        vec_act = []
        obs_vectors_orig = np.zeros((1, 76))
        for idx, obs in enumerate(vec_obs):
            for agent_id in obs.keys():
                if agent_id not in expert_trajectory[idx]:
                    expert_trajectory[idx][agent_id] = trajectory()
                elif getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
                    length, end, final_x = dump_trajectory(expert_trajectory[idx], agent_id, agent_traj)
                    counter += length
                    ends += end
                    final_xs += final_x
                    total_agent_num += 1
                    del expert_trajectory[idx][agent_id]
                    continue
                # obs_vectors_orig = np.vstack((obs_vectors_orig, obs_extractor_new(obs[agent_id])))
                obs_vectors_orig = np.vstack((obs_vectors_orig, obs[agent_id]['neighbor'].squeeze()))
                # im_show(obs[agent_id].top_down_rgb.data)
        obs_vectors = torch.tensor(obs_vectors_orig[1:, :], device=device1, dtype=torch.float32)
        acts, prob = psgail.get_action(obs_vectors)
        act_idx = 0
        prob = prob.to(device0)
        acts = acts.to(device0)
        for idx, obs in enumerate(vec_obs):
            act_n = {}
            for agent_id in obs.keys():
                if getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
                    continue
                act_tmp = acts[act_idx].cpu()
                act_n[agent_id] = act_tmp.numpy()
                expert_trajectory[idx][agent_id].states.append(obs_vectors_orig[act_idx + 1])
                expert_trajectory[idx][agent_id].probs.append(prob[act_idx])
                expert_trajectory[idx][agent_id].actions.append(act_n[agent_id])
                act_idx += 1
            vec_act.append(act_n)
        vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
        for idx, act_n in enumerate(vec_act):
            for agent_id in act_n.keys():
                # obs_vectors = obs_extractor_new(vec_obs[idx].get(agent_id)).squeeze()
                if vec_obs[idx].get(agent_id) is None:
                    expert_trajectory[idx][agent_id].next_states.append(np.zeros(76))
                else:
                    expert_trajectory[idx][agent_id].next_states.append(vec_obs[idx][agent_id]['neighbor'].squeeze())
                expert_trajectory[idx][agent_id].rewards.append(vec_rew[idx].get(agent_id))
                expert_trajectory[idx][agent_id].dones.append(vec_done[idx].get(agent_id))
        if counter >= batch_size:
            total_agent_num, counter, ends, final_xs = dump_all(expert_trajectory, agent_traj, total_agent_num, counter,
                                                                ends, final_xs)
            break

    return agent_traj.states, agent_traj.next_states, agent_traj.actions, agent_traj.probs, agent_traj.dones, agent_traj.rewards, total_agent_num, counter, ends, final_xs


# def sampling(psgail, vector_env, batch_size):
#     vector_env.seed(random.randint(1, 500))
#     vec_obs = vector_env.reset()
#     vec_done = []
#
#     expert_trajectory = {}
#     total_agent_num = 0
#     for i in range(12):
#         expert_trajectory[i] = {}
#     agent_traj = samples_agents()
#     counter = 0
#     ends = 0
#     final_xs = 0
#     while True:
#         vec_act = []
#         for idx, obs in enumerate(vec_obs):
#             act_n = {}
#             for agent_id in obs.keys():
#                 if agent_id not in expert_trajectory[idx]:
#                     expert_trajectory[idx][agent_id] = trajectory()
#                 elif getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
#                     length, end, final_x = dump_trajectory(expert_trajectory[idx], agent_id, agent_traj)
#                     counter += length
#                     ends += end
#                     final_xs += final_x
#                     total_agent_num += 1
#                     del expert_trajectory[idx][agent_id]
#                     continue
#                 obs_vectors_orig = obs[agent_id]['neighbor'].squeeze()
#                 expert_trajectory[idx][agent_id].states.append(obs_vectors_orig)
#                 obs_vectors_orig = torch.tensor([obs_vectors_orig], device=device1, dtype=torch.float32)
#                 acts, log_prob = psgail.get_action(obs_vectors_orig)
#                 acts = acts.cpu()
#                 act_n[agent_id] = acts.numpy().squeeze()
#                 expert_trajectory[idx][agent_id].probs.append(log_prob)
#                 expert_trajectory[idx][agent_id].actions.append(act_n[agent_id])
#                 # im_show(obs[agent_id].top_down_rgb.data)
#             vec_act.append(act_n)
#         vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
#         for idx, act_n in enumerate(vec_act):
#             for agent_id in act_n.keys():
#                 if vec_obs[idx].get(agent_id) is None:
#                     expert_trajectory[idx][agent_id].next_states.append(np.zeros(76))
#                 else:
#                     expert_trajectory[idx][agent_id].next_states.append(vec_obs[idx][agent_id]['neighbor'].squeeze())
#                 expert_trajectory[idx][agent_id].rewards.append(vec_rew[idx].get(agent_id))
#                 expert_trajectory[idx][agent_id].dones.append(vec_done[idx].get(agent_id))
#         if counter >= batch_size:
#             total_agent_num, counter, ends, final_xs = dump_all(expert_trajectory, agent_traj, total_agent_num, counter,
#                                                                 ends, final_xs)
#             break
#     return agent_traj.states, agent_traj.next_states, agent_traj.actions, agent_traj.probs, agent_traj.dones, agent_traj.rewards, total_agent_num, counter, ends, final_xs


def sampling_one(psgail, vector_env, batch_size):
    vector_env.seed(random.randint(1, 500))
    vec_obs = vector_env.reset()
    vec_done = []
    states = []
    actions = []
    rewards = []
    next_states = []
    probs = []
    dones = []
    counter = 0
    agents_buffer = {}
    ends = 0
    final_xs = 0
    for i in range(12):
        agents_buffer[i] = {}
    total_agent_num = 0
    while True:
        vec_act = []
        obs_vectors = np.zeros((1, 76))
        for idx, obs in enumerate(vec_obs):
            for agent_id in obs.keys():
                if agent_id not in agents_buffer[idx]:
                    agents_buffer[idx][agent_id] = 1
                    total_agent_num += 1
                if getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
                    if vec_obs[idx][agent_id]['neighbor'][0, 56] > 300:
                        ends += 1
                    final_xs += vec_obs[idx][agent_id]['neighbor'][0, 56]
                    del agents_buffer[idx][agent_id]
                    continue
                # obs_vectors = np.vstack((obs_vectors, obs_extractor_new(obs[agent_id])))
                obs_vectors = np.vstack((obs_vectors, obs[agent_id]['neighbor'].squeeze()))
                states.append(obs_vectors[-1, :])
        obs_vectors = torch.tensor(obs_vectors[1:, :], device=device1, dtype=torch.float32)
        acts, prob = psgail.get_action(obs_vectors.squeeze())
        act_idx = 0
        prob = prob.to(device0)
        acts = acts.to(device0)
        for idx, obs in enumerate(vec_obs):
            act_n = {}
            for agent_id in obs.keys():
                if getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
                    continue
                act_tmp = acts[act_idx].cpu()
                act_n[agent_id] = act_tmp.numpy()
                act_idx += 1
            vec_act.append(act_n)
        probs.append(prob)
        actions.append(acts.numpy())
        vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
        for idx, act_n in enumerate(vec_act):
            for agent_id in act_n.keys():
                # obs_vectors = obs_extractor_new(vec_obs[idx].get(agent_id))
                if vec_obs[idx].get(agent_id) is None:
                    next_states.append(np.zeros(76))
                else:
                    next_states.append(vec_obs[idx][agent_id]['neighbor'].squeeze())
                rewards.append(vec_rew[idx].get(agent_id))
                dones.append(vec_done[idx].get(agent_id))
                counter += 1
        if counter >= batch_size:
            break
    return states, next_states, actions, probs, dones, rewards, total_agent_num, counter, ends, final_xs


def _cal_angle(vec):
    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        if abs(rel_pos_vec[0]) > 60 or abs(rel_pos_vec[1]) > 15:
            continue
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    return groups

# def evaluating(psgail, sap_size=10000, env_num=12, agent_number=10):
#     env_creator = lambda: MATrafficSimV(["./ngsim"], agent_number=agent_number)
#     vector_env = ParallelEnv([env_creator] * env_num, auto_reset=True)
#     vec_obs = vector_env.reset()
#     vec_done = []
#     states = []
#     acts = []
#     rewards = []
#     next_states = []
#     probs = []
#     dones = []
#     while True:
#         vec_act = []
#         for idx, obs in enumerate(vec_obs):
#             act_n = {}
#             obs_vectors = {}
#             for agent_id in obs.keys():
#                 if (getlist(vec_done, idx) is not None and vec_done[idx][agent_id]):
#                     continue
#                 obs_vectors[agent_id] = obs_extractor(obs[agent_id])
#                 states.append(obs_vectors)
#                 log_prob, prob, act_n[agent_id] = psgail.get_action(obs_vectors)
#                 acts.append(act_n[agent_id])
#                 probs.append(prob)
#             vec_act.append(act_n)
#         vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
#         for idx, obs in enumerate(vec_obs):
#             for agent_id in vec_act[idx].keys():
#                 obs_vectors = obs_extractor(vec_obs[idx].get(agent_id))
#                 next_states.append(obs_vectors)
#                 rewards.append(vec_rew[idx].get(agent_id))
#                 dones.append(vec_done[idx].get(agent_id))
#         if len(dones) >= sap_size:
#             break
#     vector_env.close()
#     return states, next_states, acts, probs, dones, rewards

# def assign_neighbors(neighbors, targets, relative_pos, idx):
#     if abs(relative_pos[0]) < abs(targets[1]):
#         targets[1] = relative_pos[0]
#         neighbors[1] = idx
#     elif targets[0] < relative_pos[0] < targets[1]:
#         targets[0] = relative_pos[0]
#         neighbors[0] = idx
#     elif targets[1] < relative_pos[0] < targets[2]:
#         targets[2] = relative_pos[0]
#         neighbors[2] = idx
#
#
# def obs_extractor(obs):
#     if obs is None:
#         return None
#     ego_vehicle_state = obs.ego_vehicle_state
#     neighborhood_vehicle_states = obs.neighborhood_vehicle_states
#     neighbors_up_idx = -np.ones(3).astype(int)
#     neighbors_middle_idx = -np.ones(3).astype(int)
#     neighbors_down_idx = -np.ones(3).astype(int)
#     neighbors_up = np.zeros((3, 4)).astype(float)
#     neighbors_middle = np.zeros((3, 4)).astype(float)
#     neighbors_down = np.zeros((3, 4)).astype(float)
#     center_lane = ego_vehicle_state.lane_index
#     targets_up = np.array([-10000, -10000, 10000])
#     targets_middle = np.array([-10000, 0, 10000])
#     targets_down = np.array([-10000, -10000, 10000])
#     for idx, info in enumerate(neighborhood_vehicle_states):
#         relative_pos = info[1][:-1] - ego_vehicle_state[1][:-1]
#         if info.lane_index == center_lane + 1:
#             assign_neighbors(neighbors_up_idx, targets_up, relative_pos, idx)
#         elif info.lane_index == center_lane:
#             assign_neighbors(neighbors_middle_idx, targets_middle, relative_pos, idx)
#         elif info.lane_index == center_lane - 1:
#             assign_neighbors(neighbors_down_idx, targets_down, relative_pos, idx)
#     for i in range(3):
#         idx_up = neighbors_up_idx[i]
#         idx_down = neighbors_down_idx[i]
#         # relative pos
#         if idx_up != -1:
#             neighbors_up[i, :2] = neighborhood_vehicle_states[idx_up][1][:-1] - ego_vehicle_state[1][:-1]
#             neighbors_up[i, 2] = float(neighborhood_vehicle_states[idx_up][3] - ego_vehicle_state[3])
#             neighbors_up[i, 3] = float(neighborhood_vehicle_states[idx_up][4] - ego_vehicle_state[4])
#         if idx_down != -1:
#             neighbors_down[i, :2] = neighborhood_vehicle_states[idx_down][1][:-1] - ego_vehicle_state[1][:-1]
#             # relative heading
#             neighbors_down[i, 2] = float(neighborhood_vehicle_states[idx_down][3] - ego_vehicle_state[3])
#             # relative speed
#             neighbors_down[i, 3] = float(neighborhood_vehicle_states[idx_down][4] - ego_vehicle_state[4])
#     for i in range(3):
#         if i != 1:
#             idx = neighbors_middle_idx[i]
#             if idx != -1:
#                 neighbors_middle[i, :2] = neighborhood_vehicle_states[idx][1][:-1] - ego_vehicle_state[1][:-1]
#                 neighbors_middle[i, 2] = float(neighborhood_vehicle_states[idx][3] - ego_vehicle_state[3])
#                 neighbors_middle[i, 3] = float(neighborhood_vehicle_states[idx][4] - ego_vehicle_state[4])
#     neighbors_middle = np.delete(neighbors_middle, 1, axis=0)
#     flatten_up = neighbors_up.flatten()
#     flatten_middle = neighbors_middle.flatten()
#     flatten_down = neighbors_down.flatten()
#     ego_v = np.zeros(13)
#     if len(obs.events.collisions) != 0:
#         ego_v[0] = 1
#     ego_v[1] = obs.events.off_road
#     ego_v[2] = obs.events.on_shoulder
#     # pos
#     ego_v[3:5] = ego_vehicle_state[1][:-1]
#     # heading
#     ego_v[5] = ego_vehicle_state[3]
#     # speed
#     ego_v[6] = ego_vehicle_state[4]
#     # linear speed
#     ego_v[7:13] = np.concatenate((ego_vehicle_state[11][:-1], ego_vehicle_state[12][:-1], ego_vehicle_state[13][:-1]))
#     obs_vectors = np.concatenate((flatten_up, flatten_middle, flatten_down, ego_v))
#     return obs_vectors
