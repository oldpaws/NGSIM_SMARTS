from utils_psgail import *
import pickle as pk
import logging
import os
import time
from multiagent_traffic_simulator import MATrafficSim
from multiagent_traffic_simulator_orig import MATrafficSimOrig, MATrafficSimOrigV
from smarts.env.wrappers.parallel_env import ParallelEnv
from ppo import *


# def train_BC_GAIL(psgail, experts, i_episode_res, stage='gail', num_episode=10000, print_every=1, gamma=0.99,
#                   batch_size=10240,
#                   agent_num=2, mini_epoch=1):
#     logger.info('batch_size {}'.format(batch_size))
#     mini_m_epoch = 5
#     rewards_log = []
#     avg_survival_log = []
#     episodes_log = []m
#     dis_ag_rew = []
#     dis_ex_rew = []
#     dis_total_losses = []
#     pol_losses = []
#     val_losses = []
#     ends_rate = []
#     avg_finals = []
#     logger.info('stage: {}, agents num {}'.format(stage, agent_num))
#     env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
#     vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
#     for i_episode in range(i_episode_res, num_episode):
#         log_probs_buf = []
#         dist_entropy_buf = []
#         policy_loss_buf = []
#         if stage == 'bc':
#             bc_num = 5000
#             dis_rd_sample = np.random.randint(0, high=len(experts), size=bc_num)
#             cur_experts = experts[dis_rd_sample]
#             log_probs, dist_entropy, policy_loss = psgail.behavior_clone(cur_experts)
#             log_probs_buf.append(log_probs)
#             dist_entropy_buf.append(dist_entropy)
#             policy_loss_buf.append(policy_loss)
#             if (i_episode + 1) % 50 == 0:
#                 logger.info('log_prob: {}, entropy: {}, policy_loss: {}'.format(np.mean(log_probs_buf[-10:]),
#                                                                                 np.mean(dist_entropy_buf[-10:]),
#                                                                                 np.mean(policy_loss_buf[-10:])))
#             if (i_episode + 1) % 500 == 0:
#                 states, next_states, actions, log_probs, dones, rewards, total_agent_num, sap_num, ends, final_xs = sampling(
#                     psgail,
#                     vector_env,
#                     batch_size=10000)
#                 logger.info(
#                     "Stage: {}, Episode: {}, Reward: {}, survival: {}, end_rate: {}, final_pos: {}, log_prob: {}, entropy: {}, policy_loss: {}".format(
#                         stage,m
#                         i_episode + 1, np.sum(rewards) / total_agent_num, sap_num / total_agent_num,
#                         ends / total_agent_num, final_xs / total_agent_num,
#                         np.mean(log_probs_buf[-10:]), np.mean(dist_entropy_buf[-10:]), np.mean(policy_loss_buf[-10:])))
#         else:
#             if (i_episode + 1) % 200 == 0:
#                 agent_num += 10
#                 vector_env.close()
#                 logger.info('adding agents to {}'.format(agent_num))
#                 env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
#                 vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
#             dis_agent_buffer = []
#             dis_expert_buffer = []
#             dis_total_buffer = []
#             pol_buffer = []
#             val_buffer = []
#             # time1 = time.time()
#             states, next_states, actions, log_probs, dones, rewards, total_agent_num, sap_num, ends, final_xs = sampling(
#                 psgail,
#                 vector_env,
#                 batch_size=batch_size)
#             # time2 = time.time()
#             # print('sample time {}'.format(time2 - time1))
#             rewards_log.append(np.sum(rewards) / total_agent_num)
#             avg_survival_log.append(sap_num / total_agent_num)
#             ends_rate.append(ends / total_agent_num)
#             avg_finals.append(final_xs / total_agent_num)
#             episodes_log.append(i_episode)
#             batch = trans2tensor({"state": states, "action": actions,
#                                   "log_prob": log_probs,
#                                   "next_state": next_states, "done": dones})
#             sap_agents = torch.cat((batch["state"], batch["action"]), dim=1)
#             for i in range(mini_epoch):
#                 for sap_agent in sap_agents.chunk(mini_m_epoch, 0):
#                     dis_rd_sample = np.random.randint(0, high=len(experts), size=len(sap_agent))
#                     cur_experts = experts[dis_rd_sample]
#                     dis_agent_tmp, dis_expert_tmp, dis_total_tmp = psgail.update_discriminator(sap_agent, cur_experts)
#                     dis_agent_buffer.append(dis_agent_tmp)
#                     dis_expert_buffer.append(dis_expert_tmp)
#                     dis_total_buffer.append(dis_total_tmp)
#             dis_ag_rew.append(np.mean(dis_agent_buffer))
#             dis_ex_rew.append(np.mean(dis_expert_buffer))PSGAIL
#             dis_total_losses.append(np.mean(dis_total_buffer))
#             D_agents = psgail.discriminator(sap_agents)
#             batch["agents_rew"] = -torch.log(1 - D_agents)
#             batch['adv'], batch['td_target'] = psgail.compute_adv(batch, gamma)
#             for i in range(mini_epoch):
#                 for state, action, old_log_prob, adv, td_target in zip(batch["state"].chunk(mini_m_epoch, 0),
#                                                                        batch["action"].chunk(mini_m_epoch, 0),
#                                                                        batch["log_prob"].chunk(mini_m_epoch, 0),
#                                                                        batch["adv"].chunk(mini_m_epoch, 0),
#                                                                        batch['td_target'].chunk(mini_m_epoch, 0)):
#                     policy_tmp, value_tmp = psgail.update_generator(state, action, old_log_prob, adv, td_target)
#                     pol_buffer.append(policy_tmp)
#                     val_buffer.append(value_tmp)
#             pol_losses.append(np.mean(pol_buffer))
#             val_losses.append(np.mean(val_buffer))
#             if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
#                 if print_every < 10:
#                     prt = print_every
#                 else:
#                     prt = 10
#                 logger.info(
#                     "Stage: {}, Episode: {}, Reward: {}, survival: {}, end_rate: {}, final_pos: {}, pol_loss: {}, val_loss: {}, dis_ag_rew: {}, dis_ex_rew: {}, dis_total: {}".format(
#                         stage,
#                         i_episode + 1, np.mean(rewards_log[-prt:]), np.mean(avg_survival_log[-prt:]),
#                         np.mean(ends_rate[-prt:]), np.mean(avg_finals[-prt:]),
#                         np.mean(pol_losses[-prt:]),
#                         np.mean(val_losses[-prt:]),
#                         np.mean(dis_ag_rew[-prt:]),
#                         np.mean(dis_ex_rew[-prt:]),
#                         np.mean(dis_total_losses[-prt:])
#                     ))
#         if (i_episode + 1) % 50 == 0 or i_episode + 1 == num_episode:
#             logger.info('stage {}, checkpoints establish, episode {}'.format(stage, i_episode + 1))
#             with open('./models/psgail_' + str(int((1 + i_episode) / 50)) + '_' + stage + '.model',
#                       "wb") as f:
#                 pk.dump(
#                     {
#                         'model': psgail,
#                         'epoch': i_episode,
#                         'rewards_log': rewards_log,
#                         'episodes_log': episodes_log,
#                         'agent_num': agent_num,
#                         'stage': stage,
#                         'dis_ag_rew': dis_ag_rew,
#                         'dis_ex_rew': dis_ex_rew,
#                         'pol_losses': pol_losses,
#                         'val_losses': val_losses,
#                         'avg_finals': avg_finals,
#                         'ends_rate': ends_rate,
#                     },
#                     f,
#                 )
#     infos = {
#         "rewards_gail": rewards_log,
#         "episodes": episodes_log,
#         'pol_loss_gail': pol_losses,
#         'val_loss_gail': val_losses,
#         'dis_ag_rew_gail': dis_ag_rew,
#         'dis_ex_rew_gail': dis_ex_rew,
#         'avg_survival_time_gail': avg_survival_log,
#         'dis_total_losses_gail': dis_total_losses,
#         'avg_finals': avg_finals,
#         'ends_rate': ends_rate,
#     }
#     vector_env.close()
#     return infos


def train_BC_GAIL(psgail, experts, i_episode_res, stage='gail', num_episode=200000, print_every=1, gamma=0.95,
                  batch_size=2560,
                  agent_num=2, mini_epoch=40):
    logger.info('batch_size {}'.format(batch_size))
    critic_epoch = 5
    rewards_log = []
    avg_step_log = []
    episodes_log = []
    dis_ag_rew = []
    dis_ex_rew = []
    dis_total_losses = []
    pol_losses = []
    val_losses = []
    ends_rate = []
    avg_finals = []
    kl_log = []
    logger.info('stage: {}, agents num {}'.format(stage, agent_num))
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
    vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
    vector_env.seed(random.randint(1, 500))
    vec_obs = vector_env.reset()
    vec_done = []
    expert_trajectory = {}
    for i in range(12):
        expert_trajectory[i] = {}
    for i_episode in range(i_episode_res, num_episode):
        if (i_episode + 1) % 2000 == 0:
            agent_num += 5
            vector_env.close()
            logger.info('adding agents to {}'.format(agent_num))
            env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
            vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
            vector_env.seed(random.randint(1, 500))
            vec_obs = vector_env.reset()
            expert_trajectory = {}
            vec_done = []
            for i in range(12):
                expert_trajectory[i] = {}
        dis_agent_buffer = []
        dis_expert_buffer = []
        dis_total_buffer = []
        pol_buffer = []
        val_buffer = []
        kl_buffer = []
        # time1 = time.time()
        ends, t_a_n, done_agents_steps, final_xs, states, next_states, actions, log_probs, dones, rewards, vec_obs, vec_done, expert_trajectory = sampling(
            psgail, vector_env, batch_size, vec_obs, vec_done, expert_trajectory)
        # time2 = time.time()
        # print('sample time {}'.format(time2 - time1))
        rewards_log.append(np.sum(rewards) / t_a_n)
        avg_step_log.append(done_agents_steps / t_a_n)
        ends_rate.append(ends / t_a_n)
        avg_finals.append(final_xs / t_a_n)
        episodes_log.append(i_episode)
        batch = trans2tensor({"state": states, "action": actions,
                              "log_prob": log_probs,
                              "next_state": next_states, "done": dones})
        sap_agents = torch.cat((batch["state"], batch["action"]), dim=1)
        sap_agents = sap_agents.detach()
        for i in range(critic_epoch):
            dis_rd_sample = np.random.randint(0, high=len(experts), size=len(sap_agents))
            cur_experts = experts[dis_rd_sample]
            dis_agent_tmp, dis_expert_tmp, dis_total_tmp = psgail.update_discriminator(sap_agents, cur_experts)
            dis_agent_buffer.append(dis_agent_tmp)
            dis_expert_buffer.append(dis_expert_tmp)
            dis_total_buffer.append(dis_total_tmp)
        dis_ag_rew.append(np.mean(dis_agent_buffer))
        dis_ex_rew.append(np.mean(dis_expert_buffer))
        dis_total_losses.append(np.mean(dis_total_buffer))
        D_agents = psgail.discriminator(sap_agents)
        batch["agents_rew"] = -torch.log(1 - D_agents.detach())
        batch['adv'], batch['td_target'] = psgail.compute_adv(batch, gamma)
        for i in range(mini_epoch):
            policy_tmp, value_tmp, kl_div = psgail.update_generator(batch)
            pol_buffer.append(policy_tmp)
            val_buffer.append(value_tmp)
            kl_buffer.append(kl_div)
            if kl_div > psgail.klmax:
                break
        kl_log.append(np.mean(kl_buffer))
        pol_losses.append(np.mean(pol_buffer))
        val_losses.append(np.mean(val_buffer))
        if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
            if print_every < 10:
                prt = print_every
            else:
                prt = 10
            logger.info(
                "St: {}, Ep: {}, Rew: {}, ag_num: {}, time: {}, end: {}, final: {}, pol_l: {}, kl: {}, val_l: {}, ag_rew: {}, ex_rew: {}, dis_l: {}".format(
                    stage,
                    i_episode + 1, round(np.mean(rewards_log[-prt:]), 4), agent_num,
                    round(np.mean(avg_step_log[-prt:]), 4),
                    round(np.mean(ends_rate[-prt:]), 4),
                    round(np.mean(avg_finals[-prt:]), 4),
                    round(np.mean(pol_losses[-prt:]), 4),
                    round(np.mean(kl_log[-prt:]), 4),
                    round(np.mean(val_losses[-prt:]), 4),
                    round(np.mean(dis_ag_rew[-prt:]), 4),
                    round(np.mean(dis_ex_rew[-prt:]), 4),
                    round(np.mean(dis_total_losses[-prt:]), 4)
                ))
        if (i_episode + 1) % 25 == 0 or i_episode + 1 == num_episode:
            logger.info('stage {}, checkpoints establish, episode {}'.format(stage, i_episode + 1))
            with open('./models/psgail_' + str(int((1 + i_episode) / 50)) + '_' + stage + '.model',
                      "wb") as f:
                pk.dump(
                    {
                        'model': psgail,
                        'epoch': i_episode,
                        'rewards_log': rewards_log,
                        'episodes_log': episodes_log,
                        'agent_num': agent_num,
                        'stage': stage,
                        'dis_ag_rew': dis_ag_rew,
                        'dis_ex_rew': dis_ex_rew,
                        'pol_losses': pol_losses,
                        'val_losses': val_losses,
                        'avg_finals': avg_finals,
                        'ends_rate': ends_rate,
                    },
                    f,
                )
    infos = {
        "rewards_gail": rewards_log,
        "episodes": episodes_log,
        'pol_loss_gail': pol_losses,
        'val_loss_gail': val_losses,
        'dis_ag_rew_gail': dis_ag_rew,
        'dis_ex_rew_gail': dis_ex_rew,
        'avg_survival_time_gail': avg_step_log,
        'dis_total_losses_gail': dis_total_losses,
        'avg_finals': avg_finals,
        'ends_rate': ends_rate,
    }
    vector_env.close()
    return infos


def load_model(model2test):
    with open(model2test, "rb") as f:
        models = pk.load(f)
    return models


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    logfile = log_file_name
    handler = logging.FileHandler(logfile, mode='a+')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start print log")

    env_name = 'NGSIM SMARTS'
    train_episodes = 1000
    tuning_episodes = 200

    psgail = PSGAIL()
    # filename = 'psgail_1_gailre.model'
    # model = load_model(filename)
    # psgail = model['model']
    experts = np.load('experts_76.npy')
    # infos_1 = train_BC(psgail, experts, 0, stage='BC')

    # infos_1 = train_PPO(psgail, 0, num_episode=train_episodes, print_every=1, gamma=0.95, batch_size=10000)
    #
    infos_1 = train_BC_GAIL(psgail, experts, 0, stage='gail')

    for keys in infos_1:
        if keys != "episodes":
            plt.title('Reinforce training ' + keys + ' on {}'.format(env_name))
            plt.ylabel(keys)
            plt.xlabel("episodes")
            labels = ["PS-GAIL"]
            x, y = infos_1["episodes"], infos_1[keys]
            # y, x = moving_average(y, x)
            plt.plot(x, y)
            plt.legend(labels)
            plt.savefig('train_' + keys + '.jpg')
            plt.close()

    # fine tuning
    # infos_2 = train_BC_GAIL(psgail, experts, 0, stage='gail_tuning', num_episode=1000, print_every=1,
    #                         gamma=0.999, batch_size=10000, agent_num=100, mini_epoch=1)
    #
    # for keys in infos_2:
    #     if keys !="episodes"
    #         plt.title('Reinforce training ' + keys + ' on {}'.format(env_name))
    #         plt.ylabel(keys)
    #         plt.xlabel("Frame")
    #         infos = [infos_1, ]
    #         labels = ["PS-GAIL", ]
    #         for info in infos:
    #             x, y = info["episodes"], info[keys]
    #             # y, x = moving_average(y, x)
    #             plt.plot(x, y)
    #         plt.legend(labels)
    #         plt.savefig('train_' + keys + '.jpg')
    #         plt.close()
