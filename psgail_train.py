from psgail import *
from utils_psgail import *
import pickle as pk
import logging
import os
import time


def train(psgail, experts, stage, i_episode_res, num_episode=1000, print_every=10, gamma=0.95, batch_size=10000,
          agent_num=10, mini_epoch=3):
    rewards_log = []
    episodes_log = []
    dis_losses = []
    pol_losses = []
    val_losses = []
    # time1 = time.time()
    logger.info('agents num {}'.format(agent_num))
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
    vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
    # time2 = time.time()
    # print('env init', time2-time1)
    for i_episode in range(i_episode_res, num_episode):
        if i_episode % 200 == 0 and i_episode != 0:
            agent_num += 10
            vector_env.close()
            logger.info('adding agents to {}'.format(agent_num))
            env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
            vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
        if i_episode % 50 == 0 and i_episode != 0:
            logger.info('checkpoints establish, episode {}'.format(i_episode))
            with open('./models/psgail_' + str(int(i_episode / 50)) + '_' + stage + '.model',
                      "wb") as f:
                pk.dump(
                    {
                        'model': psgail,
                        'epoch': i_episode,
                        'rewards_log': rewards_log,
                        'episodes_log': episodes_log,
                        'agent_num': agent_num,
                        'stage': stage,
                        'dis_losses': dis_losses,
                        'pol_losses': pol_losses,
                        'val_losses': val_losses,
                    },
                    f,
                )
        # time1 = time.time()
        states, next_states, actions, probs, dones, rewards = sampling(psgail, vector_env, batch_size=batch_size)
        # time2 = time.time()
        # print('sampling complete, time cost', time2-time1, 's')
        rewards_log.append(np.mean(rewards))
        episodes_log.append(i_episode)
        batch = trans2tensor({"state": states, "action": actions,
                              "probs": probs,
                              "next_state": next_states, "done": dones,
                              "reward": rewards, })
        # batch["adv"] = psgail.compute_adv(batch, gamma)
        experts_sample = np.random.randint(0, high=len(experts), size=len(states))
        cur_experts = experts[experts_sample]
        # time1 = time.time()
        dis_buffer = []
        pol_buffer = []
        val_buffer = []
        for i in range(mini_epoch):
            discriminator_loss, policy_loss, value_loss = psgail.update_parameters(batch, cur_experts, gamma)
            dis_buffer.append(discriminator_loss)
            pol_buffer.append(policy_loss)
            val_buffer.append(value_loss)
        dis_losses.append(np.mean(dis_buffer))
        pol_losses.append(np.mean(pol_buffer))
        val_losses.append(np.mean(val_buffer))
        # time2 = time.time()
        # print('epoch training complete, time cost', time2 - time1, 's')
        if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
            logger.info("Episode: {}, Reward: {}".format(i_episode + 1, np.mean(rewards_log[-print_every:])))
            logger.info(
                "Episode: {}, pol_loss: {}".format(i_episode + 1, np.mean(pol_losses[-print_every:])))
            logger.info(
                "Episode: {}, val_loss: {}".format(i_episode + 1, np.mean(val_losses[-print_every:])))
            logger.info(
                "Episode: {}, dis_loss: {}".format(i_episode + 1, np.mean(dis_losses[-print_every:])))
    infos = {
        "rewards": rewards_log,
        "episodes": episodes_log,
        'pol_loss': pol_losses,
        'val_loss': val_losses,
        'dis_loss': dis_losses,
    }
    vector_env.close()
    return infos


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
    psgail = PSGAIL(discriminator_lr=5e-5,
                    policy_lr=1e-3,
                    value_lr=1e-3)
    experts = np.load('experts.npy')

    infos_1 = train(psgail, experts, 'train', 0, num_episode=train_episodes, print_every=10, gamma=0.95,
                    batch_size=10000,
                    agent_num=10, mini_epoch=3)

    plt.title('Reinforce training reward on {}'.format(env_name))
    plt.ylabel("Reward")
    plt.xlabel("Frame")
    infos = [infos_1, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["rewards"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('train_reward.jpg')
    plt.close()

    plt.title('Reinforce training policy loss on {}'.format(env_name))
    plt.ylabel("policy loss")
    plt.xlabel("Frame")
    infos = [infos_1, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["pol_loss"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('train_pol.jpg')
    plt.close()

    plt.title('Reinforce training value loss on {}'.format(env_name))
    plt.ylabel("value loss")
    plt.xlabel("Frame")
    infos = [infos_1, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["val_loss"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('train_val.jpg')
    plt.close()

    plt.title('Reinforce training discriminator loss on {}'.format(env_name))
    plt.ylabel("discriminator loss")
    plt.xlabel("Frame")
    infos = [infos_1, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["dis_loss"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('train_dis.jpg')
    plt.close()

    # fine tuning
    infos_2 = train(psgail, experts, 'tuning', 0, num_episode=tuning_episodes, print_every=10, gamma=0.99,
                    batch_size=40000,
                    agent_num=100, mini_epoch=1)

    plt.title('Reinforce tuning reward on {}'.format(env_name))
    plt.ylabel("Reward")
    plt.xlabel("Frame")
    infos = [infos_2, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["rewards"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('tuning_rew.jpg')
    plt.close()

    plt.title('Reinforce tuning policy loss on {}'.format(env_name))
    plt.ylabel("policy loss")
    plt.xlabel("Frame")
    infos = [infos_2, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["pol_loss"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('tuning_pol.jpg')
    plt.close()

    plt.title('Reinforce tuning value loss on {}'.format(env_name))
    plt.ylabel("value loss")
    plt.xlabel("Frame")
    infos = [infos_2, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["val_loss"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('tuning_val.jpg')
    plt.close()

    plt.title('Reinforce tuning discriminator loss on {}'.format(env_name))
    plt.ylabel("discriminator loss")
    plt.xlabel("Frame")
    infos = [infos_2, ]
    labels = ["PS-GAIL", ]
    for info in infos:
        x, y = info["episodes"], info["dis_loss"]
        y, x = moving_average(y, x)
        plt.plot(x, y)
    plt.legend(labels)
    plt.savefig('tuning_dis.jpg')
    plt.close()
