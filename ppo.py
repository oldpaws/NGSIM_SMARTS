import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, MultivariateNormal

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class Net(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(Net, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num - 1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            layers.append(torch.nn.Dropout(p=0.1))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)

    def forward(self, inputs):
        res = self._net(inputs)
        # res = torch.sigmoid(res)
        return res


class NetBN(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(NetBN, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num - 1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)

    def forward(self, inputs):
        res = self._net(inputs)
        # res = torch.sigmoid(res)
        return res


# class GRUNet(torch.nn.Module):
#     def __init__(self, state_space, gru_hidden, gru_nums):
#         self.grus = nn.GRU(state_space, gru_hidden, gru_nums, bias=False, dropout=0, batch_first=True)
#         self.grus.to(device)
#
#     def forward(self, inputs, hidden=None):
#         return self.grus(inputs, hidden)

class PSGAIL():
    def __init__(self,
                 discriminator_lr=1e-5,
                 policy_lr=1e-2,
                 value_lr=5e-3,
                 hidden_size=128,
                 state_action_space=38,
                 state_space=36,
                 gru_nums=64,
                 gru_hidden=4,
                 ):
        self._tau = 0.01
        self._clip_range = 0.2
        self.lambda_gp = 2
        self.discriminator = Net(64, state_action_space, output_size=1)
        self.value = NetBN(hidden_size, state_space, output_size=1, layer_num=4)
        self.target_value = NetBN(hidden_size, state_space, output_size=1, layer_num=4)
        # self.target_policy = NetBN(hidden_size, state_space, output_size=gru_hidden, layer_num=3)
        self.policy = NetBN(hidden_size, state_space, output_size=gru_hidden, layer_num=4)
        # self.policy =self.soft_update(self.target_policy, self.policy, 1)
        # self.policy = GRUNet(state_space, gru_hidden, gru_nums)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr,
                                                        betas=(0, 0.9))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr,
                                                 betas=(0, 0.9))
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        # self.last_hidden = None

    def get_r(self, obs_action):
        return self.discriminator(obs_action)

    def get_action(self, obs, action=None):
        policy_out = self.policy(obs)
        mean, var = torch.chunk(policy_out, 2, dim=-1)
        mean[:, 0] = 3 * torch.tanh(mean[:, 0])
        mean[:, 1] = 0.25 * torch.tanh(mean[:, 1])
        var = torch.nn.functional.softplus(var)
        cov_mat = torch.diag_embed(var)
        act_dist = MultivariateNormal(mean, cov_mat)
        if action is None:
            # action1 = m1.sample()
            # action2 = m2.sample()
            # log_prob1 = m1.log_prob(action1)
            # log_prob2 = m2.log_prob(action2)
            # action = torch.cat((action1, action2), dim=1)
            action = act_dist.sample()
            log_prob = act_dist.log_prob(action)
        else:
            log_prob = act_dist.log_prob(action)
        prob = torch.exp(log_prob.squeeze()).unsqueeze(1)
        dist_entropy = act_dist.entropy()
        return prob.reshape(-1, 1), action, dist_entropy.reshape(-1, 1)

    def grad_penalty(self, agent_data, experts_data):
        alpha = torch.tensor(np.random.random(size=experts_data.shape), dtype=torch.float32).cuda()
        interpolates = (alpha * experts_data + ((1 - alpha) * agent_data)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def soft_update(self, source, target, tau=0.01):
        if tau is None:
            tau = self._tau
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def compute_adv(self, batch, gamma, reward):
        s = batch["state"]
        s1 = batch["next_state"]
        done = batch["done"].reshape(-1, 1)
        with torch.no_grad():
            td_target = reward + gamma * self.value(s1) * (1 - done)
            adv = td_target - self.value(s)
        return adv, td_target

    def update_discriminator(self, batch, sap_experts):
        s = batch["state"]
        a = batch["action"]
        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        sap_agents = torch.cat((s, a), dim=1)
        sap_agents = sap_agents.detach()
        D_expert = self.discriminator(sap_experts)
        D_agents = self.discriminator(sap_agents)
        grad_penalty = self.grad_penalty(sap_agents.data, sap_experts.data)
        discriminator_loss = D_agents.mean() - D_expert.mean() + grad_penalty
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        return float(D_agents.mean().data), float(D_expert.mean().data), float(grad_penalty.data), float(
            discriminator_loss)

    def update_generator(self, batch, gamma):
        s = batch["state"]
        a = batch["action"]
        s1 = batch["next_state"]
        done = batch["done"].reshape(-1, 1)
        old_prob = batch["probs"].reshape(-1, 1)

        sap_agents = torch.cat((s, a), dim=1)
        D_agents = self.discriminator(sap_agents)
        agents_rew = (D_agents - D_agents.mean()) / (D_agents.std() + 1e-8)
        adv, td_target = self.compute_adv(batch, gamma, agents_rew.detach())

        td_target = agents_rew.detach() + gamma * self.target_value(s1) * (1 - done)

        cur_prob, _, dist_entropy = self.get_action(s, a)
        old_prob = old_prob.detach()
        ip_sp = cur_prob / (old_prob + 1e-7)
        ip_sp_clip = torch.clamp(ip_sp, 1 - self._clip_range, 1 + self._clip_range)
        policy_loss = -torch.mean(torch.min(ip_sp * adv.detach(), ip_sp_clip * adv.detach()) + 0.01 * dist_entropy)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        value_loss = torch.mean(F.mse_loss(self.value(s), td_target.detach()))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.soft_update(self.value, self.target_value, self._tau)
        return float(policy_loss.data), float(value_loss.data)

    def ppo(self, batch, gamma):
        s = batch["state"]
        a = batch["action"]
        old_prob = batch["probs"].reshape(-1, 1)
        r = batch["reward"]
        batch["reward"] = (r - r.mean()) / (r.std() + 1e-8)
        r = batch["reward"].reshape(-1, 1)

        adv, td_target = self.compute_adv(batch, gamma, r)
        cur_prob, _, dist_entropy = self.get_action(s, a)
        old_prob = old_prob.detach()
        ip_sp = cur_prob / (old_prob + 1e-7)
        ip_sp_clip = torch.clamp(ip_sp, 1 - self._clip_range, 1 + self._clip_range)
        policy_loss = -torch.mean(torch.min(ip_sp * adv.detach(), ip_sp_clip * adv.detach()) + 0.01 * dist_entropy)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        value_loss = torch.mean(F.mse_loss(self.value(s), td_target.detach()))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.soft_update(self.value, self.target_value, self._tau)
        return float(policy_loss.data), float(value_loss.data)

    def behavior_clone(self, sap_experts):
        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        s_experts = sap_experts[:, :-2]
        a_agents = self.get_action(s_experts)
        a_experts = sap_experts[:, -2:]
        policy_loss = torch.mean(F.mse_loss(a_agents, a_experts.detach()))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return float(policy_loss.data)

