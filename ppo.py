import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, MultivariateNormal
import math

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class NetCritic(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(NetCritic, self).__init__()
        layers = []
        last_size = input_size

        for i in range(layer_num - 1):
            layers.append(torch.nn.utils.spectral_norm(torch.nn.Linear(last_size, hidden_size[i])))
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(0.1))
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)
        self.noise = 0.1

    def forward(self, inputs):
        inputs += torch.normal(0, self.noise, size=inputs.shape, device=device)
        res = self._net(inputs)
        res = torch.sigmoid(res)
        return res


class NetAgent(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(NetAgent, self).__init__()
        layers = []
        last_size = input_size
        for i in range(layer_num - 1):
            layers.append(torch.nn.Linear(last_size, hidden_size[i]))
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(0.1))
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)

    def forward(self, inputs):
        res = self._net(inputs)
        return res


class PSGAIL():
    def __init__(self,
                 discriminator_lr=1e-4,
                 policy_lr=6e-4,
                 value_lr=3e-4,
                 hidden_size=[512, 256, 128, 64],
                 state_action_space=78,
                 state_space=76,
                 action_space=4,
                 ):
        self._tau = 0.01
        self._clip_range = 0.1
        self.kl_target = 0.01
        self.beta = 0.5
        self.v_clip_range = 0.2
        self.klmax = 0.15
        self.discriminator = NetCritic(hidden_size, state_action_space, output_size=1, layer_num=5)
        self.dis_crit = nn.BCELoss()
        self.value = NetAgent(hidden_size, state_space, output_size=1, layer_num=5)
        # self.target_value = NetAgent(hidden_size, state_space, output_size=1, layer_num=3)
        # self.target_policy = NetAgent(hidden_size, state_space, output_size=gru_hidden, layer_num=3)
        self.policy = NetAgent(hidden_size, state_space, output_size=action_space, layer_num=5)
        # self.policy =self.soft_update(self.target_policy, self.policy, 1)
        # self.policy = GRUNet(state_space, gru_hidden, gru_nums)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr,
                                                        betas=(0.5, 0.999), weight_decay=0.001)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr, weight_decay=0.001)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr, weight_decay=0.001)
        self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.discriminator_optimizer,
                                                                       step_size=2000,
                                                                       gamma=0.95)

    def get_action(self, obs, action=None):
        policy_out = self.policy(obs)
        mean1, var1, mean2, var2 = torch.chunk(policy_out, 4, dim=1)
        mean1 = 3 * torch.tanh(mean1)
        mean2 = 0.3 * torch.tanh(mean2)
        var1 = torch.nn.functional.softplus(var1)
        var2 = torch.nn.functional.softplus(var2)
        act1 = Normal(mean1, var1)
        act2 = Normal(mean2, var2)
        if action is None:
            action1 = act1.sample()
            action2 = act2.sample()
            log_prob1 = act1.log_prob(action1)
            log_prob2 = act2.log_prob(action2)
            log_prob = log_prob1 + log_prob2
            action1 = torch.clamp(action1, -6, 6)
            action2 = torch.clamp(action2, -0.6, 0.6)
            action = torch.cat((action1, action2), dim=1)
        else:
            log_prob1 = act1.log_prob(action[:, 0].unsqueeze(1))
            log_prob2 = act2.log_prob(action[:, 1].unsqueeze(1))
            log_prob = log_prob1 + log_prob2
        return action, log_prob.reshape(-1, 1)

    def compute_adv(self, batch, gamma):
        s = batch["state"]
        s1 = batch["next_state"]
        reward = batch['agents_rew']
        done = batch["done"].reshape(-1, 1)
        batch['old_v'] = self.value(s)
        with torch.no_grad():
            td_target = reward + gamma * self.value(s1) * (1 - done)
            adv = td_target - batch['old_v']
        return adv, td_target

    def update_discriminator(self, sap_agents, sap_experts):
        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        sap_agents = sap_agents.detach()
        D_expert = self.discriminator(sap_experts)
        D_agents = self.discriminator(sap_agents)
        experts_score = -torch.log(D_expert)
        agents_score = -torch.log(1 - D_agents)
        discriminator_loss = (agents_score + experts_score).mean()
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        self.scheduler_discriminator.step()
        return float(agents_score.mean().data), float(-torch.log(1 - D_expert).mean().data), float(discriminator_loss)

    def update_generator(self, batch):
        state = batch["state"]
        action = batch["action"]
        old_log_prob = batch["log_prob"]
        adv = batch["adv"]
        td_target = batch['td_target']
        # old_v = batch['old_v']
        # old_v = old_v.detach()

        act, log_prob = self.get_action(state, action)
        old_log_prob = old_log_prob.detach()
        old_log_prob = old_log_prob.unsqueeze(1)
        ip_sp = torch.exp(log_prob - old_log_prob)
        ip_sp_clip = torch.clamp(ip_sp, 1 - self._clip_range, 1 + self._clip_range)
        cur_prob = torch.exp(log_prob)
        kl_div = torch.nn.functional.kl_div(old_log_prob, cur_prob)
        policy_loss = -torch.mean(
            torch.min(ip_sp * adv.detach(), ip_sp_clip * adv.detach()))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        v = self.value(state)
        # clip_v = old_v + torch.clamp(v - old_v, -self.v_clip_range, self.v_clip_range)
        # v_max = torch.min(((v - td_target) ** 2), ((clip_v - td_target) ** 2))
        value_loss = torch.mean(F.mse_loss(v, td_target.detach()))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        return float(policy_loss.data), float(value_loss.data), float(kl_div.data)

    def behavior_clone(self, sap_experts):
        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        s_experts = sap_experts[:, :-2]
        a_experts = sap_experts[:, -2:]
        _, log_probs = self.get_action(s_experts, a_experts)
        log_probs = - torch.nn.functional.relu(-log_probs)
        policy_loss = -log_probs.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return float(log_probs.mean().data), float(policy_loss.data)
