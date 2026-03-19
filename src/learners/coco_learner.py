from asyncio.proactor_events import constants
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, optimizer
from torch.distributions import kl_divergence
import torch.distributions as D

class COCOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.comm_gamma = args.comm_gamma

        self.params = list(mac.parameters())
        self.consensus_builder_params = list(mac.consensus_builder_update_parameters())
        # self.device = th.device('cuda:0')
        self.last_target_update_episode = 0
        self.comm_gate = args.comm_gate

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.consensus_builder_optimiser = RMSprop(params=self.consensus_builder_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        inputs = self._build_inputs(batch)

        # Calculate estimated Q-Values
        mac_out = []
        hidden_states = []
        msg_list = []
        inf_list = []
        kl_list = []
        personal_msg_list = []

        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs, dis, inf_dis, personal_msg = self.mac.forward(batch, t=t)
            # print('dis',dis)
            # print('inf_dis',inf_dis)
            hidden_states.append(self.mac.hidden_states.view(self.args.batch_size, self.args.n_agents, -1))
            mac_out.append(agent_outs)
            personal_msg = personal_msg[:,:,1,:].squeeze(dim=2).reshape(-1, self.args.personal_msg_dim)
            msg_prob = dis.log_prob(personal_msg)
            inf_prob = inf_dis.log_prob(personal_msg)
            kl_loss = kl_divergence(dis, inf_dis).sum(dim=-1)
            # print('kl_loss', kl_loss.shape)
            kl_list.append(kl_loss)
            # print('msg_prob', msg_prob)
            # print('inf-prob', inf_prob)
            msg_list.append(msg_prob)
            inf_list.append(inf_prob)
            personal_msg = personal_msg.view(batch.batch_size, self.args.n_agents, -1)
            personal_msg_list.append(personal_msg)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        personal_msg_list = th.stack(personal_msg_list, dim=1)[:, :-1]
        hidden_states = th.stack(hidden_states, dim=1)
        # Entropy loss
        entropy_loss = (-D.Normal(self.s_mu, self.s_sigma).log_prob(personal_msg_list).sum() / mask.sum())

        kl_list = th.stack(kl_list, dim=1).reshape(self.args.batch_size, batch.max_seq_length, self.args.n_agents)
        msg_list = th.stack(msg_list, dim=1)[:,:-1].reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length-1, -1).permute(0,2,1,3).reshape(batch.batch_size, batch.max_seq_length-1, self.args.n_agents, -1)
        inf_list = th.stack(inf_list, dim=1)[:,:-1].reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length-1, -1).permute(0,2,1,3).reshape(batch.batch_size, batch.max_seq_length-1, self.args.n_agents, -1)
        # [bs*n_agents, max_seq_len, msg_dim]
        if self.args.comm_reduce:
            for bit in range(1, self.args.msg_gate):
                msg_list[:, :, :, -bit] = msg_list[:, :, :, -bit] * (self.comm_gamma ** (bit - 1))
                inf_list[:, :, :, -bit] = inf_list[:, :, :, -bit] * (self.comm_gamma ** (bit - 1))

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_hidden_states = []

        self.target_mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs, _, _, _ = self.target_mac.forward(batch, t=t)
            target_hidden_states.append(self.target_mac.hidden_states.view(self.args.batch_size, self.args.n_agents, -1))
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_hidden_states = th.stack(target_hidden_states, dim=1)
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.input == 'hidden':
            origin_obs = hidden_states[:, :-1].reshape(batch.batch_size * (batch.max_seq_length - 1) * self.args.n_agents, -1).detach()
        elif self.args.input == 'obs':
            origin_obs = inputs[:, :-1].reshape(batch.batch_size * (batch.max_seq_length - 1) * self.args.n_agents, -1)

        # alive_mask = th.ones(batch.batch_size, (batch.max_seq_length - 1), self.args.n_agents).cuda()
        alive_mask = batch['alive_allies'][:, :-1]

        alive_mask[:, 1:] = alive_mask[:, 1:] * (1 - terminated[:, :-1]) # [bs, max_seq_length, n_agents]

        valid_state_mask = (alive_mask.sum(-1, keepdim=True) > 0).flatten(start_dim=0, end_dim=1).bool()# [bs, max_seq_length, 1]

        valid_obs_mask = valid_state_mask.unsqueeze(-1).repeat([1, self.args.n_agents, 1]).flatten(0, 1).bool() # [bs * max_seq_length * n_agents, 1]

        alive_obs_mask = alive_mask.flatten(0, 2).bool().unsqueeze(-1) # [bs * max_seq_length * n_agents, 1]

        # 从原始观测中选出有效的观测: valid_obs:[2841, 42]
        # 从原始观测中选出活着的agent的观测: alive_obs:[1896, 42]
        valid_obs = th.masked_select(origin_obs, valid_obs_mask).view(-1, origin_obs.size()[-1])
        alive_obs = th.masked_select(origin_obs, alive_obs_mask).view(-1, origin_obs.size()[-1])

        obs_projection = self.mac.consensus_builder.calc_student(valid_obs)
        teacher_obs_projection = self.mac.consensus_builder.calc_teacher(valid_obs)
        real_teacher_obs_projection = self.mac.consensus_builder.calc_teacher(alive_obs)

        online_obs_prediction = obs_projection.view(-1, self.args.n_agents, self.args.consensus_builder_dim)
        teacher_obs_projection = teacher_obs_projection.view(-1, self.args.n_agents, self.args.consensus_builder_dim)
        real_teacher_obs_projection = real_teacher_obs_projection.view(-1, self.args.consensus_builder_dim).detach()

        # online_obs_prediction = online_obs_prediction - online_obs_prediction.max(dim=-1, keepdim=True)[0].detach()

        '''obs_center:[1, CB_dim]的零张量'''
        centering_teacher_obs_projection = teacher_obs_projection - self.mac.obs_center.detach()
        # centering_teacher_obs_projection = centering_teacher_obs_projection - centering_teacher_obs_projection.max(dim=-1, keepdim=True)[0].detach()

        online_obs_prediction_sharp = online_obs_prediction / self.args.online_temp
        target_obs_projection_z = F.softmax(centering_teacher_obs_projection / self.args.target_temp, dim=-1)

        # contrastive_loss:[bs * (max_seq_length-1), n_agents ,CB_dim] * [bs * (max_seq_length-1), CB_dim, n_agents]
        # =[bs * (max_seq_length-1), n_agents, n_agents]
        contrastive_loss = - th.bmm(target_obs_projection_z.detach(), F.log_softmax(online_obs_prediction_sharp, dim=-1).transpose(1, 2))

        contrastive_mask = th.masked_select(alive_mask.flatten().unsqueeze(-1), valid_obs_mask).view(-1, self.args.n_agents)
        contrastive_mask = contrastive_mask.unsqueeze(-1)
        contrastive_mask = th.bmm(contrastive_mask, contrastive_mask.transpose(1, 2))
        contrastive_mask = contrastive_mask * (1 - th.diag_embed(th.ones(self.args.n_agents))).unsqueeze(0).to(contrastive_mask.device)
        # [879, 3, 3]

        contrastive_loss = (contrastive_loss * contrastive_mask).sum() / contrastive_mask.sum()
        # print('contrastive_loss:', contrastive_loss)

        # Optimise
        self.consensus_builder_optimiser.zero_grad()
        contrastive_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.consensus_builder_params, self.args.grad_norm_clip)
        self.consensus_builder_optimiser.step()

        self.mac.obs_center = (self.args.center_tau * self.mac.obs_center + (1 - self.args.center_tau) * real_teacher_obs_projection.mean(0, keepdim=True)).detach()
        self.mac.consensus_builder.update()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("co_loss", contrastive_loss.item(), t_env)

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # mixing_state_projection:[bs, max_seq_length, n_agents, CB_dim]
        with th.no_grad():
            if self.args.input == 'hidden':
                mixing_state_projection = self.mac.consensus_builder.calc_student(hidden_states)
            elif self.args.input == 'obs':
                mixing_state_projection = self.mac.consensus_builder.calc_student(inputs)

            # mixing_state_projection = mixing_state_projection - mixing_state_projection.max(-1, keepdim=True)[0].detach()

            # mixing_state_projection_z:归一化以后的student的概率分布P_s(z):[bs, max_seq_length, n_agents, CB_dim]
            mixing_state_projection_z = F.softmax(mixing_state_projection / self.args.online_temp, dim=-1)

            # 加上alive-mask,[bs, max_seq_length, n_agents, CB_dim]
            mixing_state_projection_z = mixing_state_projection_z * batch['alive_allies'].unsqueeze(-1) # [32, 61, 3, 4]
            # print('mixing_state_projection_z',mixing_state_projection_z.shape)

            # 潜在状态的标识：[bs, max_seq_length]
            latent_state_id = mixing_state_projection_z.sum(-2).detach().max(-1)[1] # [32, 61]
            # print('latent_state-id',latent_state_id.shape)

            # 潜在状态的one-hot编码[bs, max_seq_length, CB_dim]
            latent_state_onehot = th.zeros(*latent_state_id.size(), self.args.consensus_builder_dim).cuda().scatter_(-1, latent_state_id.unsqueeze(-1), 1)

            # 将latent_state_id进行embedding：[bs, max_seq_length, n_agents, embedding_dim]
            latent_state_embedding = self.mac.embedding_net(latent_state_id)# [32, 61, 4]
            # print('latent_state_embedding', latent_state_embedding.shape)

            # 潜在状态标识的计数
            latent_state_id_count = ((latent_state_onehot[:, :-1]).sum([0, 1]) > 0).sum().float()

            if self.args.input == 'hidden':
                target_mixing_state_projection = self.target_mac.consensus_builder.calc_student(target_hidden_states)
            elif self.args.input == 'obs':
                target_mixing_state_projection = self.target_mac.consensus_builder.calc_student(inputs)
            # target_mixing_state_projection = target_mixing_state_projection - target_mixing_state_projection.max(-1, keepdim=True)[0].detach()
            target_mixing_state_projection_z = F.softmax(target_mixing_state_projection / self.args.online_temp, dim=-1)
            target_mixing_state_projection_z = target_mixing_state_projection_z * batch['alive_allies'].unsqueeze(-1)
            target_latent_state_id = target_mixing_state_projection_z.sum(-2).detach().max(-1)[1]
            target_latent_state_embedding = self.target_mac.embedding_net(target_latent_state_id)#[32, 61, 4]
            # print('latent_state_embedding', latent_state_embedding.shape)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch['state'][:, :-1])# batch['state']:[32, 61, 48]
            # print('state:', batch['state'].shape)
            target_max_qvals = self.target_mixer(target_max_qvals, batch['state'][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        rl_mask = mask.expand_as(td_error)# mask:[32, 60, 1]
        masked_td_error = td_error * rl_mask
        rl_loss = (masked_td_error ** 2).sum() / rl_mask.sum()

        bool_alive_mask = alive_mask.bool()

        masked_kl_loss = th.masked_select(kl_list[:, :-1], bool_alive_mask)
        masked_msg = th.masked_select(msg_list, bool_alive_mask.unsqueeze(dim=-1).repeat(1,1,1,self.args.personal_msg_dim))
        masked_inf = th.masked_select(inf_list, bool_alive_mask.unsqueeze(dim=-1).repeat(1,1,1,self.args.personal_msg_dim))
        masked_mi_loss = (masked_inf - masked_msg).sum() * self.args.mi_loss_weight / alive_mask.sum()
        masked_kl_loss = masked_kl_loss.sum() * self.args.mi_loss_weight / alive_mask.sum()
        loss = (th.clip(masked_kl_loss, 0, 10) + th.clip(masked_mi_loss, -10, 10)) * self.args.msg_loss_weight
        loss += rl_loss
        beta_ent = self.get_comm_entropy_beta(t_env)
        loss = loss + beta_ent * entropy_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self.target_mac.load_consensus_builder_state(self.mac)
            self._update_targets()
            self.last_target_update_episode = episode_num
            # print('personal_msg:', msg_list[-1, -1, :, :])

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("mi_loss", masked_mi_loss.mean().item(), t_env)
            self.logger.log_stat("kl_loss", masked_kl_loss.mean().item(), t_env)
            self.logger.log_stat("latent_state_id_count", latent_state_id_count.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("entropy_loss", beta_ent * entropy_loss.mean().item(), t_env)
            self.log_stats_t = t_env



    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")


    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.s_mu = th.zeros(1).cuda()
        self.s_sigma = th.ones(1).cuda()
        # self.comm_gamma = self.args.comm_gamma.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


    def _build_inputs(self, batch):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, :batch.max_seq_length])  # b1av
        if self.args.obs_last_action:
            inputs.append(th.cat([
                th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1), batch["actions_onehot"][:, :batch.max_seq_length-1]
            ], dim=1))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, batch.max_seq_length, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def get_comm_entropy_beta(self, t_env):
        comm_entropy_beta = self.args.comm_entropy_beta
        if self.args.is_comm_entropy_beta_decay and t_env > self.args.comm_entropy_beta_start_decay:
            comm_entropy_beta += 1. * (self.args.comm_entropy_beta_target - self.args.comm_entropy_beta) / \
                                 (self.args.comm_entropy_beta_end_decay - self.args.comm_entropy_beta_start_decay) * \
                                 (t_env - self.args.comm_entropy_beta_start_decay)
        return comm_entropy_beta

