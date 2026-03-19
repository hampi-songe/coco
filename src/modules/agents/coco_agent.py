from audioop import bias
import torch.nn as nn
import torch
import torch.nn.functional as F
# from torch.optim import RMSprop, optimizer
# from torch.distributions import kl_divergence
# import torch.distributions as D
# from math import gamma

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.comm_gate = args.comm_gate
        self.msg_gate = args.msg_gate
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim * 2 + self.args.integrated_msg_dim * args.n_agents, args.n_actions)
        self.alpha = torch.zeros(1, args.n_agents, args.n_agents).detach().cuda()
        self.gate_net = nn.Linear(args.attention_dim * 2, 1).cuda()
        # alpha = self.alpha

        # self.msg_encoder = nn.Linear(2args.rnn_hidden_dim * , args.personal_msg_dim * 2)

        self.msg_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim * 2, args.nn_hidden_size),
            nn.BatchNorm1d(args.nn_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.nn_hidden_size, args.personal_msg_dim * 2),
            nn.LeakyReLU())

        self.inference_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.nn_hidden_size),
            nn.BatchNorm1d(args.nn_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.nn_hidden_size, args.personal_msg_dim * 2),
            nn.LeakyReLU())

        self.embed_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.nn_hidden_size),
            nn.BatchNorm1d(args.nn_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.nn_hidden_size, args.latent_dim * 2)
        )

        self.latent_state_encoder = nn.Sequential(
            nn.Linear(args.consensus_builder_embedding_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
        )

        # self.inference_net = nn.Linear(args.rnn_hidden_dim * 2,  args.personal_msg_dim * 2)

        self.q_net = nn.Linear(self.args.personal_msg_dim, self.args.attention_dim)
        self.k_net = nn.Linear(self.args.rnn_hidden_dim ,self.args.attention_dim)
        self.v_net = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.args.personal_msg_dim, args.nn_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.nn_hidden_size, self.args.integrated_msg_dim)
        )


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def calc_hidden(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h


    def calc_value(self, bs, latent_state, hidden_state, test_mode):
        # latent_state:[bs * n_agents, CB_embedding_dim]
        # latent_state_embedding:[bs * n_agents, rnn_hidden_dim]
        # hidden_state:[bs * n_agents, rnn_hidden_dim]
        latent_state_embedding = self.latent_state_encoder(latent_state)
        personal_msg, dis, inf_dis = self.calc_msg_dis(hidden_state, bs, latent_state_embedding, test_mode=test_mode)

        if self.args.use_comm:

            '''下面是计算每个agent要发送的信息'''
            # personal_msg = self.msg_encoder(hidden_state)# [bs * n_agents, msg_dim*2]
            # personal_msg = personal_msg[:,:self.args.personal_msg_dim]

            if self.args.attn_based_intergration:
                integrated_msg, CR_alpha = self.attn_message_integration(personal_msg, hidden_state, bs)
                if CR_alpha.shape[0] == 1:
                   self.alpha = CR_alpha
            else:
                integrated_msg = self.message_integration(personal_msg)

            x = torch.cat([hidden_state, latent_state_embedding, integrated_msg], dim=-1)

        else:
            x = torch.cat([hidden_state, latent_state_embedding], dim=-1)

        q = self.fc2(x) # [bs * n_agents, n_actions]

        return q, dis, inf_dis, personal_msg

    def calc_msg_dis(self, h, bs, latent_state_embedding, test_mode):
        '''Args:
            h: [bs * n_agents, rnn_hidden_dim]
            bs: 32
            latent_embed: [bs * n_agents, rnn_hidden_dim]

        Returns: personal_msg:[bs, n_agents, personal_msg_dim]'''

        latent_parameters = self.msg_encoder(torch.cat([h, latent_state_embedding.detach()], dim=-1))
        #print('latent_parameters',latent_parameters)

        mean = torch.clamp(latent_parameters[:, :self.args.personal_msg_dim], min=-1, max=1)
        # logstd = torch.ones(mean.shape).cuda()
        std = torch.clamp(latent_parameters[:, -self.args.personal_msg_dim:], min=0.5, max=1)

        dis = torch.distributions.Normal(mean, std)

        consensus_infer = self.inference_net(latent_state_embedding.detach()).view(bs * self.n_agents, -1)
        #print('consensus_infer',consensus_infer)
        inf_mean = torch.clamp(consensus_infer[:, :self.args.personal_msg_dim], min=-1, max=1)
        # inf_logstd = torch.ones(mean.shape).cuda()
        inf_std = torch.clamp(consensus_infer[:, -self.args.personal_msg_dim:], min=0.5, max=1)

        inf_dis = torch.distributions.Normal(inf_mean, inf_std)
        #dis, inf_dis: N( [bs*n_agents, msg_dim] , [bs* n_agents, msg_dim] )

        # print('dis', dis)
        # print('inf_dis', inf_dis)

        # latent_parameters[:, -self.args.personal_msg_dim:] = torch.clamp(
        #         torch.exp(latent_parameters[:, -self.args.personal_msg_dim:]),
        #         min=self.args.var_floor)
        # print('latent_parameters', latent_parameters.shape)

        # personal_msg = dis.rsample().softmax(dim=-1)# [bs*n_agents, msg_dim]
        # print('per_msg',personal_msg.shape)
        # latent_embed = latent_parameters.reshape(bs * self.n_agents, self.args.personal_msg_dim * 2)
        personal_msg = dis.rsample()
        # personal_msg = latent_embed[:, :self.args.personal_msg_dim]

        # gaussian_embed = D.Normal(latent_embed[:, :self.args.personal_msg_dim],
        #                               (latent_embed[:, self.args.personal_msg_dim:]) ** (1 / 2))
        # personal_msg = gaussian_embed.rsample()

        #personal_msg = personal_msg.reshape(bs * self.n_agents, self.args.personal_msg_dim)
        personal_msg = personal_msg.reshape(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).reshape(bs * self.n_agents *self.n_agents, -1)

        if test_mode:
            if self.args.comm_reduce:
               _, sorted_alpha = torch.topk(self.alpha, k=self.args.comm_gate, dim=-1, largest=False)
               #print('sorted_alpha',sorted_alpha.shape)
               expanded_sorted_alpha = sorted_alpha.unsqueeze(-1).repeat(1, 1, 1, self.args.msg_gate)
               #print('expanded_sorted_alpha', expanded_sorted_alpha.shape)
               clone_personal_msg = personal_msg[:, :self.msg_gate].clone().detach().reshape(bs, self.n_agents, self.n_agents, -1)
               # print('clone_personal_msg', clone_personal_msg.shape)
               # print('bs', bs)
               clone_personal_msg = clone_personal_msg.scatter(2, expanded_sorted_alpha, 0).reshape(bs * self.n_agents * self.n_agents, -1)
               #print('', sorted_alpha)
               # print(clone_personal_msg.shape)
               personal_msg[:, :self.msg_gate] = clone_personal_msg

               # personal_msg = personal_msg.reshape(-1, self.args.personal_msg_dim)
               # msg_gate_mask = torch.zeros_like(personal_msg[:, :self.comm_gate])
               # personal_msg[:, :self.comm_gate] = msg_gate_mask

        # consensus_infer = self.inference_net(consensus).view(bs * self.n_agents, -1)
        # consensus_infer[:, self.args.personal_msg_dim:] = torch.clamp(torch.exp(consensus_infer[:, self.args.personal_msg_dim:]), min=self.args.var_floor)

        personal_msg = personal_msg.reshape(bs, self.n_agents, self.n_agents, self.args.personal_msg_dim)

        return personal_msg, dis, inf_dis

    def attn_message_integration(self, personal_msg, hidden_state, bs):# [bs*n_agents, msg_dim*n_agents]

        integrated_msg = personal_msg.reshape(bs * self.n_agents * self.n_agents, -1)
        h_repeat = hidden_state.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents,-1)
        value = self.v_net(torch.cat([h_repeat, integrated_msg], dim=-1)).view(bs, self.n_agents, self.n_agents, self.args.integrated_msg_dim)
        # key = self.k_net(hidden_state).unsqueeze(1)
        # query = self.q_net(integrated_msg).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        #
        # alpha = torch.bmm(key / (self.args.attention_dim ** (1 / 2)), query).view(bs, self.n_agents, self.n_agents)
        #
        # for i in range(self.n_agents):
        #     alpha[:, i, i] = -1e9
        # alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)
        # 1) 基于余弦相似度的 attention
        # ------------------------------------------------------------
        # key_vecs: [bs, n_agents, D]
        key_vecs = self.k_net(hidden_state).view(bs, self.n_agents, -1)
        # query_vecs: [bs, n_agents, n_agents, D]
        query_vecs = (
            self.q_net(integrated_msg)
                .view(bs, self.n_agents, self.n_agents, -1)
        )
        # 扩展 key 形状以与 query 对齐，然后在最后一维上计算余弦相似度
        # result: [bs, n_agents, n_agents]
        alpha_cos = F.cosine_similarity(
            key_vecs.unsqueeze(2).expand(-1, -1, self.n_agents, -1),
            query_vecs,
            dim=-1
        )
        # ------------------------------------------------------------

        # 2) 基于“学习门控”（learnable gating）的 attention
        #    gate 输入拼接了 key 和 query，再过一个小网络产生 [0,1] 之间的门控权重
        # ------------------------------------------------------------
        # gate_input: [bs, n_agents, n_agents, 2*D]
        gate_input = torch.cat([
            key_vecs.unsqueeze(2).expand(-1, -1, self.n_agents, -1),
            query_vecs
        ], dim=-1)
        # print('gate_input: ', gate_input.shape)
        # gate_val: [bs, n_agents, n_agents, 1]  → squeeze 之后 [bs, n_agents, n_agents]
        gate_val = torch.sigmoid(self.gate_net(gate_input)).squeeze(-1)
        # ------------------------------------------------------------

        # 你可以任选其一作为最终 alpha：
        if self.args.alpha_cos:
            alpha = alpha_cos
        elif self.args.gate_val:
            key = self.k_net(hidden_state).unsqueeze(1)
            query = self.q_net(integrated_msg).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
            alpha = torch.bmm(key / (self.args.attention_dim ** (1 / 2)), query).view(bs, self.n_agents, self.n_agents)
            alpha = torch.mul(alpha, gate_val)
        else:
            key = self.k_net(hidden_state).unsqueeze(1)
            query = self.q_net(integrated_msg).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)

            alpha = torch.bmm(key / (self.args.attention_dim ** (1 / 2)), query).view(bs, self.n_agents, self.n_agents)

            for i in range(self.n_agents):
                alpha[:, i, i] = -1e9

        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        # 后续把 alpha 用在 softmax、加权或者其他地方即可
        # （例如，alpha = F.softmax(alpha, dim=-1)）
        integrated_msg = alpha * value

        integrated_msg = integrated_msg.reshape(bs, self.n_agents, self.n_agents, -1)

        for i in range(self.n_agents):
            integrated_msg[:, i, i, :] = 0

        integrated_msg = integrated_msg.reshape(bs * self.n_agents, -1)

        CR_alpha = alpha.squeeze(-1).detach()

        return integrated_msg, CR_alpha

    def message_integration(self, personal_msg):
        integrated_msg = personal_msg.reshape(-1, self.n_agents, self.args.personal_msg_dim).repeat(1, self.n_agents, 1).reshape(-1, self.n_agents * self.args.personal_msg_dim)
        return integrated_msg