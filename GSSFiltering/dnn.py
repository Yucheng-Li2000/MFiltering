import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import configparser
import spdlayers
# from kan import *

config = configparser.ConfigParser()
config.read('./config.ini')

nGRU = int(config['DNN.size']['ngru'])
gru_scale_s = int(config['DNN.size']['gru_scale_s'])
gru_scale_k = int(config['DNN.size']['gru_scale_k'])

def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device

def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype

def _to_module_device(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return x.to(device=_module_device(module), dtype=_module_dtype(module))

def _zeros_like_module(module: nn.Module, *shape):
    return torch.zeros(*shape, device=_module_device(module), dtype=_module_dtype(module))

def _eye_like_module(module: nn.Module, n: int):
    return torch.eye(n, device=_module_device(module), dtype=_module_dtype(module))

class DNN_SKalmanNet_GSS(torch.nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 2):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # For NCLT, SyntheticNL (general)
        H1 = (x_dim + y_dim) * (10) * 8
        H2 = (x_dim * y_dim) * 1 * (4)

        # # For Time-Varyling
        # H1 = (x_dim + y_dim) * (5) * 4
        # H2 = (x_dim * y_dim) * 1 * (4)

        self.input_dim_1 = (self.x_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)
        self.input_dim_2 = (self.y_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)

        self.output_dim_1 = (self.x_dim * self.x_dim)
        self.output_dim_2 = (self.y_dim * self.y_dim)

        # input layer {x_k - x_{k-1}}
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim_1, H1),
            nn.ReLU()
        )

        # GRU 
        self.gru_input_dim = H1
        self.gru_hidden_dim = round(gru_scale_s * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)))
        self.gru_n_layer = nGRU
        self.batch_size = 1
        self.seq_len_input = 1

        self.hn1 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn1_init = self.hn1.detach().clone()
        self.GRU1 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> Pk
        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_1)
        )

        # input layer {residual}
        self.l3 = nn.Sequential(
            nn.Linear(self.input_dim_2, H1),
            nn.ReLU()
        )

        # GRU
        self.hn2 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn2_init = self.hn2.detach().clone()
        self.GRU2 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> Sk
        self.l4 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_2)
        )

    def initialize_hidden(self):
        self.hn1 = self.hn1_init.detach().clone().to(device=_module_device(self), dtype=_module_dtype(self))
        self.hn2 = self.hn2_init.detach().clone().to(device=_module_device(self), dtype=_module_dtype(self))

    def forward(self, state_inno, observation_inno, diff_state, diff_obs, linearization_error, Jacobian):
        input1 = torch.cat((state_inno, diff_state, linearization_error, Jacobian), axis=0).reshape(-1)
        input2 = torch.cat((observation_inno, diff_obs, linearization_error, Jacobian), axis=0).reshape(-1)

        l1_out = self.l1(input1)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim, device=l1_out.device, dtype=l1_out.dtype)
        GRU_in[0, 0, :] = l1_out
        GRU_out, self.hn1 = self.GRU1(GRU_in, self.hn1)
        l2_out = self.l2(GRU_out)
        Pk = l2_out.reshape((self.x_dim, self.x_dim))

        l3_out = self.l3(input2)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim, device=l3_out.device, dtype=l3_out.dtype)
        GRU_in[0, 0, :] = l3_out
        GRU_out, self.hn2 = self.GRU2(GRU_in, self.hn2)
        l4_out = self.l4(GRU_out)
        Sk = l4_out.reshape((self.y_dim, self.y_dim))

        return (Pk, Sk)


class DNN_KalmanNet_GSS(torch.nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 2):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        H1 = (x_dim + y_dim) * (10) * 8
        H2 = (x_dim * y_dim) * 1 * (4)

        self.input_dim = (self.x_dim * 2) + (self.y_dim * 2)
        self.output_dim = self.x_dim * self.y_dim

        # input layer
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim, H1),
            nn.ReLU()
        )

        # GRU
        self.gru_input_dim = H1
        self.gru_hidden_dim = gru_scale_k * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim))
        self.gru_n_layer = nGRU
        self.batch_size = 1
        self.seq_len_input = 1

        self.hn = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn_init = self.hn.detach().clone()
        self.GRU = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> kalman gain
        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim)
        )

    def initialize_hidden(self):
        self.hn = self.hn_init.detach().clone().to(device=_module_device(self), dtype=_module_dtype(self))

    def forward(self, state_inno, observation_inno, diff_state, diff_obs):
        device = _module_device(self)
        dtype = _module_dtype(self)
        state_inno = state_inno.to(device=device, dtype=dtype)
        observation_inno = observation_inno.to(device=device, dtype=dtype)
        diff_state = diff_state.to(device=device, dtype=dtype)
        diff_obs = diff_obs.to(device=device, dtype=dtype)
        input = torch.cat((state_inno, observation_inno, diff_state, diff_obs), axis=0).reshape(-1)
        l1_out = self.l1(input)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim, device=l1_out.device, dtype=l1_out.dtype)
        GRU_in[0, 0, :] = l1_out
        GRU_out, self.hn = self.GRU(GRU_in, self.hn)
        l2_out = self.l2(GRU_out)

        kalman_gain = torch.reshape(l2_out, (self.x_dim, self.y_dim))

        return (kalman_gain)


class KNet_architecture_v2(torch.nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 2, in_mult=20, out_mult=40):
        super().__init__()

        self.gru_num_param_scale = 1

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.gru_n_layer = 1
        self.batch_size = 1
        self.seq_len_input = 1  # Forward 전후로 처리해야할게 있으니 hidden 초기화를 직접 하고 시퀀스 길이를 1로!

        self.register_buffer('prior_Q', torch.eye(x_dim))
        self.register_buffer('prior_Sigma', torch.randn((x_dim, x_dim)))
        self.register_buffer('prior_S', torch.randn((y_dim, y_dim)))

        # GRU to track Q (5 x 5)
        self.d_input_Q = self.x_dim * in_mult
        self.d_hidden_Q = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Q)
        # self.h_Q_init = self.h_Q.detach().clone()

        # GRU to track Sigma (5 x 5)
        self.d_input_Sigma = self.d_hidden_Q + self.x_dim * in_mult
        self.d_hidden_Sigma = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Sigma)
        # self.h_Sigma_init = self.h_Sigma.detach().clone()

        # GRU to track S (2 x 2)
        self.d_input_S = self.y_dim ** 2 + 2 * self.y_dim * in_mult
        self.d_hidden_S = self.gru_num_param_scale * self.y_dim ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_S)
        # self.h_S_init = self.h_S.detach().clone()

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.y_dim ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.y_dim * self.x_dim
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.x_dim ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU())

        # Fully connected 5
        self.d_input_FC5 = self.x_dim
        self.d_output_FC5 = self.x_dim * in_mult
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.x_dim
        self.d_output_FC6 = self.x_dim * in_mult
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU())

        # Fully connected 7
        self.d_input_FC7 = 2 * self.y_dim
        self.d_output_FC7 = 2 * self.y_dim * in_mult
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU())

    def initialize_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()

    def forward(self, obs_diff, obs_innov_diff, state_evol_diff, state_update_diff):
        device = _module_device(self)
        dtype = _module_dtype(self)
        obs_diff = obs_diff.to(device=device, dtype=dtype)
        obs_innov_diff = obs_innov_diff.to(device=device, dtype=dtype)
        state_evol_diff = state_evol_diff.to(device=device, dtype=dtype)
        state_update_diff = state_update_diff.to(device=device, dtype=dtype)
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1], device=x.device, dtype=x.dtype)
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff.reshape((-1)))
        obs_innov_diff = expand_dim(obs_innov_diff.reshape((-1)))
        state_evol_diff = expand_dim(state_evol_diff.reshape((-1)))
        state_update_diff = expand_dim(state_update_diff.reshape((-1)))

        ####################
        ### Forward Flow ###
        ####################

        # FC 5
        in_FC5 = state_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = state_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
       
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2.reshape((self.x_dim, self.y_dim))

    
class Danse_net(torch.nn.Module):
    """
    RNN/GRU：输入观测序列 y，输出每步状态先验的均值与对角方差
    """
    def __init__(self, x_dim: int = 2, y_dim: int = 2):
        super().__init__()
        # 重要：输入应为观测维 y_dim，输出为状态维 x_dim
        self.input_size = y_dim
        self.output_size = x_dim

        self.hidden_dim = 32  
        self.num_layers = 1
        self.num_directions = 1
        self.batch_first = True

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers,
            batch_first=self.batch_first
        )

        # 全连接层：hidden -> 32 -> 均值/方差
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, 32)
        self.fc_mean = nn.Linear(32, self.output_size)
        self.fc_vars = nn.Linear(32, self.output_size)
    def initialize_hidden(self):
        # 若需要在外部reset，这里可以留空或设置一个标志位
        pass

    def init_h0(self, batch_size: int, device: torch.device):
        # 初始化 GRU 隐状态，全零
        return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, device=device, dtype=_module_dtype(self))

    def forward(self, x):
        x = x.to(device=_module_device(self), dtype=_module_dtype(self))
        """
        x: [batch, seq_len, y_dim]
        返回：
          mu:   [batch, seq_len, x_dim]
          vars: [batch, seq_len, x_dim]（对角方差，已softplus保证正）
        """
        batch_size = x.shape[0]
        device = x.device

        # 初始隐藏状态
        h0 = self.init_h0(batch_size, device)

        # RNN前向
        r_out, _ = self.rnn(x, h0)  # r_out: [batch, seq_len, hidden_dim]

        # 每步映射到均值/方差
        y = F.relu(self.fc(r_out))                 # [batch, seq_len, 32]
        mu_all = self.fc_mean(y)                   # [batch, seq_len, x_dim]
        vars_all = F.softplus(self.fc_vars(y))     # [batch, seq_len, x_dim]

        # t=1 的先验，用初始隐藏态近似（与DANSE的“右移一位”思想一致）
        h0_last = h0[-1, :, :]                     # [batch, hidden_dim]
        y0 = F.relu(self.fc(h0_last))              # [batch, 32]
        mu_1 = self.fc_mean(y0).view(batch_size, 1, -1)      # [batch, 1, x_dim]
        var_1 = F.softplus(self.fc_vars(y0)).view(batch_size, 1, -1)  # [batch, 1, x_dim]

        # 右移：用 mu_1 作为第一个时间步，其余用 r_out 的前 T-1 步
        if mu_all.size(1) > 1:
            mu = torch.cat((mu_1, mu_all[:, :-1, :]), dim=1)
            vars_ = torch.cat((var_1, vars_all[:, :-1, :]), dim=1)
        else:
            mu = mu_1
            vars_ = var_1
        return mu, vars_

class Latent_KalmanNet(nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 2, in_mult=10, out_mult=20):
        super().__init__()

        self.gru_num_param_scale = 1

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.gru_n_layer = 1
        self.batch_size = 1
        self.seq_len_input = 1  # Forward 전후로 처리해야할게 있으니 hidden 초기화를 직접 하고 시퀀스 길이를 1로!

        self.register_buffer('prior_Q', torch.eye(x_dim))
        self.register_buffer('prior_Sigma', torch.randn((x_dim, x_dim)))
        self.register_buffer('prior_S', torch.randn((y_dim, y_dim)))

        # GRU to track Q (5 x 5)
        self.d_input_Q = self.x_dim * in_mult
        self.d_hidden_Q = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Q)
        # self.h_Q_init = self.h_Q.detach().clone()

        # GRU to track Sigma (5 x 5)
        self.d_input_Sigma = self.d_hidden_Q + self.x_dim * in_mult
        self.d_hidden_Sigma = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Sigma)
        # self.h_Sigma_init = self.h_Sigma.detach().clone()

        # GRU to track S (2 x 2)
        self.d_input_S = self.y_dim ** 2 + 2 * self.y_dim * in_mult
        self.d_hidden_S = self.gru_num_param_scale * self.y_dim ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_S)
        # self.h_S_init = self.h_S.detach().clone()

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.y_dim ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.y_dim * self.x_dim
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.x_dim ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU())

        # Fully connected 5
        self.d_input_FC5 = self.x_dim
        self.d_output_FC5 = self.x_dim * in_mult
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.x_dim
        self.d_output_FC6 = self.x_dim * in_mult
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU())

        # Fully connected 7
        self.d_input_FC7 = 2 * self.y_dim
        self.d_output_FC7 = 2 * self.y_dim * in_mult
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU())

    def initialize_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()

    def forward(self, obs_diff, obs_innov_diff, state_evol_diff, state_update_diff):
        device = _module_device(self)
        dtype = _module_dtype(self)
        obs_diff = obs_diff.to(device=device, dtype=dtype)
        obs_innov_diff = obs_innov_diff.to(device=device, dtype=dtype)
        state_evol_diff = state_evol_diff.to(device=device, dtype=dtype)
        state_update_diff = state_update_diff.to(device=device, dtype=dtype)
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1], device=x.device, dtype=x.dtype)
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff.reshape((-1)))
        obs_innov_diff = expand_dim(obs_innov_diff.reshape((-1)))
        state_evol_diff = expand_dim(state_evol_diff.reshape((-1)))
        state_update_diff = expand_dim(state_update_diff.reshape((-1)))

        ####################
        ### Forward Flow ###
        ####################

        # FC 5
        in_FC5 = state_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = state_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
       
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2.reshape((self.x_dim, self.y_dim))

class MFiltering_Net(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        num_layers: int = 1,   
        device: str = 'cpu'
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.device = torch.device(device)
        self.num_layers = num_layers
        
        # ==================== 消融实验修改：f 网络输入去除了 y_dim ====================
        # NCLT/self build
        H_F_1 = x_dim * 16  # 去除了 y_dim 相关的维度加成
        H_F_2 = (x_dim * y_dim) * 8

        H_H_1 = (x_dim + y_dim) * 16
        H_H_2 = (x_dim * y_dim) * 8

        H_G_1 = (x_dim + y_dim) * 64
        H_G_2 = x_dim * y_dim * 16

        # Lorenz 
        # H_F_1 = x_dim * 32  # 去除了 y_dim 相关的维度加成
        # H_F_2 = (x_dim * y_dim) * 8

        # H_H_1 = (x_dim + y_dim) * 32
        # H_H_2 = (x_dim * y_dim) * 8

        # H_G_1 = (x_dim + y_dim) * 128
        # H_G_2 = x_dim * y_dim * 16
        
        self.K_fc = nn.Linear(H_G_2, x_dim * y_dim)

        self.input_norm = nn.LayerNorm(x_dim + y_dim)
        
        # ===== f_net (单步 LSTM) =====
        self.f_in = nn.Sequential(
            nn.Linear(x_dim, H_F_1), # 输入维度由 x_dim + y_dim 缩减为 x_dim
            nn.ReLU()
        )
        self.f_lstm = nn.LSTM(
            input_size=H_F_1, 
            hidden_size=H_F_2, 
            num_layers=num_layers, 
            batch_first=True,
        ).to(device)
        self.f_fc = nn.Linear(H_F_2, x_dim)

        # ===== h_net (单步 LSTM) =====
        self.h_in = nn.Sequential(
            nn.Linear(x_dim, H_H_1),
            nn.ReLU()
        )
        self.h_lstm = nn.LSTM(
            input_size=H_H_1, 
            hidden_size=H_H_2, 
            num_layers=num_layers, 
            batch_first=True
        ).to(device)
        self.h_fc = nn.Linear(H_H_2, y_dim)
        
        # ===== K 分支：学习卡尔曼增益 K_hat (双 Token LSTM) =====
        self.k_input_dim = x_dim + y_dim
        self.K_in = nn.Sequential(
            nn.Linear(self.k_input_dim, H_G_1),
            nn.ReLU()
        )
        self.K_lstm = nn.LSTM(
            input_size=H_G_1,
            hidden_size=H_G_2,
            num_layers=num_layers + 1,
            batch_first=True
        ).to(device)
        self.K_fc = nn.Linear(H_G_2, x_dim * y_dim)
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.initialize_hidden()

    def initialize_hidden(self):
        self.k_prev_token = None
        self.f_hidden = None 
        self.h_hidden = None 

    def _two_step_lstm(self, prev_token: torch.Tensor, cur_token: torch.Tensor, lstm: nn.LSTM):
        if prev_token is None:
            prev_token = torch.zeros_like(cur_token, device=cur_token.device, dtype=cur_token.dtype)
        seq = torch.stack([prev_token, cur_token], dim=1)
        y_out, _ = lstm(seq, None)
        return y_out[:, -1, :]

    def forward(self, x_prev: torch.Tensor, observation: torch.Tensor, P_prev: torch.Tensor):
        device = _module_device(self)
        dtype = _module_dtype(self)
        x_prev = x_prev.to(device=device, dtype=dtype)
        observation = observation.to(device=device, dtype=dtype)
        P_prev = P_prev.to(device=device, dtype=dtype)
        
        # ===== 1) 状态预测 f_net (消融修改：输入不包含 observation) =====
        token_f_in = x_prev.reshape(1, -1)                         # 去掉了 torch.cat 拼接
        token_f = self.f_in(token_f_in).unsqueeze(1)               # [1, 1, H1]
        
        x_last, self.f_hidden = self.f_lstm(token_f, self.f_hidden)
        x_delta = self.f_fc(x_last[:, -1, :]).T
        
        self.alpha.data.clamp_(0.0, 1.0)
        x_pred = self.alpha * x_prev + x_delta
        
        # x_pred = x_prev + x_delta


        # ===== 2) 观测预测 h_net (保持不变) =====
        h_hidden_prev = self.h_hidden
        
        token_h_in = x_pred.reshape(1, -1)
        token_h = self.h_in(token_h_in).unsqueeze(1)
        y_last, self.h_hidden = self.h_lstm(token_h, h_hidden_prev)
        y_pred = self.h_fc(y_last[:, -1, :]).T

        def h_wrapped(x_flat: torch.Tensor):
            x_vec = x_flat.view(self.x_dim, 1)
            token_stat = self.h_in(x_vec.reshape(1, -1)).unsqueeze(1)
            y_last_w, _ = self.h_lstm(token_stat, h_hidden_prev)
            y_out = self.h_fc(y_last_w[:, -1, :]).squeeze(0)
            return y_out

        x_pred_flat = x_pred.detach().view(-1).requires_grad_(True)
        H_J = torch.autograd.functional.jacobian(
            h_wrapped,
            x_pred_flat,
            create_graph=self.training,
            strict=True
        )
        H = H_J.view(self.y_dim, self.x_dim).to(self.device)
        
        x_delta_k = x_pred - x_prev
        residual = observation - y_pred  # observation 仅在残差（创新）计算时使用
        
        # ===== 3) K 分支：学习卡尔曼增益 K_hat (保持不变) =====
        k_input = torch.cat((
            x_delta_k,
            residual,
        ), dim=0).reshape(1, -1)
        k_token = self.K_in(k_input)
        k_last = self._two_step_lstm(self.k_prev_token, k_token, self.K_lstm)
        k_flat = self.K_fc(k_last).view(self.x_dim, self.y_dim)
        self.k_prev_token = k_token.detach()
        K_hat = k_flat
        
        return x_pred, y_pred, K_hat, H

