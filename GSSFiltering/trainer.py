import math
import time
from typing import Union
import torch
import torch.optim as optim
import numpy as np
from GSSFiltering.filtering import KalmanNet_Filter, Split_KalmanNet_Filter
from GSSFiltering.filtering import KalmanNet_Filter_v2, Model_structured_filtering
from GSSFiltering.filtering import Unsupervised_DANSE
from GSSFiltering.filtering import Latent_KalmanNet_Filter
from GSSFiltering.tester import Tester
import os
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

if not os.path.exists('./.model_saved'):
    os.mkdir('./.model_saved')

print_num = 25
save_num = int(config['Train']['valid_period'])

lr_split = float(config['Train.Split']['learning_rate'])
lr_kalman = float(config['Train.Kalman']['learning_rate'])

wd_split = float(config['Train.Split']['weight_decay'])
wd_kalman = float(config['Train.Kalman']['weight_decay'])
train_iter = int(config['Train']['train_iter'])
lr_danse = float(config['Train.DANSE']['learning_rate'])


def mse(target, predicted):
    return torch.mean((target - predicted) ** 2)


def empirical_averaging_all(target, predicted_mean, predicted_cov, beta=0.05):
    L1 = mse(target, predicted_mean)
    err = target - predicted_mean
    n_batch = predicted_cov.shape[0]
    n_time = predicted_cov.shape[1]
    E_cov = torch.tensor(0.0, device=target.device)

    for i in range(n_time):
        for j in range(n_batch):
            E_cov += torch.sum(
                torch.abs(
                    err[j, :, i:i + 1] @ err[j, :, i:i + 1].T - predicted_cov[j, i, :]
                )
            )

    L2 = E_cov / (n_batch * n_time)
    return (1 - beta) * L1 + beta * L2, L1, L2


class Trainer():
    def __init__(
        self,
        dnn: Union[
            KalmanNet_Filter,
            Split_KalmanNet_Filter,
            KalmanNet_Filter_v2,
            Model_structured_filtering
        ],
        data_path,
        save_path,
        mode=0,
        device='cpu'
    ):
        self.device = torch.device(device)
        self.save_num = save_num
        self.P_history = []
        self.dnn = dnn
        self.log_sigma_rec = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.rot_warmup_steps = 5000

        self.x_dim = self.dnn.x_dim
        self.y_dim = self.dnn.y_dim
        self.data_path = data_path
        self.save_path = save_path
        self.mode = mode
        self.mse_ema = None
        self.cov_ema = None
        self.loss_best = 1e4

        self.train_forward_time_ms = []
        self.last_train_forward_ms = float('nan')

        # ===================== Load data to device =====================
        self.data_x = torch.load(data_path + 'state.pt', map_location=self.device, weights_only=True).to(self.device)
        self.data_y = torch.load(data_path + 'obs.pt', map_location=self.device, weights_only=True).to(self.device)

        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]

        assert self.x_dim == self.data_x.shape[1]
        assert self.y_dim == self.data_y.shape[1]
        assert self.seq_len == self.data_y.shape[2]
        assert self.data_num == self.data_y.shape[0]

        # ===================== Move dnn components to device =====================
        self._move_dnn_to_device()

        # ===================== Optimizers =====================
        if self.mode == 0:
            self.loss_fn = torch.nn.MSELoss()

            if isinstance(self.dnn, Model_structured_filtering):
                params = [
                    {'params': [p for n, p in self.dnn.kf_net.named_parameters() if n != 'alpha']},
                    {'params': [self.dnn.kf_net.alpha], 'lr': lr_danse * 10}
                ]
                self.optimizer = torch.optim.Adam(params, lr=lr_danse, weight_decay=wd_kalman)
            else:
                params = list(self.dnn.kf_net.parameters())
                self.optimizer = torch.optim.Adam(params, lr=lr_kalman, weight_decay=wd_kalman)
            # NCLT
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
            # Lorenz
            # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        elif self.mode == 1:
            self.loss_fn = torch.nn.SmoothL1Loss()
            kf = self.dnn.kf_net

            if all(hasattr(kf, n) for n in ['l1', 'GRU1', 'l2', 'l3', 'GRU2', 'l4']):
                self.network1 = [kf.l1, kf.GRU1, kf.l2]
                self.network2 = [kf.l3, kf.GRU2, kf.l4]
            else:
                raise NotImplementedError("kf_net structure not supported, missing required modules")

            param_group_1 = []
            for elem in self.network1:
                elem.to(self.device)
                param_group_1 += [{'params': elem.parameters()}]

            param_group_2 = []
            for elem in self.network2:
                elem.to(self.device)
                param_group_2 += [{'params': elem.parameters()}]

            self.optimizer_list = [
                torch.optim.Adam(param_group_1, lr=lr_split, weight_decay=wd_split),
                torch.optim.Adam(param_group_2, lr=lr_split, weight_decay=wd_split)
            ]
            self.unfreeze_net_current = 1

        else:
            self.loss_fn = torch.nn.MSELoss()

            self.optimizer_enc = torch.optim.Adam(
                self.dnn.encoder.parameters(),
                lr=lr_kalman,
                weight_decay=wd_kalman
            )

            self.optimizer_kf = torch.optim.Adam(
                self.dnn.kf_net.parameters(),
                lr=lr_kalman,
                weight_decay=wd_kalman
            )

            self.scheduler_enc = optim.lr_scheduler.ExponentialLR(self.optimizer_enc, gamma=0.99)
            self.scheduler_kf = optim.lr_scheduler.ExponentialLR(self.optimizer_kf, gamma=0.99)

        self.batch_size = int(config['Train']['batch_size'])
        self.alter_num = int(config['Train.Split']['alter_period'])

        self.train_count = 0
        self.data_idx = 0

    def _move_dnn_to_device(self):
        if hasattr(self.dnn, "kf_net"):
            self.dnn.kf_net.to(self.device)

        if hasattr(self.dnn, "encoder"):
            self.dnn.encoder.to(self.device)

        tensor_attrs = [
            "init_state",
            "state_post",
            "state_post_past",
            "state_pred_past",
            "obs_past",
            "x_post",
            "x_post_prev",
            "P_post",
            "cov_post",
            "Q",
            "R",
        ]

        for attr in tensor_attrs:
            if hasattr(self.dnn, attr):
                value = getattr(self.dnn, attr)
                if torch.is_tensor(value):
                    setattr(self.dnn, attr, value.to(self.device))

        if hasattr(self.dnn, "state_history") and torch.is_tensor(self.dnn.state_history):
            self.dnn.state_history = self.dnn.state_history.to(self.device)

        if hasattr(self.dnn, "cov_trace_history") and torch.is_tensor(self.dnn.cov_trace_history):
            self.dnn.cov_trace_history = self.dnn.cov_trace_history.to(self.device)

    def _sync_cuda(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def train_batch(self):
        if self.mode == 0:
            forward_ms = self.train_batch_joint()
        elif self.mode == 1:
            if isinstance(
                self.dnn,
                (
                    Split_KalmanNet_Filter,
                )
            ):
                forward_ms = self.train_batch_alternative()
            else:
                raise NotImplementedError()
        elif self.mode == 2:
            forward_ms = self.train_batch_latent_alternating()
        else:
            raise NotImplementedError()

        self.last_train_forward_ms = float(forward_ms)
        self.train_forward_time_ms.append(self.last_train_forward_ms)
        return self.last_train_forward_ms

    def _get_batch(self):
        if self.data_idx + self.batch_size >= self.data_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(self.data_x.shape[0], device=self.device)
            self.data_x = self.data_x[shuffle_idx]
            self.data_y = self.data_y[shuffle_idx]

        batch_x = self.data_x[self.data_idx: self.data_idx + self.batch_size].to(self.device)
        batch_y = self.data_y[self.data_idx: self.data_idx + self.batch_size].to(self.device)
        return batch_x, batch_y

    def train_batch_alternative(self):
        if self.train_count > 0 and self.train_count % self.alter_num == 0:
            if self.unfreeze_net_current == 1:
                self.unfreeze_net_current = 2
            elif self.unfreeze_net_current == 2:
                self.unfreeze_net_current = 1

        self.optimizer = self.optimizer_list[self.unfreeze_net_current - 1]
        self.optimizer.zero_grad(set_to_none=True)

        batch_x, batch_y = self._get_batch()

        self._sync_cuda()
        t_fwd0 = time.perf_counter()

        x_hat = torch.zeros_like(batch_x, device=self.device)
        cov_hat = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.x_dim,
            self.x_dim,
            device=self.device
        )
        cov_hat_diag = torch.zeros(
            self.batch_size,
            self.x_dim,
            self.seq_len,
            device=self.device
        )

        for i in range(self.batch_size):
            self.dnn.state_post = batch_x[i, :, 0].reshape((-1, 1)).to(self.device)

            for ii in range(1, self.seq_len):
                self.dnn.filtering(batch_y[i, :, ii].reshape((-1, 1)).to(self.device))

            x_hat[i] = self.dnn.state_history[:, -self.seq_len:].to(self.device)

            if hasattr(self.dnn, 'cov_history'):
                cov_hat[i] = self.dnn.cov_history[-self.seq_len:, :, :].to(self.device)
                cov_hat_diag[i] = torch.diagonal(cov_hat[i], dim1=-2, dim2=-1).T

            self.dnn.reset(clean_history=False)
            self._move_dnn_to_device()


        loss = self.loss_fn(batch_x[:, :, 1:], x_hat[:, :, 1:])

        self._sync_cuda()
        forward_ms = (time.perf_counter() - t_fwd0) * 1000.0

        if not torch.isfinite(loss):
            # print(f"[Warning] Non-finite loss at train {self.train_count}: {loss}")
            self.optimizer.zero_grad(set_to_none=True)
            self.train_count += 1
            self.data_idx += self.batch_size
            return forward_ms

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dnn.kf_net.parameters(), 10)
        self.optimizer.step()

        self.train_count += 1
        self.data_idx += self.batch_size

        if self.train_count % save_num == 0:
            save_dict = {'kf_net': self.dnn.kf_net.state_dict()}
            torch.save(
                save_dict,
                './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt'
            )

        return forward_ms

    def train_batch_joint(self):
        self.optimizer.zero_grad(set_to_none=True)

        batch_x, batch_y = self._get_batch()

        self._sync_cuda()
        t_fwd0 = time.perf_counter()

        x_hat = torch.zeros_like(batch_x, device=self.device)
        y_hat = torch.zeros_like(batch_y, device=self.device)
        v_hat = torch.zeros_like(batch_x, device=self.device)
        nll_danse_sum = (
            torch.tensor(0.0, device=self.device)
            if isinstance(self.dnn, Unsupervised_DANSE)
            else None
        )
        for i in range(self.batch_size):
            self.dnn.state_post = batch_x[i, :, 0].reshape((-1, 1)).to(self.device)

            for ii in range(1, self.seq_len):
                y_obs = batch_y[i, :, ii].reshape((-1, 1)).to(self.device)
                if isinstance(self.dnn, Unsupervised_DANSE):
                    Yi_prefix = batch_y[i, :, :ii].permute(1, 0).unsqueeze(0).to(self.device)

                    mu_x_prev_seq, L_x_prev_seq, mu_x_cur_seq, L_x_cur_seq = self.dnn.filtering(Yi_prefix)

                    mu_x_prev_t = mu_x_prev_seq[:, -1, :].reshape(self.x_dim, 1).to(self.device)
                    L_x_prev_t = L_x_prev_seq[:, -1, :, :].squeeze(0).to(self.device)

                    H_t = self.dnn.GSSModel.Jacobian_g(mu_x_prev_t).to(self.device)
                    mu_y_t = self.dnn.GSSModel.g(mu_x_prev_t).to(self.device)

                    Sigma_y_t = H_t @ L_x_prev_t @ H_t.T + self.dnn.R.to(self.device)
                    Sigma_y_t = Sigma_y_t + 1e-6 * torch.eye(self.y_dim, device=self.device)

                    diff = y_obs - mu_y_t

                    inv_Sigma = torch.linalg.inv(Sigma_y_t)
                    logdet_Sigma = torch.logdet(Sigma_y_t)

                    nll_t = 0.5 * (
                        logdet_Sigma
                        + (diff.T @ inv_Sigma @ diff).squeeze()
                        + self.y_dim * math.log(2 * math.pi)
                    )
                    nll_danse_sum = nll_danse_sum + nll_t
                    y_hat[i, :, ii] = mu_y_t.squeeze()
                elif isinstance(self.dnn, Model_structured_filtering):
                    # 2. 提取当前时刻 t 的真实状态值 x_t (Ground Truth)
                    x_true_t = batch_x[i, :, ii].reshape((-1, 1)).to(self.device)
                    
                    # 3. 将真实状态传入 filtering 触发 NEES 自适应调整
                    self.dnn.filtering(y_obs, true_x=x_true_t)
                else:
                    self.dnn.filtering(batch_y[i, :, ii].reshape((-1, 1)).to(self.device))

            x_hat[i] = self.dnn.state_history[:, -self.seq_len:].to(self.device)

            if hasattr(self.dnn, "reset"):
                self.dnn.reset(clean_history=False)
                self._move_dnn_to_device()


        if isinstance(self.dnn, Unsupervised_DANSE):
            denom = float(self.batch_size * (self.seq_len - 1))
            loss = nll_danse_sum / denom
        elif isinstance(self.dnn, Model_structured_filtering):
            x_true = batch_x[:, :, 1:].to(self.device)
            x_est = x_hat[:, :, 1:].to(self.device)
            diff = x_true - x_est

            mse_loss = (diff ** 2).mean()

            diff_for_cov = diff.squeeze(0)
            diff_cov_list = []

            for t in range(diff_for_cov.shape[1]):
                diff_t = diff_for_cov[:, t].unsqueeze(1)
                cov_t = diff_t @ diff_t.T
                diff_cov_list.append(cov_t.unsqueeze(0))

            diff_cov = torch.cat(diff_cov_list, dim=0).to(self.device)

            # ================= 核心修改区域 =================
            P_est_raw = getattr(self.dnn, 'P_history', None)
            
            if P_est_raw is None or len(P_est_raw) == 0:
                cov_reg = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # 如果是列表，使用 torch.stack 沿时间维度拼接成 [T, x_dim, x_dim]
                if isinstance(P_est_raw, list):
                    P_est = torch.stack(P_est_raw, dim=0).to(self.device)
                else:
                    P_est = P_est_raw.to(self.device)
                
                # 对齐时间步（非常安全，防止初始协方差导致的步数差 1 问题）
                min_len = min(P_est.shape[0], diff_cov.shape[0])
                cov_reg = torch.abs(P_est[-min_len:] - diff_cov[-min_len:]).mean()
            # ================================================

            mse_val = mse_loss.item()
            cov_val = cov_reg.item()
            
            # EMA 动量系数 (原代码中的 beta，改名防止与截断阈值 \beta 冲突)
            ema_decay = 0.8 
            eps = 1e-8

            # 1. 更新 EMA (指数移动平均)
            if self.mse_ema is None:
                self.mse_ema = mse_val
                self.cov_ema = cov_val
            else:
                self.mse_ema = ema_decay * self.mse_ema + (1 - ema_decay) * mse_val
                self.cov_ema = ema_decay * self.cov_ema + (1 - ema_decay) * cov_val

            # 加入 eps 防止分母为 0
            L_mse_ema = self.mse_ema + eps
            L_cov_ema = self.cov_ema + eps

            # 2. 论文中指定的超参数
            gamma = 0.01      # 上限及缩放因子 (upper scaling factor)
            beta_clip = 0.0001    # 下限截断阈值 (lower clipping threshold)

            # 3. 计算公式 (18) 中的动态权重
            # 注意：分子现在是 Cov，且外面乘了 gamma
            raw_lambda = gamma * (L_cov_ema / (L_mse_ema + L_cov_ema))

            # 4. bilateral clipping (双侧截断) 到 [0.0001, 0.1]
            lambda_i = max(beta_clip, min(gamma, raw_lambda))
            # print(f"Dynamic weight (lambda_i): {lambda_i:.4f}")
            # 5. 计算最终 Loss
            # (1 - lambda_i) 会在 [0.9, 0.9999] 之间，确保了 MSE 始终是 Primary objective
            loss = (1 - lambda_i) * mse_loss + lambda_i * cov_reg

            # loss = mse_loss 
            if (self.train_count + 1) % 25 == 0:
                self.dnn.update_R_adaptive()
        else:
            loss = self.loss_fn(batch_x[:, :, 1:], x_hat[:, :, 1:])



        self._sync_cuda()
        forward_ms = (time.perf_counter() - t_fwd0) * 1000.0

        if self.train_count % print_num == 1:
            print(
                f'[Model {self.save_path}] [Train {self.train_count}] '
                f'loss [dB] = {(10 * torch.log10(loss)).item():.4f}'
            )

        if not torch.isfinite(loss):
            # print(f"[Warning] Non-finite loss at train {self.train_count}: {loss}")
            self.optimizer.zero_grad(set_to_none=True)
            self.train_count += 1
            self.data_idx += self.batch_size
            return forward_ms

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dnn.kf_net.parameters(), 10)
        self.optimizer.step()

        if isinstance(self.dnn, Model_structured_filtering) \
           or isinstance(self.dnn, Unsupervised_DANSE):
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr > 5e-4:
                self.scheduler.step()
        self.train_count += 1
        self.data_idx += self.batch_size

        if self.train_count % save_num == 0:
            save_dict = {
                'kf_net': self.dnn.kf_net.state_dict(),
            }

            # 【新增逻辑】动态检查模型是否有 r_coef 属性
            if hasattr(self.dnn, 'r_coef'):
                save_dict['r_coef'] = self.dnn.r_coef
            torch.save(
                save_dict,
                './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt'
            )

        return forward_ms

    def train_batch_latent_encoder_warmup(self):
        """
        Latent-KalmanNet Stage 1:
        Only train encoder with encoder MSE.
        Do NOT run filtering and do NOT train KalmanNet.
        """
        for p in self.dnn.encoder.parameters():
            p.requires_grad = True
        for p in self.dnn.kf_net.parameters():
            p.requires_grad = False

        self.dnn.encoder.train()
        self.dnn.kf_net.eval()

        active_opt = self.optimizer_enc
        active_sch = self.scheduler_enc
        active_opt.zero_grad(set_to_none=True)

        batch_x, batch_y = self._get_batch()

        self._sync_cuda()
        t_fwd0 = time.perf_counter()

        z_pred_list = []
        z_true_list = []

        for i in range(self.batch_size):
            self.dnn.reset(clean_history=True)
            self._move_dnn_to_device()

            self.dnn.encoder.train()
            self.dnn.kf_net.eval()

            x_prev = batch_x[i, :, 0].reshape(-1, 1).to(self.device)

            for ii in range(1, self.seq_len):
                with torch.no_grad():
                    x_prior = self.dnn.GSSModel.f(x_prev).to(self.device)

                obs_in = batch_y[i, :, ii].reshape(-1, 1).view(1, 1, -1, 1).to(self.device)
                prior_in = x_prior.view(1, -1).to(self.device)

                z_pred = self.dnn.encoder(obs_in, prior_in).view(-1, 1)

                x_true = batch_x[i, :, ii].reshape(-1, 1).to(self.device)
                with torch.no_grad():
                    z_true = self.dnn.GSSModel.g(x_true).detach().to(self.device)

                z_pred_list.append(z_pred)
                z_true_list.append(z_true)

                # Teacher forcing for encoder warmup stability
                x_prev = x_true.detach()

        z_pred_all = torch.stack(z_pred_list, dim=0).to(self.device)
        z_true_all = torch.stack(z_true_list, dim=0).to(self.device)

        encoder_loss = self.loss_fn(z_pred_all, z_true_all)
        loss = encoder_loss

        self._sync_cuda()
        forward_ms = (time.perf_counter() - t_fwd0) * 1000.0

        if not torch.isfinite(loss):
            print(f"[Warning] Non-finite encoder warmup loss at train {self.train_count}: {loss}")
            active_opt.zero_grad(set_to_none=True)
            self.train_count += 1
            self.data_idx += self.batch_size
            return forward_ms

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dnn.encoder.parameters(), 10)
        active_opt.step()

        current_lr = active_opt.param_groups[0]['lr']
        if current_lr > 5e-4:
            active_sch.step()

        if self.train_count % print_num == 1:
            print(
                f'[Model {self.save_path}] [Warmup {self.train_count}] '
                f'[Encoder-Only] '
                f'encoder_loss = {encoder_loss.detach().item():.6e}, '
                f'encoder_loss[dB] = {(10 * torch.log10(encoder_loss.detach())).item():.4f}'
            )

        self.train_count += 1
        self.data_idx += self.batch_size

        if self.train_count % save_num == 0:
            save_dict = {
                'kf_net': self.dnn.kf_net.state_dict(),
                'encoder': self.dnn.encoder.state_dict()
            }
            torch.save(
                save_dict,
                './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt'
            )

        return forward_ms

    def train_batch_latent_alternating(self):
        """
        Latent-KalmanNet Stage 2:
        Encoder warmup 已经在 main.py 中完成。
        之后固定 encoder，只训练 KalmanNet，loss 使用 state MSE。
        """

        state_loss_val = None

        # Freeze encoder, train only KalmanNet
        for p in self.dnn.encoder.parameters():
            p.requires_grad = False
        for p in self.dnn.kf_net.parameters():
            p.requires_grad = True

        self.dnn.encoder.eval()
        self.dnn.kf_net.train()

        active_opt = self.optimizer_kf
        active_sch = self.scheduler_kf
        active_name = "KalmanNet-StateMSE"

        active_opt.zero_grad(set_to_none=True)

        batch_x, batch_y = self._get_batch()

        self._sync_cuda()
        t_fwd0 = time.perf_counter()

        x_hat_list = []

        for i in range(self.batch_size):
            self.dnn.reset(clean_history=True)
            self._move_dnn_to_device()

            self.dnn.encoder.eval()
            self.dnn.kf_net.train()

            self.dnn.state_post = batch_x[i, :, 0].reshape(-1, 1).to(self.device)
            self.dnn.x_post = self.dnn.state_post.clone().to(self.device)
            self.dnn.x_post_prev = self.dnn.state_post.clone().to(self.device)

            for ii in range(1, self.seq_len):
                obs_t = batch_y[i, :, ii].reshape(-1, 1).to(self.device)
                self.dnn.filtering(obs_t)

            x_hat_list.append(self.dnn.state_history[:, -self.seq_len:].to(self.device))

        x_hat = torch.stack(x_hat_list, dim=0).to(self.device)

        state_loss = self.loss_fn(batch_x[:, :, 1:], x_hat[:, :, 1:])
        loss = state_loss
        state_loss_val = state_loss.detach()

        self._sync_cuda()
        forward_ms = (time.perf_counter() - t_fwd0) * 1000.0

        if not torch.isfinite(loss):
            # print(f"[Warning] Non-finite latent state loss at train {self.train_count}: {loss}")
            active_opt.zero_grad(set_to_none=True)
            self.train_count += 1
            self.data_idx += self.batch_size
            return forward_ms

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dnn.kf_net.parameters(), 10)

        active_opt.step()

        current_lr = active_opt.param_groups[0]['lr']
        if current_lr > 5e-4:
            active_sch.step()

        if self.train_count % print_num == 1:
            print(
                f'[Model {self.save_path}] [Train {self.train_count}] '
                f'[{active_name}] '
                f'state_loss = {state_loss_val.item():.6e}, '
                f'state_loss[dB] = {(10 * torch.log10(state_loss_val)).item():.4f}'
            )

        self.train_count += 1
        self.data_idx += self.batch_size

        if self.train_count % save_num == 0:
            save_dict = {
                'kf_net': self.dnn.kf_net.state_dict(),
                'encoder': self.dnn.encoder.state_dict()
            }
            torch.save(
                save_dict,
                './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt'
            )

        return forward_ms
    def validate(self, tester):
        if tester.loss.item() < self.loss_best:
            if isinstance(tester.filter, Latent_KalmanNet_Filter):
                save_dict = {
                    'kf_net': tester.filter.kf_net.state_dict(),
                    'encoder': tester.filter.encoder.state_dict()
                }
            else:
                save_dict = {
                    'kf_net': tester.filter.kf_net.state_dict()
                }

            torch.save(
                save_dict,
                './.model_saved/' + self.save_path[:-3] + '_best.pt'
            )

            print(
                f'Save best model at {self.save_path} & train {self.train_count} '
                f'& loss [dB] = {tester.loss:.4f}'
            )

            self.loss_best = tester.loss.item()

        self.valid_loss = tester.loss.item()