import math
import numpy as np
import torch
import torch.nn as nn
from GSSFiltering.dnn import (
    DNN_KalmanNet_GSS,
    DNN_SKalmanNet_GSS,
    KNet_architecture_v2,
    Danse_net,

    Latent_KalmanNet,
    MFiltering_Net,
)
from GSSFiltering.model import GSSModel
import torch.nn.init as init


def get_default_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def init_hidden_safe(net, device):
    if hasattr(net, "initialize_hidden"):
        try:
            net.initialize_hidden(device=device)
        except TypeError:
            net.initialize_hidden()


def to_device_tensor(x, device, dtype=None):
    if torch.is_tensor(x):
        if dtype is None:
            return x.to(device)
        return x.to(device=device, dtype=dtype)
    return x

class Unscented_Kalman_Filter():
    def __init__(self, GSSModel, alpha=1e-3, beta=2.0, kappa=0.0, device=None):
        self.device = get_default_device(device)
        self.dtype = torch.float64

        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.init_state = GSSModel.init_state.to(device=self.device, dtype=self.dtype)
        self.Q = GSSModel.cov_q.to(device=self.device, dtype=self.dtype)
        self.R = GSSModel.cov_r.to(device=self.device, dtype=self.dtype)

        self.init_cov = torch.eye(self.x_dim, device=self.device, dtype=self.dtype) * 0.1

        self.state_post = self.init_state.detach().clone()
        self.cov_post = self.init_cov.detach().clone()

        self.L = self.x_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = self.alpha ** 2 * (self.L + self.kappa) - self.L

        self.Wm = torch.zeros(2 * self.L + 1, device=self.device, dtype=self.dtype)
        self.Wc = torch.zeros(2 * self.L + 1, device=self.device, dtype=self.dtype)
        self.Wm[0] = self.lam / (self.L + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, 2 * self.L + 1):
            self.Wm[i] = 1.0 / (2 * (self.L + self.lam))
            self.Wc[i] = self.Wm[i]

        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        self.state_post = self.init_state.detach().clone()
        self.cov_post = self.init_cov.detach().clone()

        if clean_history:
            self.state_history = self.state_post.clone()
            self.cov_trace_history = torch.tensor([torch.trace(self.cov_post)], device=self.device, dtype=self.dtype)
            self.P_history = [self.cov_post.detach().clone()]
        else:
            self.state_history = torch.cat((self.state_history, self.state_post), dim=1)
            self.P_history.append(self.cov_post.detach().clone())

    def _get_sigma_points(self, mean, cov):
        mean = mean.to(device=self.device, dtype=self.dtype)
        cov = cov.to(device=self.device, dtype=self.dtype)

        cov = (cov + cov.T) / 2.0
        jitter = torch.eye(self.L, device=self.device, dtype=self.dtype) * 1e-8

        try:
            L_mat = torch.linalg.cholesky(cov + jitter)
        except torch._C._LinAlgError:
            vals, vecs = torch.linalg.eigh(cov)
            vals = torch.clamp(vals, min=1e-8)
            cov_reconstructed = vecs @ torch.diag(vals) @ vecs.T
            L_mat = torch.linalg.cholesky(cov_reconstructed + jitter)

        sigma_points = torch.zeros((self.L, 2 * self.L + 1), device=self.device, dtype=self.dtype)
        sigma_points[:, 0] = mean.view(-1)
        factor = math.sqrt(self.L + self.lam)

        for i in range(self.L):
            sigma_points[:, i + 1] = mean.view(-1) + factor * L_mat[:, i]
            sigma_points[:, i + 1 + self.L] = mean.view(-1) - factor * L_mat[:, i]
        return sigma_points

    def filtering(self, observation):
        observation = observation.to(device=self.device, dtype=self.dtype)
        self.Wm = self.Wm.to(self.device)
        self.Wc = self.Wc.to(self.device)

        sigma_points = self._get_sigma_points(self.state_post, self.cov_post)
        sigma_points_pred = torch.zeros_like(sigma_points)
        for i in range(2 * self.L + 1):
            sigma_points_pred[:, i:i + 1] = self.GSSModel.f(sigma_points[:, i].view(-1, 1)).to(device=self.device, dtype=self.dtype)

        x_predict = torch.sum(self.Wm.view(1, -1) * sigma_points_pred, dim=1, keepdim=True)
        cov_pred = torch.zeros((self.L, self.L), device=self.device, dtype=self.dtype)

        for i in range(2 * self.L + 1):
            diff = sigma_points_pred[:, i:i + 1] - x_predict
            cov_pred += self.Wc[i] * (diff @ diff.T)

        cov_pred = (cov_pred + cov_pred.T) / 2.0 + self.Q
        sigma_points_upd = self._get_sigma_points(x_predict, cov_pred)
        sigma_points_obs = torch.zeros((self.y_dim, 2 * self.L + 1), device=self.device, dtype=self.dtype)

        for i in range(2 * self.L + 1):
            sigma_points_obs[:, i:i + 1] = self.GSSModel.g(sigma_points_upd[:, i].view(-1, 1)).to(device=self.device, dtype=self.dtype)

        y_predict = torch.sum(self.Wm.view(1, -1) * sigma_points_obs, dim=1, keepdim=True)

        P_yy = torch.zeros((self.y_dim, self.y_dim), device=self.device, dtype=self.dtype)
        P_xy = torch.zeros((self.L, self.y_dim), device=self.device, dtype=self.dtype)

        for i in range(2 * self.L + 1):
            y_diff = sigma_points_obs[:, i:i + 1] - y_predict
            x_diff = sigma_points_upd[:, i:i + 1] - x_predict
            P_yy += self.Wc[i] * (y_diff @ y_diff.T)
            P_xy += self.Wc[i] * (x_diff @ y_diff.T)

        P_yy = (P_yy + P_yy.T) / 2.0 + self.R
        K_gain = torch.linalg.solve(P_yy, P_xy.T).T
        residual = observation - y_predict
        
        self.state_post = x_predict + K_gain @ residual
        cov_post = cov_pred - K_gain @ P_yy @ K_gain.T
        self.cov_post = (cov_post + cov_post.T) / 2.0

        self.state_history = torch.cat((self.state_history, self.state_post), dim=1)
        self.cov_trace_history = torch.cat((self.cov_trace_history, torch.trace(self.cov_post).reshape(-1)))
        
        # 记录协方差以计算 NEES
        self.P_history.append(self.cov_post.detach().clone())

        return self.state_post

class Extended_Kalman_Filter():
    def __init__(self, GSSModel: GSSModel, device=None):
        self.device = get_default_device(device)

        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.init_state = GSSModel.init_state.to(self.device)
        self.Q = GSSModel.cov_q.to(self.device)
        self.R = GSSModel.cov_r.to(self.device)

        self.init_cov = torch.zeros((self.x_dim, self.x_dim), device=self.device)

        self.state_history = self.init_state.detach().clone()
        self.state_post = self.init_state.detach().clone()
        self.cov_post = self.init_cov.detach().clone()

        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        self.state_post = self.state_post.to(self.device)

        if clean_history:
            self.state_history = self.init_state.detach().clone()
            self.cov_trace_history = torch.zeros((1,), device=self.device)
            self.P_history = []
        else:
            self.state_history = torch.cat((self.state_history, self.state_post), dim=1)

    def filtering(self, observation):
        with torch.no_grad():
            observation = observation.to(self.device)

            x_last = self.state_post
            x_predict = self.GSSModel.f(x_last).to(self.device)

            y_predict = self.GSSModel.g(x_predict).to(self.device)
            residual = observation - y_predict

            F_jacob = self.GSSModel.Jacobian_f(x_last).to(self.device)
            H_jacob = self.GSSModel.Jacobian_g(x_predict).to(self.device)

            cov_pred = F_jacob @ self.cov_post @ F_jacob.T + self.Q

            S = H_jacob @ cov_pred @ H_jacob.T + self.R
            S = S + torch.eye(S.size(0), device=self.device, dtype=S.dtype) * 1e-6

            K_gain = cov_pred @ H_jacob.T @ torch.linalg.inv(S)
            x_post = x_predict + K_gain @ residual

            I = torch.eye(self.x_dim, device=self.device, dtype=x_post.dtype)
            cov_post = (I - K_gain @ H_jacob) @ cov_pred
            cov_trace = torch.trace(cov_post)

            self.state_post = x_post.detach().clone()
            self.cov_post = cov_post.detach().clone()
            self.state_history = torch.cat((self.state_history, x_post.clone()), dim=1)
            self.cov_trace_history = torch.cat(
                (self.cov_trace_history, cov_trace.reshape(-1).clone())
            )
            
            if not hasattr(self, 'P_history'):
                self.P_history = []
            self.P_history.append(self.cov_post.clone())

            self.pk = cov_pred

class KalmanNet_Filter():
    def __init__(self, GSSModel: GSSModel, device=None):
        self.device = get_default_device(device)

        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_KalmanNet_GSS(self.x_dim, self.y_dim).to(self.device)
        self.init_state = GSSModel.init_state.to(self.device)
        
        # 为 Uncertainty analysis 获取 R 矩阵
        if hasattr(self.GSSModel, 'Construct_R'):
            self.R = self.GSSModel.Construct_R().to(self.device)
        else:
            self.R = self.GSSModel.cov_r.to(self.device)
        self.R = torch.diag(torch.diag(self.R))

        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        self.dnn_first = True
        init_hidden_safe(self.kf_net, self.device)
        self.state_post = self.init_state.detach().clone().to(self.device)

        if hasattr(self.GSSModel, 'init_cov'):
            self.P_post = self.GSSModel.init_cov.detach().clone().to(self.device)
        else:
            self.P_post = torch.eye(self.x_dim, device=self.device) * 0.1

        if clean_history:
            self.state_history = self.init_state.detach().clone().to(self.device)
            self.cov_trace_history = torch.zeros((1,), device=self.device)
            self.P_history = [self.P_post.detach().clone()]
        else:
            self.state_history = torch.cat((self.state_history, self.state_post), dim=1)
            self.P_history.append(self.P_post.detach().clone())

    def filtering(self, observation):
        observation = observation.to(self.device)

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last).to(self.device)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict).to(self.device)
        residual = observation - y_predict

        state_inno = self.state_post_past - self.state_pred_past
        diff_state = self.state_post - self.state_post_past
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(state_inno, residual, diff_state, diff_obs)
        x_post = x_predict + K_gain @ residual

        # ====================== Uncertainty Analysis (NEES Tracking) ======================
        try:
            H = self.GSSModel.Jacobian_g(x_predict).to(self.device)
            K = K_gain
            I = torch.eye(self.x_dim, device=self.device, dtype=x_post.dtype)
            Ht = H.T
            
            try:
                Htilde = torch.linalg.inv(Ht @ H)
            except Exception:
                Htilde = torch.linalg.pinv(Ht @ H)
                
            try:
                invKH_I = torch.linalg.inv(K @ H - I)
            except Exception:
                invKH_I = torch.linalg.pinv(K @ H - I)

            cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde
            cov_post_byK_optB = (I - K @ H) @ cov_pred_byK_opt2 @ (I - K @ H).T + K @ self.R @ K.T
            
            self.P_post = (cov_post_byK_optB + cov_post_byK_optB.T) / 2.0
        except Exception as e:
            self.P_post = torch.eye(self.x_dim, device=self.device) * 0.1
            
        self.P_history.append(self.P_post.detach().clone())
        # ===================================================================================

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), dim=1)
class KalmanNet_Filter_v2():
    def __init__(self, GSSModel: GSSModel, device=None):
        self.device = get_default_device(device)

        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = KNet_architecture_v2(self.x_dim, self.y_dim).to(self.device)
        self.init_state = GSSModel.init_state.to(self.device)

        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        self.dnn_first = True
        init_hidden_safe(self.kf_net, self.device)

        self.state_post = self.init_state.detach().clone().to(self.device)

        if clean_history:
            self.state_history = self.init_state.detach().clone().to(self.device)
            self.cov_trace_history = torch.zeros((1,), device=self.device)

        self.state_history = torch.cat((self.state_history, self.state_post), dim=1)

    def filtering(self, observation):
        observation = observation.to(self.device)

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last).to(self.device)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict).to(self.device)
        residual = observation - y_predict

        state_inno = self.state_post_past - self.state_pred_past
        diff_state = self.state_post - self.state_post_past
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(diff_obs, residual, diff_state, state_inno)
        x_post = x_predict + K_gain @ residual

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), dim=1)


class Split_KalmanNet_Filter():
    def __init__(self, GSSModel: GSSModel, device=None):
        self.device = get_default_device(device)

        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_SKalmanNet_GSS(self.x_dim, self.y_dim).to(self.device)
        self.init_state = GSSModel.init_state.to(self.device)

        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        self.dnn_first = True
        init_hidden_safe(self.kf_net, self.device)

        self.state_post = self.init_state.detach().clone().to(self.device)

        if clean_history:
            self.state_history = self.init_state.detach().clone().to(self.device)

        self.state_history = torch.cat((self.state_history, self.state_post), dim=1)

    def filtering(self, observation):
        observation = observation.to(self.device)

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last).to(self.device)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict).to(self.device)
        residual = observation - y_predict

        state_inno = self.state_post_past - self.state_pred_past
        diff_state = self.state_post - self.state_post_past
        diff_obs = observation - self.obs_past

        H_jacob = self.GSSModel.Jacobian_g(x_predict).to(self.device)
        linearization_error = y_predict - H_jacob @ x_predict
        H_jacob_in = H_jacob.reshape((-1, 1))

        Pk, Sk = self.kf_net(
            state_inno,
            residual,
            diff_state,
            diff_obs,
            linearization_error,
            H_jacob_in
        )

        K_gain = Pk @ H_jacob.T @ Sk
        x_post = x_predict + K_gain @ residual

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), dim=1)


class Unsupervised_DANSE():
    def __init__(self, GSSModel, device=None):
        self.device = get_default_device(device)

        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.R = GSSModel.cov_r.to(self.device)
        self.kf_net = Danse_net(self.x_dim, self.y_dim).to(self.device)
        self.init_state = GSSModel.init_state.to(self.device)

        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        self.dnn_first = True
        init_hidden_safe(self.kf_net, self.device)

        self.state_post = self.init_state.detach().clone().to(self.device)
        self.P_post = torch.eye(self.x_dim, device=self.device) * 0.1

        if clean_history:
            self.state_history = self.init_state.detach().clone().to(self.device)
            self.cov_trace_history = torch.zeros((1,), device=self.device)
            self.P_history = []

    def filtering(self, Yi_batch):
        Yi_batch = Yi_batch.to(self.device)

        mu_xt_yt_prev, L_xt_yt_prev_diag = self.kf_net(Yi_batch)
        L_xt_yt_prev = torch.diag_embed(L_xt_yt_prev_diag)

        H = self.GSSModel.Jacobian_g(mu_xt_yt_prev).to(self.device)

        mu_yt_current = torch.einsum('ij,btj->bti', H, mu_xt_yt_prev)
        L_yt_current = H @ L_xt_yt_prev @ H.T + self.R

        try:
            Re_t_inv = torch.linalg.inv(L_yt_current)
        except Exception:
            Re_t_inv = torch.linalg.pinv(L_yt_current)

        K_t = L_xt_yt_prev @ H.T @ Re_t_inv

        innovation = Yi_batch - mu_yt_current
        mu_xt_yt_current = mu_xt_yt_prev + torch.einsum('btij,btj->bti', K_t, innovation)
        L_xt_yt_current = L_xt_yt_prev - K_t @ H @ L_xt_yt_prev

        P_post = L_xt_yt_current[-1, -1, :, :]
        self.P_post = P_post.detach().clone()
        
        if not hasattr(self, 'P_history'):
            self.P_history = []
        self.P_history.append(self.P_post.clone())

        return mu_xt_yt_prev, L_xt_yt_prev, mu_xt_yt_current, L_xt_yt_current

class Model_structured_filtering(nn.Module):
    def __init__(self, GSSModel, device=None):
        super(Model_structured_filtering, self).__init__() # 必须添加这一行
        self.device = get_default_device(device)
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.init_state = GSSModel.init_state.to(self.device)
        self.kf_net = MFiltering_Net(x_dim=self.x_dim, y_dim=self.y_dim, device=self.device).to(self.device)

        # ================= 动态 R 参数 =================                       
        self.register_buffer('r_coef', torch.tensor(0.1, device=self.device)) 
        self.R = torch.eye(self.y_dim, device=self.device) * self.r_coef
        self.nees_buffer = []                       # 用于存放 25 轮内的所有 NEES 值
        # ==============================================
        
        self.reset(clean_history=True)

    def reset(self, clean_history=False):
        init_hidden_safe(self.kf_net, self.device)
        self.dnn_first = True
        self.state_post = self.init_state.detach().clone().to(self.device)
        self.P_post = torch.eye(self.x_dim, device=self.device) * 0.1

        if clean_history:
            self.state_history = self.state_post.detach().clone().to(self.device)
            self.P_history = [self.P_post.detach().clone()]
        else:
            self.P_history.append(self.P_post)

    def filtering(self, observation, true_x=None):
        observation = observation.to(self.device)
        x_prev = self.state_post.to(self.device)
        P_prev = self.P_post.to(self.device)

        x_pred, y_pred, K_hat, H = self.kf_net(x_prev, observation, P_prev)
        
        H, K_hat = H.to(self.device), K_hat.to(self.device)
        x_pred, y_pred = x_pred.to(self.device), y_pred.to(self.device)

        residual = observation - y_pred
        I = torch.eye(self.x_dim, device=self.device, dtype=x_pred.dtype)

        x_post = x_pred + K_hat @ residual
        P_post = (I - K_hat @ H) @ P_prev @ (I - K_hat @ H).T + K_hat @ self.R @ K_hat.T

        # ================== 仅计算并缓存 NEES，不在这里更新 r_coef ==================
        if true_x is not None:
            with torch.no_grad():
                t_x = true_x.reshape(self.x_dim, 1)
                x_p = x_post.reshape(self.x_dim, 1)
                P_p = P_post.reshape(self.x_dim, self.x_dim)
                err = t_x - x_p 
                try:
                    P_inv = torch.linalg.inv(P_p + 1e-6 * torch.eye(self.x_dim, device=self.device))
                except:
                    P_inv = torch.linalg.pinv(P_p)
                
                step_nees = (err.T @ P_inv @ err).item()
                self.nees_buffer.append(step_nees) # 放入缓存
        # =========================================================================

        self.state_post = x_post.detach().clone()
        self.P_post = P_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post), dim=1)
        self.dnn_first = False
        self.P_history.append(self.P_post)

        return x_post, y_pred, residual

    # ================= 新增：每 25 轮统一调用的更新函数 =================
    def update_R_adaptive(self):
        if not self.nees_buffer:
            return
        
        with torch.no_grad():
            # 1. 计算这段时间的平均 NEES
            avg_nees = sum(self.nees_buffer) / len(self.nees_buffer)
            
            # 2. 反馈调节比例
            ratio = avg_nees / float(self.x_dim)
            ratio = max(0.5, min(2.0, ratio)) 
            
            # 3. 更新系数
            old_coef = self.r_coef
            self.r_coef = self.r_coef * ratio 

            
            # 4. 更新 R 矩阵
            self.R = torch.eye(self.y_dim, device=self.device) * self.r_coef
            
            print(f" >>> [R-Update] Avg NEES: {avg_nees:.4f}, Coef: {old_coef:.6e} -> {self.r_coef:.6e}")
            
            # 5. 清空缓存，等待下 25 轮
            self.nees_buffer = []

class Latent_KalmanNet_Filter:
    def __init__(self, GSSModel, device=None):
        self.device = get_default_device(device)
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        # 【核心修改点 3】：传入 x_dim=self.x_dim
        self.encoder = Encoder_conv_with_prior(
            encoded_space_dim=self.y_dim, 
            x_dim=self.x_dim
        ).to(self.device)

        self.kf_net = Latent_KalmanNet(x_dim=self.x_dim, y_dim=self.y_dim).to(self.device)
        self.init_state = GSSModel.init_state.to(self.device)
        
        # 为 Uncertainty analysis 获取 R 矩阵
        if hasattr(self.GSSModel, 'Construct_R'):
            self.R = self.GSSModel.Construct_R().to(self.device)
        else:
            self.R = self.GSSModel.cov_r.to(self.device)
        self.R = torch.diag(torch.diag(self.R))

        self.reset(clean_history=True)
        
    def reset(self, clean_history=False):
        self.dnn_first = True
        init_hidden_safe(self.kf_net, self.device)
        init_hidden_safe(self.encoder, self.device)

        self.state_post = self.init_state.detach().clone().to(self.device)

        if hasattr(self.GSSModel, 'init_cov'):
            self.P_post = self.GSSModel.init_cov.detach().clone().to(self.device)
        else:
            self.P_post = torch.eye(self.x_dim, device=self.device) * 0.1

        if clean_history:
            self.state_history = self.init_state.detach().clone().to(self.device)
            self.cov_trace_history = torch.zeros((1,), device=self.device)
            self.P_history = [self.P_post.detach().clone()]
        else:
            self.state_history = torch.cat((self.state_history, self.state_post), dim=1)    
            self.P_history.append(self.P_post.detach().clone())

    def filtering(self, observation):
        observation = observation.to(self.device)

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last).to(self.device)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()
            
        observation_lift = self.encoder(
            observation.reshape(1, 1, -1, 1), 
            x_predict.reshape(1, -1)
        ).reshape(-1, 1)
        H_mat = self.GSSModel.Jacobian_g(x_predict).to(self.device)
        y_predict = torch.matmul(H_mat, x_predict.reshape(-1, 1))
        residual = observation - y_predict

        state_inno = self.state_post_past - self.state_pred_past
        diff_state = self.state_post - self.state_post_past
        diff_obs = observation - observation_lift

        K_gain = self.kf_net(diff_obs, residual, diff_state, state_inno)
        x_post = x_predict + K_gain @ residual

        # ====================== Uncertainty Analysis (NEES Tracking) ======================
        try:
            H = H_mat
            K = K_gain
            I = torch.eye(self.x_dim, device=self.device, dtype=x_post.dtype)
            Ht = H.T
            
            try:
                Htilde = torch.linalg.inv(Ht @ H)
            except Exception:
                Htilde = torch.linalg.pinv(Ht @ H)
                
            try:
                invKH_I = torch.linalg.inv(K @ H - I)
            except Exception:
                invKH_I = torch.linalg.pinv(K @ H - I)

            cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde
            cov_post_byK_optB = (I - K @ H) @ cov_pred_byK_opt2 @ (I - K @ H).T + K @ self.R @ K.T
            
            self.P_post = (cov_post_byK_optB + cov_post_byK_optB.T) / 2.0
        except Exception as e:
            self.P_post = torch.eye(self.x_dim, device=self.device) * 0.1
            
        self.P_history.append(self.P_post.detach().clone())
        # ===================================================================================

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), dim=1)
class Encoder_conv_with_prior(nn.Module):
    # 【修改点 1】: 增加 x_dim 参数
    def __init__(self, encoded_space_dim, x_dim):
        super(Encoder_conv_with_prior, self).__init__()
        self.encoded_space_dim = encoded_space_dim
        self.x_dim = x_dim  

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(True),
            nn.Dropout(0.05),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(True),
            nn.Dropout(0.05),

            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, 128),
            nn.ReLU(True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 1))
        self.flatten = nn.Flatten(start_dim=1)

        # 【修改点 2】: prior_fc 的输入维度改为真实的 x_dim
        self.prior_fc = nn.Sequential(
            nn.Linear(self.x_dim, 128),  
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
        )

        with torch.no_grad():
            dummy_in = torch.zeros(1, 1, encoded_space_dim, 1)
            feat = self.encoder_conv(dummy_in)
            feat = self.adaptive_pool(feat)
            flat_dim = feat.numel()

        self.encoder_lin = nn.Sequential(
            nn.Linear(flat_dim + 128, 512),
            nn.ReLU(True),
            nn.Dropout(0.05),

            nn.Linear(512, 256),
            nn.ReLU(True),

            nn.Linear(256, 128),
            nn.ReLU(True),

            nn.Linear(128, 64),
            nn.ReLU(True),

            nn.Linear(64, encoded_space_dim)
        )

    def forward(self, x, prior):
        device = next(self.parameters()).device
        x = x.to(device)
        prior = prior.to(device)

        x = self.encoder_conv(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)

        prior = self.prior_fc(prior)

        comb = torch.cat((prior, x), dim=1)
        out = self.encoder_lin(comb)

        return out