#! /user/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gingkg
@contact: sby2015666@163.com
@software: PyCharm
@project: AlphaZero_Gomoku_PaddlePaddle
@file: policy_value_net_paddlepaddle.py
@date: 2021/3/4
@desc:
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # 公共网络层
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 行动策略网络层
        self.act_conv1 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.act_fc1 = nn.Linear(4*self.board_width*self.board_height, self.board_width*self.board_height)
        self.val_conv1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, padding=0)
        self.val_fc1 = nn.Linear(2*self.board_height*self.board_width, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, inputs):
        # 公共网络层
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行动策略层
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(x_act.shape[0], -1)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 状态价值网络层
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(x_val.shape[0], -1)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    """策略价值网络"""
    def __init__(self, board_width, board_height, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        if self.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-3

        self.policy_value_net = Net(self.board_width, self.board_height).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(),
                                          lr=0.02, weight_decay=self.l2_const)
        if model_file:
            net_params = self.policy_value_net.state_dict()
            pre_net_params = torch.load(model_file)
            # 将预加载模型里不属于新模型的参数剔除
            new_net_params = {k: v for k, v in pre_net_params.items() if k in net_params}
            self.policy_value_net.load_state_dict(new_net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        保证送进来的state_batch就是numpy变量
        """
        state_batch = torch.from_numpy(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = torch.exp(log_act_probs)
        return act_probs.detach().cpu().numpy(), value.detach().cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.available
        # print(board.current_state())
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height)).astype("float32")
        # current_state = torch.from_numpy(current_state).to(self.device)
        act_probs, value = self.policy_value(current_state)
        act_probs = act_probs.flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])

        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        """perform a training step"""
        state_batch = torch.from_numpy(state_batch).to(self.device)
        mcts_probs = torch.from_numpy(mcts_probs).to(self.device)
        winner_batch = torch.from_numpy(winner_batch).to(self.device)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.zero_grad()
        log_act_probs, value = self.policy_value_net(state_batch)
        value = value.reshape(-1,)
        value_loss_func = nn.MSELoss(reduction="mean")
        value_loss = value_loss_func(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, axis=1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs)*log_act_probs, axis=1))
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)



































