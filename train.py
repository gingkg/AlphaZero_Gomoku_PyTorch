#! /user/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gingkg
@contact: sby2015666@163.com
@software: PyCharm
@project: AlphaZero_Gomoku_PaddlePaddle
@file: train.py
@date: 2021/3/6
@desc: 对于五子棋的AlphaZero的训练的实现
"""

from __future__ import print_function
import random
import numpy as np
import os
from collections import defaultdict, deque
from game import Board, Game_UI
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaGoZero import MCTSPlayer
from policy_value_net_torch import PolicyValueNet
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sns.set()

data_train = {"batch": [], "value": [], "criterion": []}
data_eval = {"batch": [], "win_ratio": []}


class TrainPipeline:
    def __init__(self, init_model=None, is_shown=False):
        # 五子棋逻辑和棋盘UI的参数
        self.board_with = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_with,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.is_shown = is_shown
        self.game = Game_UI(self.board, is_shown)
        # 训练参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 基于KL自适应地调整学习率
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.save_freq = 50
        self.check_freq = 100
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # 用于纯粹的mcts的模拟数量，用作评估训练策略的对手
        self.pure_mcts_playout_num = 1000
        if init_model:
            # 从初始的策略价值网开始训练
            self.policy_value_net = PolicyValueNet(self.board_with,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # 从新的策略价值网络开始训练
            self.policy_value_net = PolicyValueNet(self.board_with,
                                                   self.board_height)

        # 定义训练机器人
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_date(self, play_data):
        """通过旋转和翻转来增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1,2,3,4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_with)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我博弈数据进行训练"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            # print("====================================")
            # print(play_data)
            # play_data = list(play_data)
            # print(play_data)
            # print("====================================")
            play_data = list(play_data)[:]  # 这里可能有问题
            self.episode_len = len(play_data)
            # print("hfuiehriughier:", str(self.episode_len))
            # 增加数据
            play_data = self.get_equi_date(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]

        state_batch = np.array(state_batch).astype("float32")

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype("float32")

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype("float32")

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate*self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs*(np.log(old_probs+1e-10)-np.log(new_probs+1e-10)), axis=1))
            if kl > self.kl_targ*4:  # early stopping if D_KL diverges badly
                break

        # 自适应调节学习率
        if kl > self.kl_targ*2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ/2 and self.lr_multiplier < 10:
            self.lr_multiplier += 1.5

        # ???
        explained_var_old = (1 -
                             np.var(np.array(winner_batch)-old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evalute(self, n_games=10):
        """
       通过与纯的MCTS算法对抗来评估训练的策略
       注意：这仅用于监控训练进度
       """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            print("\revaluation process: {}/{}".format(i+1, n_games), end="")
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i%2)
            win_cnt[winner] += 1
        print("")
        win_ratio = 1.0 * (win_cnt[1] + 0.5*win_cnt[-1]) / n_games  # ???
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """开始训练"""
        root = os.getcwd()
        dst_path = os.path.join(root, 'dist')
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    data_train["batch"] += ([i+1, i+1])
                    data_train["value"] += [float(loss), float(entropy)]
                    data_train["criterion"] += ["loss", "entropy"]
                    print("loss :{}, entropy:{}".format(loss, entropy))

                if (i+1) % self.save_freq == 0:
                    self.policy_value_net.save_model(os.path.join(dst_path, 'current_policy_step.model'))
                # 检查当前模型的性能，保存模型的参数
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evalute(2)
                    data_eval["batch"].append(i+1)
                    data_eval["win_ratio"].append(win_ratio)
                    self.policy_value_net.save_model(os.path.join(dst_path, 'current_policy.model'))
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # 更新最好的策略
                        self.policy_value_net.save_model(os.path.join(dst_path, 'best_policy.model'))
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 8000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

        except KeyboardInterrupt:
            print('\n\rquit')


def draw(data_train, data_eval):
    data_train = DataFrame(data_train)
    data_eval = DataFrame(data_eval)

    plt.figure()
    ax = sns.lineplot(x="batch", y="value", hue="criterion", style="criterion", data=data_train)
    plt.xlabel("batch", fontsize=12)
    plt.ylabel("value", fontsize=12)
    plt.title("AlphaZero Gomoku Performance(train)", fontsize=14)
    plt.show()
    fig = ax.get_figure()
    fig.savefig("images/AlphaZero_Gomoku_Performance(train).png")

    plt.figure()
    ax = sns.lineplot(x="batch", y="win_ratio", data=data_eval)
    plt.xlabel("batch", fontsize=12)
    plt.ylabel("win ratio", fontsize=12)
    plt.title("AlphaZero Gomoku Performance(eval)", fontsize=14)
    plt.show()
    fig = ax.get_figure()
    fig.savefig("images/AlphaZero_Gomoku_Performance(eval).png")

    with pd.ExcelWriter(r'data/data.xlsx') as writer:
        data_train.to_excel(writer, sheet_name='train')
        data_eval.to_excel(writer, sheet_name='eval')


if __name__ == "__main__":
    is_shown = False
    # model_path = 'dist/best_policy.model'
    model_path = 'dist/current_policy.model'

    # training_pipeline = TrainPipeline(model_path, is_shown)
    training_pipeline = TrainPipeline(None, is_shown)
    training_pipeline.run()

    draw(data_train, data_eval)










