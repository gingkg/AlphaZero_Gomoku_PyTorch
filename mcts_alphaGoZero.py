#! /user/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gingkg
@contact: sby2015666@163.com
@software: PyCharm
@project: AlphaZero_Gomoku_PaddlePaddle
@file: mcts_alphaGoZero.py
@date: 2021/3/5
@desc: 蒙特卡罗树搜索AlphaGo Zero形式，使用策略值网络引导树搜索和评估叶节点
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x-np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """MCTS树中的节点。
    每个节点跟踪其自身的值Q，先验概率P及其访问次数调整的先前得分u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """通过创建新子项来展开树。
        action_priors：一系列动作元组及其先验概率根据策略函数.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择能够提供最大行动价值Q的行动加上奖金u（P）。
        return：（action，next_node）的元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """从叶节点评估中更新节点值
        leaf_value: 这个子树的评估值来自从当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值， 取对于所有访问次数的平均数
        self._Q += 1.0*(leaf_value-self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """就像调用update（）一样，但是对所有祖先进行递归应用。
        """
        # 如果它不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
        调整了访问次数，u。
        c_puct：控制相对影响的（0，inf）中的数字，该节点得分的值Q和先验概率P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查叶节点（即没有扩展的节点）。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    """对蒙特卡罗树搜索的一个简单实现"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn：一个接收板状态和输出的函数（动作，概率）元组列表以及[-1,1]中的分数
        （即来自当前的最终比赛得分的预期值玩家的观点）对于当前的玩家。
        c_puct：（0，inf）中的数字，用于控制探索的速度收敛于最大值政策。 更高的价值意味着
        依靠先前的更多。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """从根到叶子运行单个播出，获取值
         叶子并通过它的父母传播回来。
         State已就地修改，因此必须提供副本。
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 使用网络评估叶子，该网络输出（动作，概率）元组p的列表以及当前玩家的[-1,1]中的分数v。
        action_probs, leaf_value = self._policy(state)
        # 查看游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 对于结束状态,将叶子节点的值换成"true"
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 在本次遍历中更新节点的值和访问次数
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """按顺序运行所有播出并返回可用的操作及其相应的概率。
        state: 当前游戏的状态
        temp: 介于(0,1]之间的临时参数控制探索的概率
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 根据根节点处的访问计数来计算移动概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """在当前的树上向前一步，保持我们已经知道的关于子树的一切.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer:
    """基于MCTS的AI玩家"""
    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.available
        # 像alphaGo Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 添加Dirichlet Noise进行探索（自我训练所需）
                move = np.random.choice(
                    acts,
                    p=0.75*probs+0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点并重用搜索树
                self.mcts.update_with_move(move)
            else:
                # 使用默认的temp = 1e-3，它几乎相当于选择具有最高概率的移动
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)














