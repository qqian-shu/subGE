import numpy as np
import pandas as pd


class MDP(object):
    def __init__(self, value, subgraph_num, k):
        super(MDP, self).__init__()
        self.value = value
        self.action_num = 2
        self.n_features = subgraph_num
        self.k = k

    def reset(self):
        initial_observation = self.get_observation()
        initial_observation = np.array(initial_observation)
        return initial_observation

    def get_observation(self):
        select_num = int(self.k * self.n_features)
        observation = select_num
        return observation

    def step(self, action, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, last_acc):
        if action == 0:  # add
            if self.k <= (1 - self.value):
                self.k += self.value
        elif action == 1:  # minus
            if self.k >= self.value:
                self.k -= self.value

        eva_loss, eva_acc, _, rl_reward, _, _, _, _ = net.evaluate(test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, self.k)
        s_ = self.get_observation()
        # print(s_)
        s_ = np.array(s_)
        # reward function
        if rl_reward > last_acc:
            reward = 1
            done = True
        elif rl_reward - last_acc >= -0.1:
            reward = -1
            done = True
        elif rl_reward - last_acc < -0.1:
            reward = -2
            done = True
        else:
            reward = 0
            done = True
        return s_, reward, done, rl_reward

class QTable:
    def __init__(self, actions, learning_rate=0.85, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


def isTerminal(k_record, limited_epochs, delta_k, start_check_epochs=300)->bool:
    assert start_check_epochs > limited_epochs, 'the length of k_record is not long enough'
    if len(k_record) >= start_check_epochs:
        record_len = len(k_record)
        record_last_limited_epochs = np.array(k_record[record_len-limited_epochs:])
        range_ = np.max(record_last_limited_epochs) - np.min(record_last_limited_epochs)

        if range_ <= delta_k:
            return True
        else:
            return False

    else:
        return False


def run_QL(env, RL, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, initial_acc):
    observation = env.reset()
    while True:
        action = RL.choose_action(str(observation))
        observation_, reward, done, initial_acc = env.step(action, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, initial_acc)

        RL.learn(str(observation), action, reward, str(observation_), done)
        env.k = round(env.k, 4)
        if done:
            return env.k, reward


def generate_experience(env, RL, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, initial_acc):
    observation = env.reset()
    RL.check_state_exist(str(observation))
    initial_k = round(env.k, 4)
    k_list = np.linspace(0, 1, 21)
    k_list = k_list[1:]
    for k in k_list:
        env.k = round(k, 4)
        observation = env.reset()
        RL.check_state_exist(str(observation))
        action1 = RL.actions[0]
        observation_, reward, done = env.step(action1, net, test_x, test_dsi, test_sadj, test_t, test_t_mi,
                                              test_mask, initial_acc)
        RL.learn(str(observation), action1, reward, str(observation_), done)
        # print(k, 'action1', action1, observation, observation_, reward)
        RL.learn(str(observation), action1, reward, str(observation_), done)

        env.k = round(k, 4)
        observation = env.reset()
        RL.check_state_exist(str(observation))
        action2 = RL.actions[1]
        observation_, reward, done = env.step(action2, net, test_x, test_dsi, test_sadj, test_t, test_t_mi,
                                              test_mask, initial_acc)
        RL.learn(str(observation), action1, reward, str(observation_), done)
        # print(k, 'action2', action2, observation, observation_, reward)
    env.k = initial_k


if __name__ == "__main__":
    env = Maze()
    RL = QTable(actions=list(range(env.action_num)))

    env.after(100, update)
    env.mainloop()
