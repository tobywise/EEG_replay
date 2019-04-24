import networkx as nx
import numpy as np
from scipy.stats import norm

def add_features(X):
    return X.reshape(X.shape[0], -1)

def select_path(transition_matrix, outcome_state):

    G = nx.DiGraph(transition_matrix)

    path = nx.shortest_path(G, 0, outcome_state)

    m = transition_matrix.copy()

    for i in range(len(m)):
        if i not in path:
            m[i, :] = 0
            m[:, i] = 0

    return m

def generate_test_matrix(transition_matrix, outcome_states, nstates=7, lag=5, ntimepoints=1000, plot=False):

    G = nx.DiGraph(transition_matrix)

    rv = norm(loc=lag/2, scale=lag/4)

    X = np.zeros((ntimepoints, nstates))

    paths = []
    for o in outcome_states:
        paths.append(nx.shortest_path(G, 0, o))
    paths = np.array(paths)

    max_seq_length = np.max(paths.shape)
    n_replay_events = ntimepoints / (lag * max_seq_length + 3)

    for e in range(n_replay_events.astype(int)):
        for n, s in enumerate(paths[np.random.choice(range(len(paths)))]):
            X[(lag * max_seq_length + 3) * e + lag * n:(lag * max_seq_length + 3)
                                                       * e + lag * n + lag, s] = rv.pdf(np.arange(lag))

    if plot:
        sns.heatmap(X)

    return X


#
#
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sys
# sys.path.insert(0,'code')
# from sequenceness import sequenceness_regression
# # from utils import select_path
#
#
# transition_matrix = np.loadtxt(r'task/Task_information/transition_matrix.txt')
#
# matrices = [transition_matrix]
#
# for n, i in enumerate([5, 6]):
#     matrices.append(select_path(transition_matrix, i))
#
# X = generate_test_matrix(transition_matrix, [5], lag=5)
#
# seq_a = sequenceness_regression(X, matrices[1], max_lag=40, alpha=True)
# seq_b = sequenceness_regression(X, matrices[2], max_lag=40, alpha=True)
#
# # plt.plot(seq[0])
# # plt.plot(seq[1])
#
# plt.plot(seq_a[2])
# plt.plot(seq_b[2])