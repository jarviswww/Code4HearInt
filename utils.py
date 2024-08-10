import torch
import numpy as np
from torch.utils.data import Dataset

def item_to_centroids(item_to_cat):
    sorted_keys = sorted(item_to_cat.keys())
    unique_keys_count = len(sorted_keys)
    matrix = np.zeros((unique_keys_count))
    for i, key in enumerate(sorted_keys):
        matrix[i] = item_to_cat[key]
    return torch.tensor(matrix).long()


def renumber_cat(original_dict):
    sorted_values = np.unique(sorted(original_dict.values(), reverse=True))
    value_indices = {value: index for index, value in enumerate(sorted_values)}

    new_dict = {key: value_indices[value] for key, value in original_dict.items()}
    return new_dict


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num])
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
            adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
            num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

        else:
            # sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
            adj_entity[entity] = np.concatenate([neighbor, [0] * (sample_num - len(neighbor))])
            num_entity[entity] = np.concatenate([neighbor_weight + [0] * (sample_num - len(neighbor))])
        # adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        # num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    for id, i in enumerate(num_entity):
        sum_ = np.sum(i)
        if sum_ != 0:
            for j in range(len(i)):
                num_entity[id][j] = num_entity[id][j] / sum_

    return adj_entity, num_entity


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [[0] * (max_len - le) + list(upois) if le < max_len else list(upois[-max_len:])
               for upois, le in zip(inputData, len_data)]
    us_msks = [[0] * (max_len - le) + [1] * le if le < max_len else [1] * max_len
               for le in len_data]

    # get position
    position = list()
    for items in inputData:
        pos = list()
        for id_ in range(len(items)):
            pos.append(id_ + 1)
        position.append(pos)

    # pad position
    _poses = np.zeros((len(inputData), max_len))
    for i in range(len(inputData)):
        seq = inputData[i]
        pos = position[i]
        length = len(seq)
        _poses[i][-length:] = pos

    item_all = inputData
    for i in range(len(item_all)):
        node = np.unique(item_all[i])
        item_all[i] = node.tolist() + (max_len - len(node)) * [0]

    return us_pois, us_msks, max_len, _poses, item_all


class Data(Dataset):
    def __init__(self, data, adj_global, train_len=None):
        self.data = data
        inputs, mask, max_len, pos, item_all = handle_data(data[0], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.global_adj = np.asarray(adj_global)
        self.length = len(data[0])
        self.max_len = max_len
        self.item_all = item_all
        self.pos = pos

    def __getitem__(self, index):
        u_input, mask, target, pos, items = self.inputs[index], self.mask[index], self.targets[index], self.pos[index], \
                                            self.item_all[index]
        max_n_node = self.max_len
        node = np.unique(u_input)
        session_items = node.tolist() + (max_n_node - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        adj = np.zeros((max_n_node, max_n_node))
        adj_nd = np.zeros((max_n_node, max_n_node))

        for i in sorted(np.arange(1, max_n_node), reverse=True):  1
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            adj_nd[u][u] = 4
            if u_input[i - 1] == 0:
                break
            v = np.where(node == u_input[i - 1])[0][0]
            if u != v:
                adj_nd[u][v] = self.global_adj[u_input[i]][u_input[i - 1]]
                adj_nd[v][u] = self.global_adj[u_input[i - 1]][u_input[i]]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        u_input_index = []
        for i in u_input:
            u_input_index.append(node.tolist().index(i))
        return [torch.tensor(target), torch.tensor(u_input), torch.tensor(adj),
                torch.tensor(adj_nd),
                torch.tensor(session_items), torch.tensor(u_input_index)]

    def __len__(self):
        return self.length

    def get_max_len(self):
        return self.max_len


def renumberItems(train_x, train_y, test_x, test_y):
    # get item set
    item_set = set()
    for items in train_x:
        for id_ in range(len(items)):
            item_set.add(items[id_])

    for item in train_y:
        item_set.add(item)

    for items in test_x:
        for id_ in range(len(items)):
            item_set.add(items[id_])

    for item in test_y:
        item_set.add(item)

    # renumbered
    item_list = sorted(list(item_set))
    item_dict = dict()
    for i in range(1, len(item_set) + 1):
        item = item_list[i - 1]
        item_dict[item] = i

    train_x_new = list()
    train_y_new = list()
    test_x_new = list()
    test_y_new = list()

    for items in train_x:
        new_list = []
        for item in items:
            new_item = item_dict[item]
            new_list.append(new_item)
        train_x_new.append(new_list)
    for item in train_y:
        new_item = item_dict[item]
        train_y_new.append(new_item)
    for items in test_x:
        new_list = []
        for item in items:
            new_item = item_dict[item]
            new_list.append(new_item)
        test_x_new.append(new_list)
    for item in test_y:
        new_item = item_dict[item]
        test_y_new.append(new_item)

    # --------重新统计renumber之后的itemset-----------#
    item_set = set()
    for items in train_x_new:
        for id_ in range(len(items)):
            item_set.add(items[id_])

    for item in train_y_new:
        item_set.add(item)

    for items in test_x_new:
        for id_ in range(len(items)):
            item_set.add(items[id_])

    for item in test_y_new:
        item_set.add(item)

    return train_x_new, train_y_new, test_x_new, test_y_new, item_set, item_dict  # renumbered 之后的对应关系


def frequencyAdj(num_node, nums, neighbors):
    adj = torch.zeros(num_node, num_node)
    for idx, i in enumerate(neighbors):
        sum = 0
        for idx2, j in enumerate(i):
            if j != 0:
                sum += 1
            if j == 0:
                continue
            if adj[idx][j] == 0 and idx != j:  # 去掉item自己
                adj[idx][j] = nums[idx][idx2]
                # adj[j][idx] = nums[idx][idx2]
            else:
                continue

    adj = adj / (torch.sum(adj, -1) + 1e-7).unsqueeze(-1)

    adj = torch.where(adj == 0, 99, adj)
    adj = torch.where(adj < 0.25, 1, adj)
    adj = torch.where(adj < 0.50, 2, adj)
    adj = torch.where(adj < 0.75, 3, adj)
    adj = torch.where(adj < 1, 4, adj)
    adj = torch.where(adj == 99, 0, adj)

    items_num = adj.shape[0]
    adj_diag = torch.eye(items_num, items_num)
    adj = adj + adj_diag
    return adj  # , trans_to_cuda(torch.Tensor(adj_norm))

# Recall, also HR
def get_recall(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """
    truths = truth.expand_as(pre)
    hits = (pre == truths).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (pre == truths).nonzero().size(0)
    recall = n_hits / truths.size(0)
    return recall


# MRR
def get_mrr(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B, 1) real label
    :return: MRR(Float), the mrr score
    """
    targets = truth.view(-1, 1).expand_as(pre)
    # ranks of the targets, if it appears in your indices
    hits = (targets == pre).nonzero()
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / targets.size(0)
    return mrr


def get_hr(recall_items: list, true_items: list):
    N = len(recall_items)
    M = len(true_items)
    if N == 0 or M == 0:
        return 0
    hit_num = 0
    for item in true_items:
        if item in recall_items:
            hit_num += 1
    return hit_num / M
