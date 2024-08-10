import os
import pickle
import datetime
import argparse
from model import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='datasets/', help='path of datasets')
parser.add_argument('--dataset', default='Tmall-2', help='Tmall-2/diginetica-2/retailrocket-2')
parser.add_argument('--epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--batchSize', type=int, default=100, help='batch size')
parser.add_argument('--embedding', type=int, default=100, help='embedding size of items')
parser.add_argument('--posembedding', type=int, default=100, help='embedding size of position embedding')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=int, default=0.6,  help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=1,  help='the number of steps after which the learning rate decay.')
parser.add_argument('--layer', type=float, default=1, help='model depth')
parser.add_argument('--category', default='cluster', help='cluster')
parser.add_argument('--k', type=float, default=100, help='number of cluster centroids')
parser.add_argument('--threshold', type=float, default=0.0, help='threshold of cosine similarity')
parser.add_argument('--p', type=float, default=0.4, help='probability of item masking')
parser.add_argument('--theta', type=float, default=0.2, help='scale of cross-scale contrastive learning')
parser.add_argument('--temp', type=float, default=0.1, help='temperature for cross-scale contrastive learning')
parser.add_argument('--dropout_in', type=float, default=0.0, help='Dropout rate at input layer.')
parser.add_argument('--dropout_hid', type=float, default=0.0, help='Dropout rate at hidden layer.')
parser.add_argument('--isvalidation', action='store_true', help='validation')

opt = parser.parse_args(args=[])
print(opt)
if torch.cuda.is_available  ():
    print('model is on gpu')
else:
    print('model is on cpu')

# -------------------- data load --------------------- #
train = pickle.load(open(opt.path + opt.dataset +'/train.txt', 'rb'))
test = pickle.load(open(opt.path + opt.dataset +'/test.txt', 'rb'))
neighbors = pickle.load(open(opt.path + opt.dataset +'/adj12' + '.pkl', 'rb'))  # [1:]  # 去掉item0
nums = pickle.load(open(opt.path + opt.dataset +'/num12' + '.pkl', 'rb'))  # [1:]

# ----------------- data preprocess ------------------ #
train_x = train[0]
train_y = train[1]
test_x = test[0]
test_y = test[1]

train_x, train_y, test_x, test_y, item_set, item_dict = renumberItems(train_x, train_y, test_x, test_y)
all_items = torch.Tensor([i for i in item_set]).long()
num_node = len(all_items) + 1
print('the number of nodes is :{}'.format(num_node))
print('the number of clusters is is :{}'.format(opt.k))

train = (train_x, train_y)  # combine the sequence+target and label
test = (test_x, test_y)  # combine the sequence+target and label

if opt.isvalidation:
    print('start validation')
    train_data, valid_data = split_validation(train, 0.2)
    test_data = valid_data
else:
    train_data = train
    test_data = test

adj_freq = frequencyAdj(num_node, nums, neighbors)

length = len(max(train_x, key=len))
train_data = Data(train_data, adj_freq, length)
test_data = Data(test_data, adj_freq, length)
max_len = train_data.__len__()

# ----------------- define the model ------------------ #

model = trans_to_cuda(HearInt(opt, num_node))

# ------------------ Train and Test ------------------- #
print('--------------------------Start Training--------------------------')
train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                           shuffle=True, pin_memory=True)
opti = model.optimizer
criterion = nn.CrossEntropyLoss().cuda()

best_result = [0, 0]
best_epoch = [0, 0]
best_result = 0
best_model_result = []
for epoch in range(opt.epoch):
    model.train()
    print('---------------This is epoch: {}---------------'.format(epoch))
    for step, data in enumerate(tqdm(train_loader)):
        target = data[0]
        if opt.category == 'cluster' and step % 5 == 0:
            model.e_step()
        Result, CL_Loss = forward(model, data)
        NP_Loss = trans_to_cuda(criterion(Result, trans_to_cuda(target - 1).long()))

        Loss = NP_Loss + CL_Loss
        if step % 1000==0:
            print("CL_Loss:{}".format(CL_Loss))
            print("Loss:{}".format(Loss))

        opti.zero_grad()
        Loss.backward()
        opti.step()


    model.scheduler.step()

    print('-------------------Start Predicting---------------------: ', datetime.datetime.now())

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    y_pre_1_all = torch.LongTensor().cuda()
    y_pre_1_all_10 = torch.LongTensor()
    y_pre_1_all_5 = torch.LongTensor()

    predict_nums = [i + 1 for i in range(20)]
    hit = [[] for _ in range(len(predict_nums))]
    mrr = [[] for _ in range(len(predict_nums))]
    ndcg = [[] for _ in range(len(predict_nums))]

    for data in test_loader:
        with torch.no_grad():
            max_len_test = test_data.get_max_len()
            y_pre_1 = model.predict(data, 20)

            y_pre_1_all = torch.cat((y_pre_1_all, y_pre_1), 0)
            y_pre_1_all_10 = torch.cat((y_pre_1_all_10, y_pre_1.cpu()[:, :10]), 0)
            y_pre_1_all_5 = torch.cat((y_pre_1_all_5, y_pre_1.cpu()[:, :5]), 0)

    recall = get_recall(y_pre_1_all, trans_to_cuda(torch.Tensor(test_y)).long().unsqueeze(1) - 1)
    recall_10 = get_recall(y_pre_1_all_10, torch.Tensor(test_y).unsqueeze(1) - 1)
    recall_5 = get_recall(y_pre_1_all_5, torch.Tensor(test_y).unsqueeze(1) - 1)
    mrr = get_mrr(y_pre_1_all, trans_to_cuda(torch.Tensor(test_y)).long().unsqueeze(1) - 1)
    mrr_10 = get_mrr(y_pre_1_all_10, torch.Tensor(test_y).unsqueeze(1) - 1)
    mrr_5 = get_mrr(y_pre_1_all_5, torch.Tensor(test_y).unsqueeze(1) - 1)

    if best_result < recall:
        best_result = recall
        best_model_result = [recall_5, recall_10, recall, mrr_5, mrr_10, mrr]
        #state_dict = {"model": model.state_dict(), "embedding": model.embedding.state_dict()}
        ##torch.save(state_dict,'saved/' + 'dg_hearint.pth')

    print("Results of LocalSession:\n")
    print("Recall@20: " + "%.4f" % recall + " Recall@10: " + "%.4f" % recall_10 + "  Recall@5:" + "%.4f" % recall_5)
    print("MRR@20:" + "%.4f" % mrr.tolist() + " MRR@10:" + "%.4f" % mrr_10.tolist() + " MRR@5:" + "%.4f" % mrr_5.tolist())
    print("\n")
    print("Best Results:\n")
    print(best_model_result)

    torch.cuda.empty_cache()