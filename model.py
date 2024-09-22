import faiss
import torch
import math
import numpy as np
from utils import *
from torch import nn
from torch.nn import Module
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j, target):
        SIZE = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(representations, representations.t().contiguous())
        sim_ij = torch.diag(similarity_matrix, SIZE)
        sim_ji = torch.diag(similarity_matrix, -SIZE)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        negatives_mask = trans_to_cuda(~torch.eye(SIZE * 2, SIZE * 2, dtype=bool)).float()
        negative_sample_mask = self.sample_mask(target)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        denominator = negative_sample_mask * denominator  
        loss_partial = -torch.log(nominator / (torch.sum(denominator, dim=1) + 1e-7))
        loss = torch.sum(loss_partial) / (2 * SIZE)
        return loss

    def sample_mask(self, targets):
        targets = targets.cpu().numpy()
        targets = np.concatenate([targets, targets])

        cl_dict = {}
        for i, target in enumerate(targets):
            cl_dict.setdefault(target, []).append(i)
        mask = np.ones((len(targets), len(targets)))
        for i, target in enumerate(targets):
            for j in cl_dict[target]:
                if abs(j - i) != len(targets) / 2:  # 防止mask将正样本的位置置为零
                    mask[i][j] = 0
        return trans_to_cuda(torch.Tensor(mask)).float()


class LocalAggregator(nn.Module):
    def __init__(self, dim, dropout=0.1, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.hidden = int(dim / 2)
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.dp = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        self.linear = nn.Linear(2 * dim, dim)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, hidden, adj, adjND, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        h = self.dp(h)
        output = torch.matmul(alpha, h)
        # alpha = torch.sigmoid(self.linear(torch.cat((hidden, output), -1)))
        # final_out = alpha * output + (1 - alpha) * hidden

        return output


class SelfAttention(nn.Module):
    def __init__(self, item_dim, dp):
        super(SelfAttention, self).__init__()
        self.dim = item_dim
        self.q = nn.Linear(self.dim, self.dim)
        self.k = nn.Linear(self.dim, self.dim)
        self.v = nn.Linear(self.dim, self.dim)
        self.atten_dp = nn.Dropout(dp)
        self.LN = nn.LayerNorm(self.dim)

    def forward(self, q, k, v, mask=None):
        q_ = torch.selu(self.q(q))
        k_ = k
        v_ = v
        scores = torch.matmul(q_, k_.transpose(1, 2)) / math.sqrt(self.dim)
        mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
        scores = scores.masked_fill(mask == 0, -np.inf)
        # alpha_ent_expand, alpha_ent = self.get_alpha(x=q[:, -1, :])
        scores = F.softmax(scores, dim=-1)  # entmax_bisect(scores, alpha_ent, dim=-1)
        v_ = self.atten_dp(v_)
        att_v = torch.matmul(scores, v_)  # B, seq, dim
        return att_v


class FeedForward(nn.Module):
    def __init__(self, item_dim, dp):
        super(FeedForward, self).__init__()
        self.dim = item_dim
        self.fn1 = nn.Linear(self.dim, self.dim)
        self.fn2 = nn.Linear(self.dim, self.dim)
        self.act1 = nn.SELU()
        self.ln = nn.LayerNorm(self.dim)
        self.dp = nn.Dropout(dp)

    def forward(self, inputs):
        hidden = self.fn2(self.act1(self.fn1(inputs)))
        hidden_out = inputs + self.dp(hidden)
        return hidden_out


class SpatialEncoder(nn.Module):
    def __init__(self, dim, dp):
        super(SpatialEncoder, self).__init__()
        self.dim = dim
        self.dp = dp
        self.agg1 = LocalAggregator(self.dim, self.dp)
        self.agg2 = LocalAggregator(self.dim, self.dp)

    def forward(self, input1, input2, adjS, adjND):
        output1 = self.agg1(input1, adjS, adjS)
        output2 = self.agg2(input2, adjND, adjND)
        return output1, output2


class TemporalEncoder(nn.Module):
    def __init__(self, dim, dp):
        super(TemporalEncoder, self).__init__()
        self.dim = dim
        self.dp = dp
        self.self_atten = SelfAttention(self.dim, 0.1)
        self.ff = FeedForward(self.dim, 0.1)

    def forward(self, inputs, mask):
        satten_out = self.self_atten(inputs, inputs, inputs, mask)
        satten_out = self.ff(satten_out)
        return satten_out


class TempSpatialEncoder(nn.Module):
    def __init__(self, opt, item_dim, pos_embedding, layers, dropout_in, dropout_hid):
        super(TempSpatialEncoder, self).__init__()
        self.opt = opt
        self.dim = item_dim
        self.layers = layers
        self.dpin = dropout_in
        self.dphid = dropout_hid
        self.pos_embedding = pos_embedding

        self.TempEnds = []
        self.SpatEnds = []
        for i in range(self.layers):
            Te = TemporalEncoder(self.dim, self.dphid)
            Sp = SpatialEncoder(self.dim, self.dphid)
            self.add_module('temporal_encoder_{}'.format(i), Te)
            self.add_module('spatial_encoder_{}'.format(i), Sp)
            self.TempEnds.append(Te)
            self.SpatEnds.append(Sp)

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.final_cat = nn.Linear(self.dim * 2, self.dim)
        self.cat = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.fn = nn.Linear(self.dim * 2, self.dim)
        self.gnn_cat = nn.Linear(self.dim * 2, self.dim)

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=False)
        self.dpin1 = nn.Dropout(self.dpin)
        self.dpin2 = nn.Dropout(self.dpin)

    def forward(self, seq_input, gnn_input, adj, input_index, mask):
        adj = adj[0]
        adjND = adj[1]
        seq_input_mask = mask[0]
        gnn_input_mask = mask[1]
        seq_mask = mask[2]
        gnn_mask = mask[3]
        len = seq_input.shape[1]
        batch_size = seq_input.shape[0]

        pos_emb = self.pos_embedding.weight[:len]  # Seq_len x Embed_dim
        pos = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # Batch x Seq_len x Embed_dim

        if self.opt.dataset == 'Tmall-2':
            seq_input = seq_input * seq_input_mask
        else:
            seq_input = (seq_input + pos) * seq_input_mask

        seq_out = self.seq_encoder(seq_input, seq_mask)
        seq_out_q = seq_out

        gnn_input = gnn_input * gnn_input_mask
        gnn_out = self.gnn_encoder(gnn_input, adj, adjND, input_index)
        gnn_out_q = self.SoftAtten(gnn_out, pos, gnn_mask)

        gf = torch.sigmoid(self.cat(torch.cat([seq_out_q, gnn_out_q], dim=-1)))
        session_rep = gf * seq_out_q + (1 - gf) * gnn_out_q

        return session_rep

    def predict(self, seq_input, gnn_input, adj, adjND, input_index, mask):
        seq_input_mask = mask[0]
        gnn_input_mask = mask[1]
        seq_mask = mask[2]
        gnn_mask = mask[3]

        len = seq_input.shape[1]
        batch_size = seq_input.shape[0]
        pos_emb = self.pos_embedding.weight[:len]  # Seq_len x Embed_dim
        pos = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # Batch x Seq_len x Embed_dim

        if self.opt.dataset == 'Tmall-2':
            seq_input = seq_input * seq_input_mask
        else:
            seq_input = (seq_input + pos) * seq_input_mask

        seq_out = self.seq_encoder(seq_input, seq_mask)
        seq_out_q = seq_out

        gnn_input = gnn_input * gnn_input_mask
        gnn_out = self.gnn_encoder(gnn_input, adj, adjND, input_index)
        gnn_out_q = self.SoftAtten(gnn_out, pos, gnn_mask)

        gf = torch.sigmoid(self.cat(torch.cat([seq_out_q, gnn_out_q], dim=-1)))
        session_rep = gf * seq_out_q + (1 - gf) * gnn_out_q

        return session_rep

    def avgPool(self, input, mask):
        input = input * mask
        input_sum = torch.sum(input, dim=1)
        dev = torch.sum(mask.squeeze(-1).int(), dim=-1).unsqueeze(-1).repeat(1, input.shape[-1])
        return input_sum / dev

    def seq_encoder(self, inputs, mask):
        inputs = self.dpin1(inputs)
        last = inputs
        for i in range(self.layers):
            output = self.TempEnds[i](last, mask)
            last = output
        return output[:, -1, :]

    def gnn_encoder(self, session_items, adj, adjND, inputs_index):
        session_items = self.dpin2(session_items)
        last_0 = session_items
        last_1 = session_items
        for i in range(self.layers):
            output_0, output_1 = self.SpatEnds[i](last_0, last_1, adj, adjND)
            last_0 = output_0
            last_1 = output_1

        for i in range(len(output_0)):
            output_0[i] = output_0[i][inputs_index[i]]
            output_1[i] = output_1[i][inputs_index[i]]

        hidden = torch.cat((output_0, output_1), dim=-1)
        alpha = torch.sigmoid(self.gnn_cat(hidden))
        output = alpha * output_0 + (1 - alpha) * output_1

        return output

    def SoftAtten(self, hidden, pos, mask):
        mask = mask.float().unsqueeze(-1)  # Batch x Seq_len x 1

        batch_size = hidden.shape[0]  # Batch
        lens = hidden.shape[1]  # Seq_len
        pos_emb = pos  # Batch x Seq_len x Embed_dim

        hs = torch.sum(hidden * mask, -2) / (torch.sum(mask, 1) + 1e-7)
        hs = hs.unsqueeze(-2).repeat(1, lens, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nl = hidden[:, -1, :].unsqueeze(-2).repeat(1, lens, 1)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        return select


class HearInt(Module):
    def __init__(self, opt, num_node):
        super(HearInt, self).__init__()
        # hyper-parameter definition
        self.opt = opt
        self.item_dim = opt.embedding
        self.pos_dim = opt.posembedding
        self.batch_size = opt.batchSize

        self.num_node = num_node
        self.temp_cross = opt.temp
        self.theta = opt.theta

        self.dropout_in = opt.dropout_in
        self.dropout_hid = opt.dropout_hid

        self.category = opt.category
        self.layers = opt.layer
        self.item_centroids = None
        self.item_2cluster = None
        self.k = opt.k
        self.p = opt.p
        self.threshold = opt.threshold

        # embedding definition
        self.embedding = nn.Embedding(self.num_node, self.item_dim, max_norm=1.5)
        self.pos_embedding = nn.Embedding(self.num_node, self.item_dim, max_norm=1.5)

        # component definition
        self.model = TempSpatialEncoder(self.opt, self.item_dim, self.pos_embedding, self.layers, self.dropout_in,
                                        self.dropout_hid)
        self.crossCl = ContrastiveLoss(self.batch_size, self.temp_cross)

        # training definition
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step,
                                                         gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.item_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, target, inputs, adjND, adjS, session_items, inputs_index):

        # ------------------------------ interest-orient decoupling module -------------------------------------#
        x = inputs.detach().cpu().numpy()
        short_subsession = self.short_term_decouple(inputs)
        long_subsession, adjS, adjND = self.long_term_decouple(adjS, adjND, inputs, session_items)

        # ---------------------------------- input and mask generator --------------------------------------#
        gnn_seq = torch.ones_like(long_subsession)
        for i in range(len(session_items)):
            gnn_seq[i] = long_subsession[i][inputs_index[i]]
        seq_seq_mask = (short_subsession != 0).float()
        gnn_seq_mask = (gnn_seq != 0).float()

        short_inputs_embed = self.embedding.weight[short_subsession]
        long_subsession_embed = self.embedding.weight[long_subsession]

        timeline_mask_0 = trans_to_cuda(torch.BoolTensor(short_subsession.detach().cpu().numpy() == 0))
        mask_crop_0 = ~timeline_mask_0.unsqueeze(-1)  # broadcast in last dim
        timeline_mask_2 = trans_to_cuda(torch.BoolTensor(long_subsession.detach().cpu().numpy() == 0))
        mask_crop_1 = ~timeline_mask_2.unsqueeze(-1)  # broadcast in last dim

        mask = [mask_crop_0, mask_crop_1, seq_seq_mask, gnn_seq_mask]
        adj = [adjS, adjND]

        # ---------------------------------- Attention and GNN encoders -------------------------------------- -
        #
        # #
        output = self.model(short_inputs_embed, long_subsession_embed, adj, inputs_index, mask)

        # ------------------------------------------Cross-scale CL--------------------------------------------- #

        item2cluster = self.item_2cluster[target]
        target_features = self.item_centroids[item2cluster]
        crossCl = self.crossCl(output, target_features, item2cluster)

        # ------------------------------------------Compute Score---------------------------------------------- #
        Result1 = self.decoder(output)

        return Result1, crossCl * self.theta  

    def short_term_decouple(self, input):
        input = input.detach().cpu()
        croped_input = self.crop(input)
        return croped_input

    def crop(self, input):
        item_attributes = self.item_2cluster[input].detach().cpu()
        mask1 = torch.where(input != 0, 1, 0).detach().cpu()

        item_attributes_embed = self.item_centroids[item_attributes].detach().cpu()
        mask2 = mask1.unsqueeze(-1).repeat(1, 1, item_attributes_embed.shape[-1])

        item_attributes = mask1 * item_attributes
        item_attributes_embed = mask2 * item_attributes_embed

        p = 1 - self.p
        last_click_embed = item_attributes_embed[:, -1].unsqueeze(1).repeat(1, item_attributes_embed.shape[1],
                                                                            1)
        sim_matrix = torch.cosine_similarity(item_attributes_embed, last_click_embed, dim=2)
        mask2 = 1 - torch.where(sim_matrix > 0, 1, 0)
        rand_matrix = torch.rand(item_attributes.shape[0], item_attributes.shape[1])
        mask_matrix_reverse = rand_matrix * mask2
        mask = 1 - torch.where(mask_matrix_reverse < p, 0, 1)

        return trans_to_cuda(input * mask).long()

    def long_term_decouple(self, adj, adjND, inputs, items):
        adj = adj.detach().cpu()
        adjND = adjND.detach().cpu()
        items = items.detach().cpu()
        p = 1 - self.p
        maskpad1 = (inputs != 0).cpu()
        maskpad2 = (items != 0).cpu()
        item_cat_seq = self.item_2cluster[inputs].detach().cpu() * maskpad1
        item_cat_embed = self.item_centroids[item_cat_seq].detach().cpu()
        item_cat_items = self.item_2cluster[items].detach().cpu() * maskpad2
        most_click = []
        pad_cat = torch.tensor(0)
        for i in item_cat_seq:
            weights = torch.where(i != pad_cat, 1, 0)
            most_click.append(torch.argmax(torch.bincount(i, weights)))
        most_click = torch.tensor(most_click)
        most_clicks = most_click.unsqueeze(-1).repeat(1, adj.shape[-1])
        most_clicks_embed = self.item_centroids[most_clicks].detach().cpu()
        rand_matrix = torch.rand(item_cat_seq.shape[0], item_cat_seq.shape[1])

        sim_matrix = torch.cosine_similarity(item_cat_embed, most_clicks_embed, dim=2)
        mask2 = 1 - torch.where(sim_matrix > 0, 1, 0)
        mask_matrix_reverse = rand_matrix * mask2
        mask = 1 - torch.where(mask_matrix_reverse < p, 0, 1)

        mask_col = mask.unsqueeze(1).repeat(1, mask.shape[-1], 1)
        mask_row = mask.unsqueeze(-1).repeat(1, 1, mask.shape[-1])
        adj = adj * mask_col * mask_row
        adjND = adjND * mask_col * mask_row
        items = items * mask

        return trans_to_cuda(items), trans_to_cuda(adj), trans_to_cuda(adjND)  # , trans_to_cuda(items)

    def e_step(self):
        items_embedding = self.embedding.weight.detach().cpu().numpy()
        self.item_centroids, self.item_2cluster = self.run_kmeans(items_embedding[:])

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=x.shape[-1], niter=50, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        self.cluster_cents = cluster_cents
        _, I = kmeans.index.search(x, 1)
        self.items_cents = I

        # convert to cuda Tensors for broadcast
        centroids = trans_to_cuda(torch.Tensor(cluster_cents))
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = trans_to_cuda(torch.LongTensor(I).squeeze())
        return centroids, node2cluster

    def decoder(self, select):
        l_c = (select / torch.norm(select, dim=-1).unsqueeze(1))
        l_emb = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)
        z = 13 * torch.matmul(l_c, l_emb.t())

        return z

    def predict(self, data, k):

        target, x_test, adjS, adjND, session_items, inputs_index = data
        inputs = trans_to_cuda(x_test).long()
        adjS = trans_to_cuda(adjS).float()
        adjND = trans_to_cuda(adjND).float()
        session_items = trans_to_cuda(session_items).long()

        gnn_seq = torch.ones_like(session_items)
        for i in range(len(session_items)):
            gnn_seq[i] = session_items[i][inputs_index[i]]

        gnn_seq_mask = (gnn_seq != 0).float()
        seq_seq_mask = (inputs != 0).float()

        timeline_mask_2 = trans_to_cuda(torch.BoolTensor(inputs.detach().cpu().numpy() == 0))
        mask_crop_2 = ~timeline_mask_2.unsqueeze(-1)  # broadcast in last dim
        timeline_mask_1 = trans_to_cuda(torch.BoolTensor(session_items.detach().cpu().numpy() == 0))
        mask_crop_1 = ~timeline_mask_1.unsqueeze(-1)  # broadcast in last dim

        inputs_embed = self.embedding.weight[inputs]
        session_items_embed = self.embedding.weight[session_items]

        mask = [mask_crop_2, mask_crop_1, seq_seq_mask, gnn_seq_mask]

        output = self.model.predict(inputs_embed, session_items_embed, adjS, adjND,
                                    inputs_index,
                                    mask)
        result1 = self.decoder(output)
        rank1 = torch.argsort(result1, dim=1, descending=True)
        return rank1[:, 0:k]


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
        # return variable
    else:
        return variable


def forward(model, data):
    target, u_input, adjS, adjND, session_items, u_input_index = data

    session_items = trans_to_cuda(session_items).long()
    u_input = trans_to_cuda(u_input).long()
    adjND = trans_to_cuda(adjND).float()
    adjS = trans_to_cuda(adjS).float()
    target = trans_to_cuda(target).long()
    Result1, FarL = model(target, u_input, adjND, adjS, session_items, u_input_index)
    return Result1, FarL

