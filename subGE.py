import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import random
from GNN import *
import tqdm
import attention
from RL import *
from sklearn.metrics import roc_auc_score, f1_score


hid_units = [16]
n_heads = [6, 1]


def preprocess(dataset):
    adj = np.load('D:/subGE/' + dataset + '/origin/adj.npy', allow_pickle=True)
    feature = np.load('D:/subGE/' + dataset + '/origin/features.npy', allow_pickle=True)
    subadj = np.load('D:/subGE/' + dataset + '/origin/sub_adj.npy', allow_pickle=True)
    label = np.load('D:/subGE/' + dataset + '/origin/graphs_label.npy', allow_pickle=True)

    new_label = np.array([np.argmax(one_hot) for one_hot in label])
    label_index = []
    for i in range(label.shape[-1]):
        tmp = np.where(new_label == i)
        label_index.append(tmp)
    return feature, new_label, label.shape[1], np.ones([subadj.shape[0], subadj.shape[1]]), subadj, adj, subadj.shape[1], subadj.shape[-1], label_index


def get_dsi_index(label_index, train_t):
    dsi_index = []
    for t in train_t:
        temp = list(range(len(label_index)))
        temp.remove(t)
        temp_label = label_index[(random.sample(temp, 1)[0])]
        dsi_index.append(random.sample(list(temp_label[0]), 1)[0])
    return dsi_index


def divide(data, label, sub_adj, sub_mask, test_begin_index, test_end_index, label_index):
    data_size = data.shape[0]
    train_x = np.concatenate([data[0:test_begin_index], data[test_end_index:data_size]])
    train_t = np.concatenate([label[0:test_begin_index], label[test_end_index:data_size]])
    train_sadj = np.concatenate([sub_adj[0:test_begin_index], sub_adj[test_end_index:data_size]])
    train_mask = np.concatenate([sub_mask[0:test_begin_index], sub_mask[test_end_index:data_size]])

    dsi_index = get_dsi_index(label_index, train_t)
    train_x_dsi = data[dsi_index]
    train_t_dsi = label[dsi_index]
    train_sadj_dsi = sub_adj[dsi_index]
    train_mask_dsi = sub_mask[dsi_index]

    test_x = data[test_begin_index:test_end_index]
    test_t = label[test_begin_index:test_end_index]
    test_sadj = sub_adj[test_begin_index:test_end_index]
    test_mask = sub_mask[test_begin_index:test_end_index]

    dsi_index = get_dsi_index(label_index, test_t)
    test_x_dsi = data[dsi_index]
    test_t_dsi = label[dsi_index]
    test_sadj_dsi = sub_adj[dsi_index]
    test_mask_dsi = sub_mask[dsi_index]

    return train_x, train_t, train_sadj, train_mask, train_x_dsi, train_t_dsi, train_sadj_dsi, train_mask_dsi, test_x, test_t, test_sadj, test_mask, test_x_dsi, test_t_dsi, test_sadj_dsi, test_mask_dsi


def load_batch(x, sadj, t, mask, train_x_dsi, train_sadj_dsi, train_t_dsi, train_mask_dsi, i, batch_size):
    data_size = x.shape[0]
    if i + batch_size > data_size:
        index = [j for j in range(i, data_size)]
        dsi_index = [j for j in range(i, data_size)]
    else:
        index = [j for j in range(i, i + batch_size)]
        dsi_index = [j for j in range(i, i + batch_size)]
    return x[index], sadj[index], t[index], mask[index], train_x_dsi[dsi_index], train_sadj_dsi[dsi_index], train_t_dsi[dsi_index], train_mask_dsi[dsi_index]


class subGE(object):
    def __init__(self, session, embedding, ncluster, num_subg, subg_size, batch_size, learning_rate, momentum):
        self.sess = session
        self.ncluster = ncluster
        self.embedding = embedding
        self.num_subg = num_subg
        self.subg_size = subg_size
        self.batch_size = batch_size
        self.output_dim = [32]
        self.GIN_dim = [16]
        self.SAGE_dim = [32]
        self.sage_k = 1
        self.lr = learning_rate
        self.mom = momentum

        self.build_placeholders()
        self.forward_propagation()

        train_var = tf.compat.v1.trainable_variables()
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), train_var)
        self.pred = tf.compat.v1.to_int32(tf.argmax(self.probabilities, 1))
        correct_prediction = tf.equal(self.pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.auc = tf.compat.v1.py_func(roc_auc_score, (self.labels, self.pred), tf.double)
        self.f1 = tf.compat.v1.py_func(f1_score, (self.labels, self.pred), tf.double)
        self.optimizer = tf.compat.v1.train.MomentumOptimizer(self.lr, self.mom).minimize(self.loss + self.l2)
        self.init = tf.compat.v1.global_variables_initializer()
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())


    def fcn(self, inputs, input_dim, output_dim, activation=None):
        W = tf.Variable(tf.random.truncated_normal(
            [input_dim, output_dim], stddev=0.1))
        b = tf.Variable(tf.zeros([output_dim]))
        XWb = tf.matmul(inputs, W) + b

        if (activation == None):
            outputs = XWb
        else:
            outputs = activation(XWb)
        return outputs

    def build_placeholders(self):
        self.sub_adj = (tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_subg, self.subg_size, self.subg_size)))
        self.sub_feature = (tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_subg, self.subg_size, self.embedding)))
        self.sub_feature_dsi = (tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_subg, self.subg_size, self.embedding)))
        self.sub_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_subg))
        self.sub_mask_dsi = tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_subg))
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=(None))
        self.label_mi = tf.compat.v1.placeholder(tf.int32, shape=(None, 2 * self.num_subg))
        self.lr = tf.compat.v1.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.compat.v1.placeholder(tf.float32, [], 'momentum')
        self.dropout = tf.compat.v1.placeholder_with_default(0.5, shape=())
        self.top_k = tf.compat.v1.placeholder_with_default(0.5, shape=())

    def sub_GCN(self):
        gcn_outs = []
        W = tf.Variable(tf.random.truncated_normal([self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            self.d_matrix = self.sub_adj[:, i, :, :]
            gcn_out = GCN(self.sub_feature[:, i, :, :], self.d_matrix, self.output_dim, dropout=0.5).build()
            gcn_out = tf.matmul(tf.transpose(gcn_out, [0, 2, 1]), W)
            gcn_outs.append(tf.reshape(gcn_out, [-1, 1, gcn_out.shape[1]]))
        self.gcn_result = tf.concat(gcn_outs, 1) 

        gcn_outs_dsi = []
        W_dsi = tf.Variable(tf.random.truncated_normal(
            [self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            gcn_out_dsi = GCN(self.sub_feature_dsi[:, i, :, :], self.d_matrix_dsi, self.output_dim, dropout=0.5).build()
            gcn_out_dsi = tf.matmul(tf.transpose(gcn_out_dsi, [0, 2, 1]), W_dsi)
            gcn_outs_dsi.append(tf.reshape(gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def sub_GAT(self):
        gcn_outs = []
        W = tf.Variable(tf.random.truncated_normal([self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            _, gcn_out, _, _, _, _ = attention.GAT().inference(self.sub_feature[:, i, :, :], self.ncluster, 0, self.output_dim, n_heads, tf.nn.leaky_relu, False, 1)
            gcn_out = tf.matmul(tf.transpose(gcn_out, [0, 2, 1]), W)
            gcn_outs.append(tf.reshape(gcn_out, [-1, 1, gcn_out.shape[1]]))
        self.gcn_result = tf.concat(gcn_outs, 1)

        gcn_outs_dsi = []
        W_dsi = tf.Variable(tf.random.truncated_normal([self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            _, gcn_out_dsi, _, _, _, _ = attention.GAT().inference(self.sub_feature_dsi[:, i, :, :], self.ncluster, 0, self.output_dim, n_heads, tf.nn.leaky_relu, False, 1)
            gcn_out_dsi = tf.matmul(tf.transpose(gcn_out_dsi, [0, 2, 1]), W_dsi)
            gcn_outs_dsi.append(tf.reshape(gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def sub_GIN(self):
        gcn_outs = []
        for i in range(self.num_subg):
            self.d_matrix = self.sub_adj[:, i, :, :]
            gcn_out = GCN(self.sub_feature[:, i, :, :], self.d_matrix, self.output_dim, dropout=0.5).build()
            for i in range(1):
                gcn_out = self.fcn(gcn_out, self.output_dim[i], self.GIN_dim[i])
            gcn_outs.append(tf.reshape(gcn_out, [-1, 1, gcn_out.shape[1] * gcn_out.shape[2]]))
        self.gcn_result = tf.concat(gcn_outs, 1)

        gcn_outs_dsi = []
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            gcn_out_dsi = GCN(self.sub_feature_dsi[:, i, :, :], self.d_matrix_dsi, self.output_dim, dropout=0.5).build()
            for i in range(1):
                gcn_out_dsi = self.fcn(gcn_out_dsi, self.output_dim[i], self.GIN_dim[i])
            gcn_outs_dsi.append(tf.reshape(gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1] * gcn_out_dsi.shape[2]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def sub_SAGE(self):
        gcn_outs = []
        for i in range(self.num_subg):
            self.d_matrix = self.sub_adj[:, i, :, :]
            gcn_out = SAGE(self.sub_feature[:, i, :, :], self.d_matrix, self.SAGE_dim, dropout=0.5).build()
            gcn_outs.append(tf.reshape(gcn_out, [-1, 1, gcn_out.shape[1] * gcn_out.shape[2]]))
        self.gcn_result = tf.concat(gcn_outs, 1)

        gcn_outs_dsi = []
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            gcn_out_dsi = SAGE(self.sub_feature_dsi[:, i, :, :], self.d_matrix_dsi, self.SAGE_dim, dropout=0.5).build()
            gcn_outs_dsi.append(tf.reshape(gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1] * gcn_out_dsi.shape[2]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def graph_gat(self):
        self.embedding_origin = self.gcn_result
        self.index, self.gatembedding, self.gat_result, self.embedding_topk, self.select_num, self.a_index = attention.GAT().inference(self.gcn_result, self.ncluster, 0, hid_units, n_heads, tf.nn.leaky_relu, False, self.top_k)

        self.index_dsi, self.gatembedding_dsi, self.gat_result_dsi, _, __, _ = attention.GAT().inference(self.gcn_result_dsi, self.ncluster, 0, hid_units, n_heads, tf.nn.leaky_relu, False, self.top_k)

    def bilinear(self, x, y, out_dim, flag):
        w = tf.ones([out_dim, x.shape[-1], y.shape[-1]])
        w = tf.expand_dims(w, 0)
        w = tf.tile(w, tf.stack([x.shape[1], 1, 1, 1]))

        x = tf.expand_dims(x, 2)
        x = tf.expand_dims(x, 4)
        x = tf.tile(x, tf.stack([1, 1, out_dim, 1, y.shape[-1]]))
        tmp = tf.reduce_sum(tf.multiply(x, w), 3)

        y = tf.expand_dims(y, 2)
        y = tf.tile(y, tf.stack([1, 1, out_dim, 1]))
        if flag:
            tmp = tf.tile(tmp, (1, 1, 1, 1))
        out = tf.reduce_sum(tf.multiply(tmp, y), 3)

        return out

    def forward_propagation(self):
        with tf.compat.v1.variable_scope('sub_gcn'):
            if sg_encoder == 'GCN':
                self.sub_GCN()
            elif sg_encoder == 'GAT':
                self.sub_GAT()
            elif sg_encoder == 'GIN':
                self.sub_GIN()
            elif sg_encoder == 'SAGE':
                self.sub_SAGE()

        with tf.compat.v1.variable_scope('graph_gat'):
            self.graph_gat()

        with tf.compat.v1.variable_scope('fn'):
            vote_layer = tf.reduce_sum(self.gat_result, axis=1) # 根据graph_gat中的各子图预测值，决定整体的预测
            w1 = tf.cast(tf.equal(self.labels, 1), dtype='float64') * 1
            w2 = tf.cast(tf.equal(self.labels, 0), dtype='float64') * 1
            w = w1+w2
            self.loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=vote_layer, weights=w)
            self.ww = tf.tile(tf.expand_dims(w, 1), (1, self.label_mi.shape[1]))
            global_embedding = tf.reduce_sum(self.gatembedding, axis=1) # 加和各子图embedding，得到全局embedding
            global_embedding = tf.tile(tf.expand_dims(global_embedding, 1), (1, self.num_subg, 1))
            sc = self.bilinear(global_embedding, self.gatembedding, 1, False)
            sc_dsi = self.bilinear(global_embedding, self.gatembedding_dsi, 1, True)
            sc_dsi = tf.reshape(sc_dsi, [-1, sc.shape[1], 1])
            self.sc = tf.sigmoid(tf.concat([sc, sc_dsi], 1))
            self.sc = tf.concat([self.sc, 1 - self.sc], 2)
            self.loss += MI_loss * tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=self.label_mi, logits=self.sc, weights=self.ww) 
            self.probabilities = tf.nn.softmax(vote_layer, name="probabilities")
            self.sub_true = tf.compat.v1.to_int32(tf.argmax(self.gat_result, 2))
            self.tmp_labels = tf.tile(tf.expand_dims(self.labels, 1), (1, self.num_subg))
            self.RL_reward = tf.reduce_mean(tf.cast(tf.equal(self.sub_true, self.tmp_labels), "float"))

    def train(self, batch_x, batch_x_dsi, batch_adj, batch_t, batch_t_mi, batch_mask, learning_rate=1e-3, momentum=0.9, k=0.5):
        feed_dict = {self.sub_feature: batch_x,
            self.sub_feature_dsi: batch_x_dsi,
            self.sub_adj: batch_adj,
            self.labels: batch_t,
            self.label_mi: batch_t_mi,
            self.sub_mask: batch_mask,
            self.sub_mask_dsi: batch_mask,
            self.lr: learning_rate,
            self.mom: momentum,
            self.top_k: k
        }
        _, loss, ww,acc, pred, sub_true, gcn_result, gat_result, sub_feature, index, embedding_origin, embedding_topk, select_num, a_index, sc = self.sess.run(
            [self.optimizer, self.loss,self.ww, self.accuracy, self.pred, self.sub_true, self.gcn_result, self.gat_result, self.sub_feature, self.index, self.embedding_origin, self.embedding_topk, self.select_num, self.a_index, self.sc], feed_dict=feed_dict)
        return loss, acc, pred, sub_true, gcn_result, gat_result, sub_feature, index, embedding_origin, embedding_topk, select_num, a_index

    def evaluate(self, batch_x, batch_x_dsi, batch_adj, batch_t, batch_t_mi, batch_mask, k):
        feed_dict = {
            self.sub_feature: batch_x,
            self.sub_feature_dsi: batch_x_dsi,
            self.sub_adj: batch_adj,
            self.labels: batch_t,
            self.label_mi: batch_t_mi,
            self.sub_mask: batch_mask,
            self.sub_mask_dsi: batch_mask,
            self.top_k: k
        }
        acc, pred, index, rl_reward, embedding_origin, embedding_topk, select_num, a_index, auc, f1 = self.sess.run(
            [self.accuracy, self.pred, self.index, self.RL_reward, self.embedding_origin, self.embedding_topk, self.select_num, self.a_index, self.auc, self.f1], feed_dict=feed_dict)
        return acc, pred, index, rl_reward, embedding_origin, embedding_topk, select_num, a_index, auc, f1


def main(params):
    global max_pool
    global MI_loss
    global sg_encoder
    folds = params['folds']
    dataset = params['dataset']
    num_epochs = params['num_epochs']
    max_pool = params['max_pool']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    MI_loss = params['MI_loss']
    sg_encoder = params['sg_encoder']
    k = params['start_k']

    feature, label, ncluster, sub_mask, sub_adj, vir_adj, num_subg, subg_size, label_index = preprocess(dataset)
    test_size = int(feature.shape[0] / folds)
    train_size = feature.shape[0] - test_size
    learning_rate = learning_rate
    with tf.compat.v1.Session() as sess:
        net = subGE(sess, feature.shape[-1], ncluster, num_subg, subg_size, batch_size, learning_rate, momentum)
        accs = []
        for fold in range(folds):
            sess.run(tf.compat.v1.global_variables_initializer())
            vir_acc_fold = []
            if fold < folds - 1:
                train_x, train_t, train_sadj, train_mask, train_x_dsi, train_t_dsi, train_sadj_dsi, train_mask_dsi, test_x, test_t, test_sadj, test_mask, test_x_dsi, test_t_dsi, test_sadj_dsi, test_mask_dsi \
                    = divide(feature, label, sub_adj, sub_mask, fold * test_size, fold * test_size + test_size, label_index)
            else:
                train_x, train_t, train_sadj, train_mask, train_x_dsi, train_t_dsi, train_sadj_dsi, train_mask_dsi,  test_x, test_t, test_sadj, test_mask, test_x_dsi, test_t_dsi, test_sadj_dsi, test_mask_dsi \
                    = divide(feature, label, sub_adj, sub_mask, feature.shape[0] - test_size, feature.shape[0], label_index)
            max_fold_acc = 0
            max_fold_auc = 0
            max_fold_f1 = 0
            k_step_value = round(0.5 / net.num_subg, 4)
            mdp = MDP(value=k_step_value, subgraph_num=net.num_subg, k=k)
            RL = QTable(actions=list(range(mdp.action_num)), learning_rate=0.02)
            k_record = []
            eva_acc_record = []

            tbar = tqdm.tqdm(range(num_epochs))
            train_acc_record = []
            train_loss_record = []
            endingRLEpoch = 0

            for epoch in tbar:
                train_loss = 0
                train_acc = 0
                batch_num = 0
                idx = np.random.permutation(feature.shape[2])
                for i in range(0, train_size, batch_size):
                    x_batch, sadj_batch, t_batch, mask_batch, x_batch_dsi, sadj_batch_dsi, t_batch_dsi, mask_batch_dsi = load_batch(train_x, train_sadj, train_t, train_mask, train_x_dsi, train_sadj_dsi, train_t_dsi, train_mask_dsi, i, batch_size)
                    t_batch_mi = [[1] * num_subg + [0] * num_subg] * len(t_batch)

                    loss, acc, pred, sub_true, gcn_result, gat_result, sub_feature, index, embedding_origin, embedding_topk, select_num, a_index = net.train(x_batch, x_batch_dsi, sadj_batch, t_batch, t_batch_mi, mask_batch, learning_rate, momentum, k)
                    limited_epoch = 20
                    delta_k = 0.04
                    if epoch >= 100 and (not isTerminal(k_record, limited_epochs=limited_epoch, delta_k=delta_k)):
                        k, reward = run_QL(mdp, RL, net, x_batch, x_batch_dsi, sadj_batch, t_batch, t_batch_mi, mask_batch, acc)
                        k_record.append(round(k, 4))
                        endingRLEpoch = epoch
                    else:
                        k_record.append(round(k, 4))

                    batch_num += 1
                    train_loss += loss
                    train_acc += acc

                    batch_num += 1
                    if i == 0:
                        all_mask = sub_true
                    else:
                        all_mask = np.concatenate([all_mask, sub_true], 0)
                test_t_mi = [[1] * num_subg + [0] * num_subg] * len(test_t)
                test_dsi = test_x[:, :, idx, :]
                eva_acc, eva_pred, eva_index, _, eva_embedding_origin, eva_embedding_topk, eva_selecnum, eva_a_index, auc, f1 = net.evaluate(test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, k)

                if eva_acc > max_fold_acc:
                    max_fold_acc = eva_acc
                    vir_acc_fold.append(eva_acc)

                if auc > max_fold_auc:
                    max_fold_auc = auc

                if f1 > max_fold_f1:
                    max_fold_f1 = f1

                train_loss_record.append(train_loss / batch_num)
                train_acc_record.append(eva_acc)
                tbar.set_description_str("folds {}/{}".format(fold + 1, folds))

                tbar.set_postfix_str("k:{:.2f}, loss: {:.2f}, best_acc:{:.4f}, best_auc:{:.4f}, best_f1_score:{:.4f}, RL:{}".format(k, train_loss / batch_num, max_fold_acc, max_fold_auc, max_fold_f1, endingRLEpoch))

                # save subgraphs
                if epoch == num_epochs-1:
                    adj_ids = np.arange(fold * test_size, fold * test_size + test_size)
                    np.save(fr'D:\subGE\bbbp_data\adj_ids_{fold}.npy', adj_ids)
                    k_shape = test_x.shape
                    k_sub_feature = np.zeros((k_shape[0], eva_selecnum, k_shape[2], k_shape[3]))
                    k_shape = test_sadj.shape
                    k_sub_adj = np.zeros((k_shape[0], eva_selecnum, k_shape[2], k_shape[3]))
                    for i in range(k_shape[0]):
                        k_sub_feature[i, ...] = test_x[i, eva_a_index[i, :eva_selecnum], ...]
                        k_sub_adj[i, ...] = test_sadj[i, eva_a_index[i, :eva_selecnum], ...]
                    np.save(fr'D:\subGE\bbbp_data\k_sub_feature_{fold}.npy', k_sub_feature)
                    np.save(fr'D:\subGE\bbbp_data\k_sub_adj_{fold}.npy', k_sub_adj)

            accs.append(max_fold_acc)
            # save data
            f = pd.DataFrame(k_record)
            f.to_csv(fr"D:\subGE\bbbp_data\k_{fold}.csv")
            
            f = pd.DataFrame(eva_acc_record)
            f.to_csv(fr"D:\subGE\bbbp_data\eva_acc_{fold}.csv")
            
            f = pd.DataFrame(train_acc_record)
            f.to_csv(fr"D:\subGE\bbbp_data\train_acc_{fold}.csv")
            
            f = pd.DataFrame(train_loss_record)
            f.to_csv(fr"D:\subGE\bbbp_data\train_loss_{fold}.csv")

        accs = np.array(accs)
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        ans = {
            "mean": mean,
            "std": std
        }
        ####################
        return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="bbbp_data")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_pool', type=float, default=0.06)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sg_encoder', type=str, default='GAT')
    parser.add_argument('--MI_loss', type=float, default=0)
    parser.add_argument('--start_k', type=float, default=1)  

    args = parser.parse_known_args()[0]

    params = {
    'dataset' : args.dataset,
    'folds' : 4,
    'num_epochs' : args.num_epoch,
    'batch_size' : args.batch_size,
    'max_pool' : args.max_pool,
    'learning_rate' : args.lr,
    'momentum' : args.momentum,
    'sg_encoder' : args.sg_encoder,
    'MI_loss' : args.MI_loss,
    'start_k' : args.start_k,
    }

    ans = main(params)
