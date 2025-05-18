# %%
import json
import copy
import numpy as np
import pathlib
import torch
import torch.nn as nn
import tqdm.auto
import datetime
from collections import Counter
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loss(nn.Module):
    def __init__(self, loss_type="cross_entropy", beta=None, samples_per_class=None, class_balanced=False):
        super().__init__()
        self.loss_type = loss_type
        self.class_balanced = class_balanced
        self.beta = beta
        self.samples_per_class = samples_per_class
        self.eps = 1e-7
    def forward(self, inputs, targets):
        if self.loss_type == "cross_entropy":
            if self.class_balanced:
                if self.samples_per_class is not None:
                    ef_num = []
                    for ccount in self.samples_per_class:
                        ef_num.append(1.0 - (self.beta ** ccount))
                    ef_num = np.array(ef_num)
                    class_weights = (1.0 - self.beta) / (ef_num + self.eps)
                    class_weights = class_weights / np.sum(class_weights) * len(self.samples_per_class)
                    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=inputs.device)
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                    loss = loss_fn(inputs, targets)
                    return loss
                else:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(inputs, targets)
                    return loss
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(inputs, targets)
                return loss
        return None

cell_count = 98
ds_type = 'Dev'
image_key = 'volt_BC_dev'
now = datetime.datetime.now()

class LSTMBlock2(nn.Module):
    def __init__(self, input_size, hidden_size=192, num_layers=1, dropout=0.0, proj_size=0):
        super().__init__()
        self._lstm_ = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if (num_layers > 1) else 0,
            proj_size=proj_size
        )
    def forward(self, x):
        y = self._lstm_(x)[0]
        y = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        y = y[0].gather(
            dim=1,
            index=(y[1] - 1).view(-1, 1, 1).expand(y[0].shape[0], 1, y[0].shape[2]).to(x.data.device)
        )
        return y.squeeze()

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MLSTM_FCN(nn.Module):
    def __init__(self, num_features, num_classes, conv_channels=[128, 256, 128], out_channels=128, conv_kernels=[8, 5, 3], lstm_hidden_states=256, num_layers=1, eps=1.0E-03, momentum=0.99, dropout=0.8, activation=nn.ReLU):
        super().__init__()
        self._conv1_ = nn.Conv1d(in_channels=num_features, out_channels=conv_channels[0], kernel_size=conv_kernels[0], padding='same')
        self._bn1_ = nn.BatchNorm1d(num_features=out_channels, eps=1.0E-03, momentum=0.99)
        self._act1_ = activation()
        self._se1_ = SELayer(channel=conv_channels[0])
        self._conv2_ = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=conv_kernels[1], padding='same')
        self._bn2_ = nn.BatchNorm1d(num_features=conv_channels[1], eps=1.0E-03, momentum=0.99)
        self._act2_ = activation()
        self._se2_ = SELayer(channel=conv_channels[1])
        self._conv3_ = nn.Conv1d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=conv_kernels[2], padding='same')
        self._bn3_ = nn.BatchNorm1d(num_features=conv_channels[2], eps=1.0E-03, momentum=0.99)
        self._act3_ = activation()
        self._pool_ = nn.AdaptiveAvgPool1d(1)
        self._lstm_ = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_states, num_layers=1, batch_first=True, dropout=0.8 if (num_layers > 1) else 0, proj_size=conv_channels[-1])
        self._drop_ = nn.Dropout(p=dropout)
        self._last_ = nn.AdaptiveAvgPool1d(num_classes)
    def forward(self, x):
        y_conv = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=-1.0)[0]
        y_conv = y_conv.permute((0, 2, 1))
        y_conv = self._conv1_(y_conv)
        y_conv = self._bn1_(y_conv)
        y_conv = self._act1_(y_conv)
        y_conv = self._se1_(y_conv)
        y_conv = self._conv2_(y_conv)
        y_conv = self._bn2_(y_conv)
        y_conv = self._act2_(y_conv)
        y_conv = self._se2_(y_conv)
        y_conv = self._conv3_(y_conv)
        y_conv = self._bn3_(y_conv)
        y_conv = self._act3_(y_conv)
        y_conv = self._pool_(y_conv)
        lstm_out, _ = self._lstm_(x)
        y_lstm = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, padding_value=-1.0)
        y_lstm = y_lstm[0].gather(
            dim=1,
            index=(y_lstm[1] - 1).view(-1, 1, 1).expand(y_lstm[0].shape[0], 1, y_lstm[0].shape[2]).to(x.data.device)
        )
        ret = torch.cat((y_conv.permute(0, 2, 1), y_lstm), dim=2)
        return self._last_(ret.permute(0, 2, 1).squeeze())

class LSTM_FCN(nn.Module):
    def __init__(self, num_features, num_classes, conv_channels=[128, 256, 128], out_channels=128, conv_kernels=[8, 5, 3], lstm_hidden_states=256, num_layers=1, eps=1.0E-03, momentum=0.99, dropout=0.8, activation=nn.ReLU):
        super().__init__()
        self._conv1_ = nn.Conv1d(in_channels=num_features, out_channels=conv_channels[0], kernel_size=conv_kernels[0], padding='same')
        self._bn1_ = nn.BatchNorm1d(num_features=out_channels, eps=1.0E-03, momentum=0.99)
        self._act1_ = activation()
        self._conv2_ = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=conv_kernels[1], padding='same')
        self._bn2_ = nn.BatchNorm1d(num_features=conv_channels[1], eps=1.0E-03, momentum=0.99)
        self._act2_ = activation()
        self._conv3_ = nn.Conv1d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=conv_kernels[2], padding='same')
        self._bn3_ = nn.BatchNorm1d(num_features=conv_channels[2], eps=1.0E-03, momentum=0.99)
        self._act3_ = activation()
        self._pool_ = nn.AdaptiveAvgPool1d(1)
        self._lstm_ = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_states, num_layers=num_layers, batch_first=True, dropout=0.8 if (num_layers > 1) else 0, proj_size=conv_channels[-1])
        self._drop_ = nn.Dropout(p=dropout)
        self._last_ = nn.AdaptiveAvgPool1d(num_classes)
    def forward(self, x):
        y_conv = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=-1.0)[0]
        y_conv = y_conv.permute((0, 2, 1))
        y_conv = self._conv1_(y_conv)
        y_conv = self._bn1_(y_conv)
        y_conv = self._act1_(y_conv)
        y_conv = self._conv2_(y_conv)
        y_conv = self._bn2_(y_conv)
        y_conv = self._act2_(y_conv)
        y_conv = self._conv3_(y_conv)
        y_conv = self._bn3_(y_conv)
        y_conv = self._act3_(y_conv)
        y_conv = self._pool_(y_conv)
        y_lstm = self._lstm_(x)[0]
        y_lstm = nn.utils.rnn.pad_packed_sequence(y_lstm, batch_first=True, padding_value=-1.0)
        y_lstm = y_lstm[0].gather(
            dim=1,
            index=(y_lstm[1] - 1).view(-1, 1, 1).expand(y_lstm[0].shape[0], 1, y_lstm[0].shape[2]).to(x.data.device)
        )
        ret = torch.concat((y_conv.permute(0, 2, 1), y_lstm), dim=2)
        return self._last_(ret.permute(0, 2, 1).squeeze())

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1, dropout=0.2, proj_size=0):
        super().__init__()
        self._lstm_ = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if (num_layers > 1) else 0,
            proj_size=proj_size
        )
    def forward(self, x):
        y = self._lstm_(x)[0]
        return y

class FCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128, kernel_size=8, eps=1.0E-03, momentum=0.99, activation=nn.ReLU):
        super().__init__()
        self._conv_ = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self._bn_ = nn.BatchNorm1d(num_features=out_channels, eps=eps, momentum=momentum)
        self._act_ = activation()
    def forward(self, x):
        y = self._conv_(x)
        y = self._bn_(y)
        y = self._act_(y)
        return y

class Dataset(torch.utils.data.Dataset):
    def __init__(self, label_path, image_base_path, image_type_key, loss_weight_func=lambda x: np.sqrt(1.0 / x)):
        super().__init__()
        with open(label_path, 'rt') as f:
            self._labels_ = json.load(f)
        self._image_base_ = image_base_path
        self._type_ = image_type_key
    def __len__(self):
        return len(self._labels_['images'])
    def count_samples_per_class(self):
        category_ids = [anno['category_id'] for anno in self._labels_['annotations']]
        class_counts = Counter(category_ids)
        return dict(class_counts)
    def __getitem__(self, key):
        image = self._labels_['images'][key]
        voltage_image = np.load(self._image_base_ / image['files'][self._type_]).astype(np.float32)
        return voltage_image, np.array(int(self._labels_['annotations'][key]['category_id']))

def pad_collate(batch):
    inputs, label = zip(*batch)
    return torch.nn.utils.rnn.pack_sequence([torch.tensor(x) for x in inputs], enforce_sorted=False), torch.tensor(np.array(label)).long()

def get_class_distribution(loader):
    class_count = torch.zeros(3)
    for _, labels in loader:
        for label in labels:
            class_count[label] += 1
    return class_count

def train(model, global_model, trainloader, epochs, loss, lr):
    model.train()
    criterion = loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        total_loss = 0.0
        tqdm_dl = tqdm.auto.tqdm(trainloader, desc=f'Epoch {i + 1} / {epochs}')
        for idx, (x, y) in enumerate(tqdm_dl):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(True)
            outputs = model(x)
            loss_val = criterion(outputs, y)
            total_loss += loss_val.detach()
            loss_val.backward()
            opt.step()
            tqdm_dl.set_postfix({'loss': f'{total_loss / (idx + 1):.3f}'})
    return model

def test(model, testloader, loss):
    model.eval()
    criterion = loss
    tqdm_t_dl = tqdm.auto.tqdm(testloader, desc=f'Test')
    corrects = 0
    total = 0
    loss_val = 0.0
    for x, y in tqdm_t_dl:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        corrects += torch.sum((outputs.argmax(dim=1) == y))
        total += y.shape[0]
        loss_val += criterion(outputs, y).item()
    accuracy = (corrects / total).item()
    test_loss = loss_val / len(testloader)
    return test_loss, accuracy

def average_weights(w, r):
    w_avg = {key: torch.stack([r[i] * w[i][key].float() for i in range(len(w))], 0).sum(0) for key in w[0].keys()}
    return w_avg

def val(model, valloaders, loss):
    model.eval()
    all_preds = []
    all_labels = []
    tqdm_v_dl = tqdm.auto.tqdm(valloaders, desc=f'Vallidation')
    for x, y in tqdm_v_dl:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return f1

def average_loss(o, int_lists):
    orders_set = []
    for order in o:
        orders_set.append(int_lists[order])
    column_sums = [int(sum(column) / len(column)) for column in zip(*orders_set)]
    return column_sums

def updata_model(global_net, clients_net):
    for model in clients_net:
        model.load_state_dict(global_net.state_dict())

def Federated_Client(model, global_model, epochs, p, lr, trainloaders, testloaders, int_lists):
    w_locals = []
    local_los = []
    local_acc = []
    n_locals = []
    order = []
    l_data_size = []
    for i in range(len(trainloaders)):
        loss_sample = int_lists[i]
        u = sum([int(j) for j in (get_class_distribution(trainloaders[i]).tolist())])
        l_beta = (u - 1) / u
        loss_f = Loss(loss_type="cross_entropy", beta=l_beta, samples_per_class=loss_sample, class_balanced=True)
        local = train(model[i], global_model, trainloaders[i], epochs, loss_f, lr)
        los, acc = test(local, testloaders, loss_f)
        local_los.append(los)
        local_acc.append(acc)
        order.append(i)
        local_weight = copy.deepcopy(local.state_dict())
        l_data_dist = sum([int(j) for j in (get_class_distribution(trainloaders[i]).tolist())])
        l_data_size.append(l_data_dist)
        w_locals.append(local_weight)
    pair = list(zip(local_los, w_locals, order, l_data_size))
    pair.sort(reverse=False)
    n_locals_ret = []
    pick_order_ret = []
    pick_data_size_ret = []
    l_local_list_ret = []
    for j in range(p):
        n_locals_ret.append(pair[j][1])
        pick_order_ret.append(pair[j][2])
        pick_data_size_ret.append(pair[j][3])
        l_local_list_ret.append(pair[j][0])
    return n_locals_ret, pick_order_ret, pick_data_size_ret, l_local_list_ret

def Federated_Server(Global_Iteration, epochs, a, b, r, lr, trainloaders, testloaders, valloaders, int_lists):
    learning_rate = lr
    alpha = int(a)
    beta = int(b)
    gamma = r
    now = datetime.datetime.now()
    acc_data = []
    loss_data = []
    f1_data = []
    base_model = MLSTM_FCN(num_classes=3, num_features=cell_count, conv_kernels=[7, 5, 3]).to(device)
    global_model = base_model
    clients_net = [copy.deepcopy(base_model).to(device) for _ in range(10)]
    for model in clients_net:
        model.load_state_dict(global_model.state_dict())
    for i in range(int(Global_Iteration)):
        if i < 20:
            local_weights_0, order, local_datasize, local_loss = Federated_Client(clients_net, global_model, epochs, alpha, learning_rate, trainloaders, testloaders, int_lists)
            global_lists = average_loss(order, int_lists)
            total_data = sum(local_datasize)
            g_beta = (total_data - 1) / total_data
            inverse_local_loss = [1 / x for x in local_loss]
            total_loss = sum(inverse_local_loss)
            ratio_loss = [y / total_loss for y in inverse_local_loss]
            ratio_data = [z / total_data for z in local_datasize]
            ratio = [(gamma * ld) + ((1 - gamma) * dd) for ld, dd in zip(ratio_loss, ratio_data)]
            w_avg0 = average_weights(local_weights_0, ratio)
            global_model.load_state_dict(w_avg0)
            updata_model(global_model, clients_net)
        else:
            local_weights, order, local_datasize, local_loss = Federated_Client(clients_net, global_model, epochs, beta, learning_rate, trainloaders, testloaders, int_lists)
            global_lists = average_loss(order, int_lists)
            total_data = sum(local_datasize)
            g_beta = (total_data - 1) / total_data
            inverse_local_loss = [1 / x for x in local_loss]
            total_loss = sum(inverse_local_loss)
            ratio_loss = [y / total_loss for y in inverse_local_loss]
            ratio_data = [z / total_data for z in local_datasize]
            ratio = [(gamma * ld) + ((1 - gamma) * dd) for ld, dd in zip(ratio_loss, ratio_data)]
            w_avg = average_weights(local_weights, ratio)
            global_model.load_state_dict(w_avg)
            updata_model(global_model, clients_net)
        loss_g = Loss(loss_type="cross_entropy", beta=((total_data - 1) / total_data), samples_per_class=global_lists, class_balanced=True)
        los, accuracy = test(global_model, testloaders, loss_g)
        f1 = val(global_model, valloaders, loss_g)
        acc_data.append(accuracy)
        loss_data.append(los)
        f1_data.append(f1)
    return global_model, acc_data, loss_data, f1_data
