""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class IBLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, in_dim)  # 用于重参数化
        self.fc_logvar = nn.Linear(hidden_dim, in_dim)  # 用于计算方差
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        hidden = F.relu(self.fc(x))
        hidden = self.dropout(hidden)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class DMTIB(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList([IBLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])

        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim)-1):
            self.MMClasifier.append(LinearLayer(self.views*hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views*hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence, mu, logvar = dict(), dict(), dict(), dict(), dict(), dict()
        for view in range(self.views):
            feature[view], mu[view], logvar[view] = self.FeatureInforEncoder[view](data_list[view])
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])

            feature[view] = feature[view] * TCPConfidence[view]

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        for view in range(self.views):
            kl = -0.5 * torch.mean(1 + logvar[view] - mu[view].pow(2) - logvar[view].exp())
            MMLoss = MMLoss + 0.1 * torch.mean(kl)
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target)+criterion(TCPLogit[view], label))

            MMLoss = MMLoss+confidence_loss
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit





