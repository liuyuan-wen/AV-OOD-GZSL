import torch.nn as nn
import torch
import os
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import sys

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        # return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        # self.apply(weights_init)
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        super(EmbeddingNet, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)
        self.apply(weights_init)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EarlyStopping:
    def __init__(self, start_patience=5, patience=8, verbose=True):
        self.start_patience = start_patience
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epoch_hms = []
        self.hm_fluctuates = []
        self.last_epoch_hm = None

    def __call__(self, val_hm):
        if self.last_epoch_hm is None:
            self.last_epoch_hm = val_hm
            # print("last_epoch_hm initialized")
            return
        self.epoch_hms.append(val_hm)
        self.hm_fluctuates.append(abs(self.last_epoch_hm - val_hm))
        self.mean_hm_fluctuate = sum(self.hm_fluctuates) / len(self.hm_fluctuates)
        # print(f"hm_fluctuates: {self.hm_fluctuates}")
        # print(f"mean_hm_fluctuate: {self.mean_hm_fluctuate}")
        if len(self.hm_fluctuates) >= self.start_patience and abs(val_hm - self.last_epoch_hm) < self.mean_hm_fluctuate:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            # print("counter return to 0")
        self.last_epoch_hm = val_hm

    def should_stop(self):
        return self.early_stop
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_CRITIC(nn.Module):
    def __init__(self, resSize, attSize):   # 1024, 300
        super(MLP_CRITIC, self).__init__()
        ndh = int((resSize + attSize)/2)    # size of the hidden units in discriminator
        self.fc1 = nn.Linear(resSize + attSize, ndh)
        self.fc2 = nn.Linear(ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_G(nn.Module):
    def __init__(self, resSize, attSize, nz):   # 1024, 300, 1024
        super(MLP_G, self).__init__()
        ngh = int((resSize)/2)
        ngh1 = ngh
        self.fc1 = nn.Linear(attSize + nz, ngh) 
        self.fc2 = nn.Linear(ngh, ngh1)
        self.fc3 = nn.Linear(ngh1, resSize)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.output_activation = nn.Tanh()
        self.dropout = nn.Dropout(0.0)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.relu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        
        return h

      
class Dec(nn.Module):
    def __init__(self, resSize, attSize):   # 1024, 300
        super(Dec, self).__init__()
        ngh = int((resSize + attSize)/2)
        # ngh = 512
        self.fc1 = nn.Linear(resSize, ngh)  # (1024, 512)
        self.fc2 = nn.Linear(ngh, ngh)  # (512, 512)
        self.fc3 = nn.Linear(ngh, attSize)  # (512, 300)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, feat):      
        h = self.lrelu(self.fc1(feat))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h
    

# 定义变分自编码器（VAE）类
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)  # Split into mean and logvar
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

class CosineSimilarityLoss(nn.Module):
    def __init__(self, target_similarity):
        super(CosineSimilarityLoss, self).__init__()
        self.target_similarity = target_similarity

    def forward(self, x, y):
        cos_sim = 0
        for i in range(x.shape[0]):
            # cos_sim = torch.add(cos_sim, cal_cos_sim(fake[i].unsqueeze(0), input_res[i].unsqueeze(0)))
            cos_sim += F.cosine_similarity(x[i].unsqueeze(0), y[i].unsqueeze(0)).mean()
        # cos_sim /= x.shape[0]
        # loss = torch.mean((cos_sim - self.target_similarity*x.shape[0]) ** 2)
        loss = abs(cos_sim - self.target_similarity*x.shape[0])
        return loss

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def map_label(labels, classes_ids):
    mapped_label = torch.LongTensor(labels.size())
    for i in range(len(classes_ids)):
        mapped_label[labels==classes_ids[i]] = i    
    corres_ids = [classes_ids[i] for i in range(len(classes_ids))]
    return mapped_label, corres_ids

def generate_syn_feature(netG, classes, attribute, num):
    # generate num synthetic samples for each class in classes
    # nclass = classes.size(0)    # 55
    nclass = len(classes)
    resSize=512*2
    attSize=300
    nz=512*2
    syn_feature = torch.FloatTensor(nclass*num, resSize).cuda()    # (25*1200=30000, 8192)
    syn_label = torch.LongTensor(nclass*num).cuda()      # (25*1200=30000)
    syn_att = torch.FloatTensor(num, attSize).cuda()    # (1200, 300)
    syn_noise = torch.FloatTensor(num, nz).cuda()   # (1200, 300)
    
    for i in range(nclass):
        iclass = classes[i]
        # iclass = i
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)    # (1200, 8192)
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # Gradient penalty of WGAN
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1).cuda()   # (256, 1)
    alpha = alpha.expand(real_data.size())  # (256, 512)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)    # (128, 8192)
    interpolates = interpolates.cuda()

    interpolates = interpolates.requires_grad_()

    disc_interpolates = netD(interpolates, input_att)   # (128, 1)

    ones = torch.ones(disc_interpolates.size()).cuda()  # (128, 1)

    _gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True)   # ((128, 8192),)
    gradients = _gradients[0]     # (128, 8192)

    lambda1 = 10    # gradient penalty regularizer, following WGAN-GP
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty