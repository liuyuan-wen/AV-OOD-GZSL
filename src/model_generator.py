#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np
import src.model_addition as model
from src.utils import cal_cos_sim
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Inspired from https://github.com/AnjanDutta/sem-pcyc
class Generator(nn.Module):
    def __init__(self, args):   # k_input = 512
        super(Generator, self).__init__()
        resSize = args.input_size_audio + args.input_size_video
        self.nz = resSize
        self.netD = model.MLP_CRITIC(resSize=resSize, attSize=300)
        self.netG = model.MLP_G(resSize=resSize, attSize=300, nz=self.nz)
        self.netDec = model.Dec(resSize=resSize, attSize=300)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizerDec = optim.Adam(self.netDec.parameters(), lr=0.0001, betas=(0.5, 0.999))


    def optimize_params(self, audio, video, cls_numeric, cls_embedding, audio_negative, video_negative, negative_cls_embedding):
        
        batch_size = audio.shape[0]
        resSize = 2*audio.shape[1]
        attSize = cls_embedding.shape[1]
        input_res = torch.FloatTensor(batch_size, resSize).cuda()  # (256, 300)
        input_att = torch.FloatTensor(batch_size, attSize).cuda()  # (256, 300)
        input_label = torch.LongTensor(batch_size).cuda()  # (256)
        noise = torch.FloatTensor(batch_size, self.nz).cuda()   # (256, 300)
        one = torch.FloatTensor([1]).cuda()
        one = torch.tensor(one.item()).cuda()
        mone = one * -1
        mone = torch.tensor(mone.item()).cuda()
        emb_criterion = nn.CosineEmbeddingLoss(margin=0).cuda()
        recons_criterion = nn.MSELoss().cuda()
        self.netD.train()
        self.netG.train()
        self.netDec.train()
        ############################
        # (1) Update D network: optimize WGAN-GP objective
        ############################
        for p in self.netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        critic_iter=5
        for iter_d in range(critic_iter):   # 5
            # Sample a batch
            feature = torch.cat((audio.detach(), video.detach()), dim=1)
            batch_feature, batch_label, batch_att = feature, cls_numeric, cls_embedding
            input_res.copy_(batch_feature)
            input_att.copy_(batch_att)
            input_label.copy_(batch_label)

            # Decoder training
            self.netDec.zero_grad()
            recons = self.netDec(input_res)
            R_cost = recons_criterion(recons, input_att)
            R_cost.backward()
            self.optimizerDec.step()
            
            # Discriminator training with real
            self.netD.zero_grad()
            criticD_real = self.netD(input_res, input_att)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)  # maximize E[criticD(real)]

            # train Discriminator with fakeG
            noise.normal_(0, 1)
            fake = self.netG(noise, input_att)     # (256, 512)
            criticD_fake = self.netD(fake.detach(), input_att)    # (256, 1)
            criticD_fake = criticD_fake.mean()    # (scalar)
            criticD_fake.backward(one)  # minimize E[criticD(fake)]

            # WGAN gradient penalty
            gradient_penalty = model.calc_gradient_penalty(self.netD, input_res, fake.data, input_att)
            gradient_penalty.backward() # minimize gradient penalty

            Wasserstein_dist_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            self.optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective
        ############################
        for p in self.netD.parameters(): # reset requires_grad
            p.requires_grad = False
        for p in self.netG.parameters(): # reset requires_grad
            p.requires_grad = True

        self.netG.zero_grad()
        noise.normal_(0, 1)
        fake = self.netG(noise, input_att)
        criticG_fake = self.netD(fake, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        criticG_real = self.netD(input_res, input_att)
        criticG_real = criticG_real.mean()
        Wasserstein_dist_G = criticG_real - criticG_fake
        errG = G_cost

        ## cosine embedding loss for matching pairs
        temp_label = torch.ones(fake.shape[0]).cuda()   # (256)
        # fake and input_resv are matched already
        embed_match = emb_criterion(fake, input_res, temp_label)     # (scalar)

        ### cosine embedding loss for non-matching pairs
        # Randomly permute the labels and real input data
        rand_index = torch.randperm(input_label.shape[0]).cuda()
        new_label = input_label[rand_index]
        new_feat = input_res[rand_index, :]
        z1 = input_label.cpu().numpy()
        z2 = new_label.cpu().numpy()
        temp_label = -1 * torch.ones(fake.shape[0])

        # Label correction for pairs that remain matched after random permutation
        if len(np.where(z1==z2)[0])>0:
            temp_label[torch.from_numpy(np.where(z1==z2)[0])] = 1
        temp_label = temp_label.cuda()
        embed_nonmatch = emb_criterion(fake, new_feat, temp_label)
        embed_err = embed_match + embed_nonmatch
        cosem_weight = 0.1   # recons_weight for decoder, 0.1
        errG += cosem_weight*embed_err    
        # embed_err = torch.tensor([0])

        # Attribute reconstruction loss
        self.netDec.zero_grad()
        recons = self.netDec(fake)    
        R_cost = recons_criterion(recons, input_att)
        recons_weight=0.01
        errG += recons_weight*R_cost    # 0.01

        errG.backward()
        self.optimizerG.step()
        self.optimizerDec.step()

        for p in self.netG.parameters(): # reset requires_grad
            p.requires_grad = False
        self.netD.eval()
        self.netG.eval()
        self.netDec.eval()

        costs = (D_cost.data.item(), G_cost.data.item(), embed_err.data.item(), R_cost.data.item(), Wasserstein_dist_D.data.item(), Wasserstein_dist_G.data.item())
        return costs
