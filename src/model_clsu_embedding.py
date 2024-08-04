import os
import numpy as np
import src.model_addition as model
import torch
import torch.nn as nn
import torch.optim as optim

# Inspired from https://github.com/AnjanDutta/sem-pcyc
class CLSU_Embedding(nn.Module):
    def __init__(self, args):
        super(CLSU_Embedding, self).__init__()
        self.args = args
        self.input_size_audio = args.input_size_audio
        self.input_size_video = args.input_size_video
        self.dim_out = args.clsu.dim_out
        self.hidden_size_encoder=args.clsu.hidden_size_encoder
        self.hidden_size_decoder=args.clsu.hidden_size_decoder
        self.r_enc=args.clsu.r_enc
        self.r_proj=args.clsu.r_proj
        self.r_dec=args.clsu.r_dec
        self.depth_transformer=args.clsu.depth_transformer    # default=1
        self.momentum=args.clsu.momentum
        self.earlystop=args.clsu.earlystop
        self.lr = args.clsu.lr

        self.A_enc = model.EmbeddingNet(
            input_size=self.input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )
        self.V_enc = model.EmbeddingNet(
            input_size=self.input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

        self.cross_attention=model.Transformer(dim=300, depth=self.depth_transformer, heads=3, dim_head=100, mlp_dim=64, dropout=self.r_enc)   # k_attnhidd = 64

        self.W_proj= model.EmbeddingNet(
            input_size=300,
            hidden_size=self.hidden_size_decoder, 
            output_size=self.dim_out,
            dropout=self.r_proj,
            momentum=self.momentum,
            use_bn=True
        )

        self.D = model.EmbeddingNet(
            input_size=self.dim_out,
            output_size=300,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )

        self.A_proj = model.EmbeddingNet(input_size=300, 
                                   hidden_size=self.hidden_size_decoder, 
                                   output_size=self.dim_out, 
                                   dropout=self.r_proj, 
                                   momentum=self.momentum,
                                   use_bn=True)

        self.V_proj = model.EmbeddingNet(input_size=300, 
                                   hidden_size=self.hidden_size_decoder, 
                                   output_size=self.dim_out, 
                                   dropout=self.r_proj, 
                                   momentum=self.momentum,
                                   use_bn=True)

        self.A_rec = model.EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.V_rec = model.EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 300))

        self.optimizer_gen = optim.Adam(
                                        list(self.A_proj.parameters()) + 
                                        list(self.V_proj.parameters()) +
                                        list(self.A_rec.parameters()) + 
                                        list(self.V_rec.parameters()) +
                                        list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
                                        list(self.cross_attention.parameters()) + 
                                        list(self.D.parameters()) +
                                        list(self.W_proj.parameters()) , 
                                        lr=self.lr, weight_decay=1e-5)

        self.scheduler_gen =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        self.early_stopping = model.EarlyStopping(start_patience=self.earlystop, patience=8)

        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1)

    def change_param(self, lr, weight_decay=None, r_enc=None, r_proj=None, r_dec=None):
        self.lr = lr
        if weight_decay is None:
            weight_decay = 1e-5
        if r_enc is not None:
            self.r_enc=r_enc
        if r_proj is not None:
            self.r_proj=r_proj
        if r_dec is not None:
            self.r_dec=r_dec
        self.optimizer_gen = optim.Adam(
                                        list(self.A_proj.parameters()) + 
                                        list(self.V_proj.parameters()) +
                                        list(self.A_rec.parameters()) + 
                                        list(self.V_rec.parameters()) +
                                        list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
                                        list(self.cross_attention.parameters()) + 
                                        list(self.D.parameters()) +
                                        list(self.W_proj.parameters()) , 
                                        lr=self.lr, weight_decay=weight_decay)
    
    def print_lr(self):
        print('Current lr: ', self.lr)

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):

        self.phi_a = self.A_enc(audio)
        self.phi_v = self.V_enc(image)

        self.phi_a_neg=self.A_enc(negative_audio)
        self.phi_v_neg=self.V_enc(negative_image)

        self.w=word_embedding
        self.w_neg=negative_word_embedding

        self.theta_w = self.W_proj(word_embedding)
        self.theta_w_neg=self.W_proj(negative_word_embedding)

        self.rho_w=self.D(self.theta_w)
        self.rho_w_neg=self.D(self.theta_w_neg)


        #========================================================================
        # Cross-attention block
        self.positive_input=torch.stack((self.phi_a + self.pos_emb1D[0, :], self.phi_v + self.pos_emb1D[1, :]), dim=1)
        self.negative_input=torch.stack((self.phi_a_neg + self.pos_emb1D[0, :], self.phi_v_neg + self.pos_emb1D[1, :]), dim=1)


        self.phi_attn= self.cross_attention(self.positive_input)
        self.phi_attn_neg = self.cross_attention(self.negative_input)

        self.audio_fe_attn = self.phi_a + self.phi_attn[:, 0, :]
        self.video_fe_attn= self.phi_v + self.phi_attn[:, 1, :]

        self.audio_fe_neg_attn = self.phi_a_neg + self.phi_attn_neg[:, 0, :]
        self.video_fe_neg_attn = self.phi_v_neg + self.phi_attn_neg[:, 1, :]
        #========================================================================

        self.theta_v = self.V_proj(self.video_fe_attn)
        self.theta_v_neg=self.V_proj(self.video_fe_neg_attn)

        self.theta_a = self.A_proj(self.audio_fe_attn)
        self.theta_a_neg=self.A_proj(self.audio_fe_neg_attn)

        self.phi_v_rec = self.V_rec(self.theta_v)
        self.phi_a_rec = self.A_rec(self.theta_a)

        self.phi_v_neg_rec = self.V_rec(self.theta_v_neg)
        self.phi_a_neg_rec = self.A_rec(self.theta_a_neg)

        self.rho_a=self.D(self.theta_a)
        self.rho_a_neg=self.D(self.theta_a_neg)

        self.rho_v=self.D(self.theta_v)
        self.rho_v_neg=self.D(self.theta_v_neg)


    def backward(self, optimize):
                                             # l_t, l_reg, l_rec
        positive_loss = self.loss_function_w_a(True, True, True, self.w, self.phi_a, self.phi_v, self.theta_a, self.theta_v, self.theta_w, \
                                           self.theta_a_neg, self.theta_v_neg, self.theta_w_neg, self.phi_a_rec, self.phi_v_rec, \
                                           self.rho_a, self.rho_v, self.rho_w, self.rho_a_neg, self.rho_v_neg)
        
        negative_loss = self.loss_function_w_a(self.args.clsu.ltrip_neg, self.args.clsu.lreg_neg, self.args.clsu.lrec_neg, self.w_neg, self.phi_a_neg, self.phi_v_neg, self.theta_a_neg, self.theta_v_neg, self.theta_w_neg, \
                                        self.theta_a, self.theta_v, self.theta_w, self.phi_a_neg_rec, self.phi_v_neg_rec, \
                                        self.rho_a_neg, self.rho_v_neg, self.rho_w_neg, self.rho_a, self.rho_v)

        loss_gen = positive_loss + negative_loss

        if optimize == True:
            self.optimizer_gen.zero_grad()
            loss_gen.backward()
            self.optimizer_gen.step()

        loss_numeric = loss_gen

        return loss_numeric

    def optimize_params(self, audio, video, cls_numeric, cls_embedding, audio_negative, video_negative, negative_cls_embedding, optimize=False):

        self.forward(audio, video, audio_negative, video_negative, cls_embedding, negative_cls_embedding)
        loss_numeric = self.backward(optimize)

        return loss_numeric

    def get_embeddings(self, audio, video, embedding, get_fusion=False):

        phi_a = self.A_enc(audio)
        phi_v = self.V_enc(video)
        theta_w=self.W_proj(embedding)

        input_concatenated=torch.stack((phi_a+self.pos_emb1D[0,:], phi_v+self.pos_emb1D[1,:]), dim=1)

        phi_attn= self.cross_attention(input_concatenated)

        if get_fusion is True:
            return phi_attn

        phi_a = phi_a + phi_attn[:,0,:]
        phi_v = phi_v + phi_attn[:,1,:]

        theta_v = self.V_proj(phi_v)
        theta_a = self.A_proj(phi_a)

        return theta_a, theta_v, theta_w

    def loss_function_w_a(self, lt, lreg, lrec, w, phi_a, phi_v, theta_a, theta_v, theta_w, theta_a_neg, theta_v_neg, theta_w_neg, phi_a_rec, phi_v_rec, rho_a, rho_v, rho_w, rho_a_neg, rho_v_neg):
        #-----------------------Base triplet loss-----------------------
        first_pair = self.triplet_loss(theta_a, theta_w, theta_a_neg) + \
                        self.triplet_loss(theta_v, theta_w, theta_v_neg)
        second_pair = self.triplet_loss(theta_w, theta_a, theta_w_neg) + \
                        self.triplet_loss(theta_w, theta_v, theta_w_neg)
        l_bt=first_pair+second_pair
        
        #-----------------------Triplet l_ct loss-----------------------
        l_ctv=self.triplet_loss(rho_w, rho_v, rho_v_neg)
        l_cta=self.triplet_loss(rho_w, rho_a, rho_a_neg)
        l_ct=l_cta+l_ctv

        #-----------------------Triplet l_wt loss-----------------------
        l_tv = self.triplet_loss(theta_w, theta_v, theta_v_neg)
        l_ta = self.triplet_loss(theta_w, theta_a, theta_a_neg)
        l_at = self.triplet_loss(theta_a, theta_w, theta_w_neg)
        l_vt = self.triplet_loss(theta_v, theta_w, theta_w_neg)
        l_wt=l_ta+l_at+l_tv+l_vt

        if lt is True:
            l_t = l_bt + l_ct + l_wt
        else:
            l_t = 0

        #-----------------------Regularization loss-----------------------
        if lreg is True:
            l_reg = (self.criterion_reg(phi_v_rec, phi_v) + \
                            self.criterion_reg(phi_a_rec, phi_a))
        else:
            l_reg = 0

        #-----------------------Reconstruction loss-----------------------
        if lrec is True:
            l_rec = self.criterion_reg(w, rho_v) + \
                    self.criterion_reg(w, rho_a) + \
                    self.criterion_reg(w, rho_w)
        else:
            l_rec = 0

        total_loss = l_t + l_reg + l_rec
        # print("loss_t: ", l_t.item(), "loss_reg: ", l_reg.item(), "loss_rec: ", l_rec.item())

        return total_loss
    
    def loss_function_wo_a(self, lt, lreg, lrec, w, phi_v, theta_v, theta_w, theta_v_neg, theta_w_neg, phi_v_rec, rho_v, rho_w, rho_v_neg):
        #-----------------------Base triplet loss-----------------------
        first_pair = self.triplet_loss(theta_v, theta_w, theta_v_neg)
        second_pair = self.triplet_loss(theta_w, theta_v, theta_w_neg)
        l_bt=first_pair+second_pair

        #-----------------------Regularization loss-----------------------
        if lreg is True:
            l_reg = (self.criterion_reg(phi_v_rec, phi_v) + \
                            self.criterion_reg(theta_v, theta_w))
        else:
            l_reg = 0

        #-----------------------Reconstruction loss-----------------------
        if lrec is True:
            l_rec = self.criterion_reg(w, rho_v) + \
                    self.criterion_reg(w, rho_w)
        else:
            l_rec = 0

        #-----------------------Triplet l_ct loss-----------------------
        l_ctv=self.triplet_loss(rho_w, rho_v, rho_v_neg)
        l_ct=l_ctv

        #-----------------------Triplet l_wt loss-----------------------
        l_tv = self.triplet_loss(theta_w, theta_v, theta_v_neg)
        l_vt = self.triplet_loss(theta_v, theta_w, theta_w_neg)
        l_wt=l_tv+l_vt

        if lt is True:
            l_t = l_bt + l_ct + l_wt
        else:
            l_t = 0

        total_loss = l_t + l_reg + l_rec
        # print("loss_t: ", l_t.item(), "loss_reg: ", l_reg.item(), "loss_rec: ", l_rec.item())

        return total_loss