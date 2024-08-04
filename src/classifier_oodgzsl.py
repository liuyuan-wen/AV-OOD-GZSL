import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import src.model_addition as model_addition
import src.utils as utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import copy
import os


class CLASSIFIER:
    # train_Y is integer 
    def __init__(self, args, train_dataset, test_dataset, syn_feature, syn_label, dataset_name, seen_classifier, unseen_classifier, mask=None, zsl_embedding=False, fsl_embedding=False, model_path=None):
        self.train_X = train_dataset.all_data['cat'].to(args.device)
        self.train_Y = train_dataset.all_data['target_mapped'].to(args.device)
        
        self.test_unseen_feature = test_dataset.unseen_data['cat'].to(args.device)
        self.test_unseen_label = test_dataset.unseen_data['target'].to(args.device)
        self.unseenclasses = test_dataset.unseen_data['ids']
        self.test_seen_feature = test_dataset.seen_data['cat'].to(args.device)
        self.test_seen_label = test_dataset.seen_data['target_mapped'].to(args.device)
        self.seenclasses = test_dataset.seen_data['ids']

        self.class_attr = test_dataset.data["text"]["data"].to(args.device)

        self.args = args
        self.batch_size = args.ood.bs
        self.nepoch = args.ood.epochs  # 50
        self.nclass = len(self.seenclasses)
        self.hidden_size1 = args.ood.hidden_size1  # 512
        self.hidden_size2 = args.ood.hidden_size2  # 128
        self.input_dim = self.train_X.size(1)  # 1024
        self.syn_feat = syn_feature
        self.syn_label = syn_label
        self.ood_model = OOD_Detector(self.nclass, self.input_dim)
        self.dataset_name = dataset_name
        self.zsl_embedding = zsl_embedding
        self.fsl_embedding = fsl_embedding
        self.model_path = model_path

        if self.fsl_embedding is True:
            self.seen_cls_model = seen_classifier
        else:
            self.seen_cls_model = seen_classifier.best_model
        if self.zsl_embedding is True:
            self.unseen_cls_model = unseen_classifier
        else:
            self.unseen_cls_model = unseen_classifier.best_model

        self.ood_model.apply(model_addition.weights_init)
        self.criterion = HLoss()
        self.cross_entropy = nn.CrossEntropyLoss()  # CrossEntropyLoss() = NLLLoss(LogSoftmax(), target)
        self.input = torch.FloatTensor(self.batch_size, self.input_dim)   # (512, 1024)
        self.label = torch.LongTensor(self.batch_size).fill_(0)      # (512)
        self.lr = args.ood.lr
        self.beta1 = args.ood.beta
        self.mask = mask
        
        # setup optimizer
        self.od_optimizer = optim.Adam(self.ood_model.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.scheduler = ReduceLROnPlateau(self.od_optimizer, mode='min', factor=0.8, patience=5, threshold=0.0001, verbose=True)

        if args.device == 'cuda':
            self.cuda = True
        if self.cuda:
            self.ood_model.cuda()
            self.criterion = self.criterion.cuda()
            self.cross_entropy = self.cross_entropy.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.index_in_epoch_syn = 0
        self.ntrain = self.train_X.size()[0]  # 55716

        self.best_results, self.final_results = self.fit()
        self.best_acc_seen, self.best_acc_unseen, self.best_H, self.best_seen_ood_acc, self.best_unseen_ood_acc, self.best_fsl_acc, self.best_zsl_acc = self.best_results
        self.final_acc_seen, self.final_acc_unseen, self.final_H, self.final_seen_ood_acc, self.final_unseen_ood_acc, self.final_fsl_acc, self.final_zsl_acc = self.final_results
    
    def fit(self):
        if self.model_path is not None and os.path.exists(self.model_path):
            self.ood_model = torch.load(self.model_path)
            self.ood_model.eval()
            print(f"Loaded {self.model_path}")
            ent_thresh = self.ood_model.get_entropy_thresh()
            print(f"entropy threshold={ent_thresh:.10f}")
            entropy_seen, acc_seen, seen_ood_acc, fsl_acc = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses, ent_thresh, type_classes='seen', save_match=False)
            entropy_unseen, acc_unseen, unseen_ood_acc, zsl_acc = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, ent_thresh, type_classes='unseen', save_match=False)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen+1e-12)
            best_results = [acc_seen, acc_unseen, H, seen_ood_acc, unseen_ood_acc, fsl_acc, zsl_acc]
            final_results = best_results

            if self.args.test_metrics:
                print('___testing fpr_60___')
                if "VGGSound" in self.model_path:
                    ent_thresh = 1.0142463597294549
                elif "UCF" in self.model_path:
                    ent_thresh = 6.361310114302927e-05
                elif "ActivityNet" in self.model_path:
                    ent_thresh = 0.07819021670803451
                entropy_seen, acc_seen, seen_ood_acc, fsl_acc = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses, ent_thresh, type_classes='seen')
                entropy_unseen, acc_unseen, unseen_ood_acc, zsl_acc = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, ent_thresh, type_classes='unseen')
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen+1e-12)
                print(f"TPR={seen_ood_acc:.4f} FPR={1-unseen_ood_acc:.4f}\nS={acc_seen.item():.4f} U={acc_unseen.item():.4f} HM={H.item():.4f}")
                print('___test completed___')

                print("___testing thresholds___")
                print(f"Threshold\tTPR\tFPR\tS\tU\tHM")
                for i in range(10, 50, 1):
                    ent_thresh = i*0.1
                    entropy_seen, acc_seen, seen_ood_acc, fsl_acc = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses, ent_thresh, type_classes='seen')
                    entropy_unseen, acc_unseen, unseen_ood_acc, zsl_acc = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, ent_thresh, type_classes='unseen')
                    H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen+1e-12)
                    print(f"{ent_thresh:.16f}\t{seen_ood_acc:.4f}\t{1-unseen_ood_acc:.4f}\t{acc_seen.item():.4f}\t{acc_unseen.item():.4f}\t{H.item():.4f}")
                print('___test completed___')

        else:
            best_epoch = 0
            best_seen = 0
            best_unseen = 0
            best_seen_ood_acc = 0
            best_unseen_ood_acc = 0
            best_H = 0
            best_fsl_acc = 0
            best_zsl_acc = 0
            epoch_data_list = []
            for epoch in range(self.nepoch):  # 50
                train_loss = 0.0
                entr_seen = 0
                entr_unseen = 0
                hbsz = int(self.batch_size/2) # half batch-size: 64
                batch_num = 0
                # Training of OD dectector
                for i in range(0, self.ntrain, self.batch_size): 
                    batch_num += 1
                    self.ood_model.zero_grad()
                    batch_input, batch_label = self.next_batch(hbsz)
                    batch_input2, batch_label2 = self.next_batch_syn(hbsz)
                    self.input[:hbsz].copy_(batch_input)  # (256, 1024)
                    self.label[:hbsz].copy_(batch_label)  # (256)
                    self.input[hbsz:].copy_(batch_input2)  # (256, 1024)
                    self.label[hbsz:].copy_(batch_label2)  # (256)
                    pred = self.ood_model(self.input)  # (512, 166)

                    ## For seen classes, minimize entropy
                    # CrossEntropyLoss() = NLLLoss(LogSoftmax(), target)
                    loss1 = self.criterion(pred[:hbsz], neg=True) + self.cross_entropy(pred[:hbsz], self.label[:hbsz])
                    ## For unseen classes, maximize entropy
                    loss2 = self.criterion(pred[hbsz:], neg=False)

                    entropy_loss = loss1 + loss2
                    entropy_loss.backward()
                    train_loss += loss2.item()
                    entr_seen += self.criterion(pred[:hbsz], batch=True).sum()  # (64)
                    entr_unseen += self.criterion(pred[hbsz:], batch=True).sum()  # (64)
                    self.od_optimizer.step()
                self.scheduler.step(train_loss)

                # GZSL Evaluation using OD
                ent_thresh_mean = entr_seen.data.item()/self.ntrain  # scalar, average entropy of seen samples
                ent_thresh = ent_thresh_mean
                self.ood_model.def_entropy_thresh(ent_thresh)
                entropy_seen, acc_seen, seen_ood_acc, fsl_acc = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses, ent_thresh, type_classes='seen')
                entropy_unseen, acc_unseen, unseen_ood_acc, zsl_acc = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, ent_thresh, type_classes='unseen')

                epoch_data_list.append((1/(entropy_seen+1e-10), 1/(entropy_unseen+1e-10)))

                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen+1e-12)
                if H > best_H:
                    # print(f'{ent_thresh_mean:.4f}, {ent_thresh:.4f}')
                    best_epoch = epoch
                    best_seen = acc_seen
                    best_unseen = acc_unseen
                    best_seen_ood_acc = seen_ood_acc
                    best_unseen_ood_acc = unseen_ood_acc
                    best_H = H
                    best_fsl_acc = fsl_acc
                    best_zsl_acc = zsl_acc
                    best_model = copy.deepcopy(self.ood_model)
                    if self.args.test_metrics:
                        utils.draw_roc_curve(1, [(1/(entropy_seen+1e-10), 1/(entropy_unseen+1e-10))], None, None, 'roc_ood', f'ROC/{self.dataset_name}')
                    
                if epoch == self.nepoch-1:
                    final_epoch = epoch
                    final_seen = acc_seen
                    final_unseen = acc_unseen
                    final_seen_ood_acc = seen_ood_acc
                    final_unseen_ood_acc = unseen_ood_acc
                    final_H = H
                    final_fsl_acc = fsl_acc
                    final_zsl_acc = zsl_acc
                    final_model = copy.deepcopy(self.ood_model)
            if self.args.test_metrics:
                utils.draw_roc_curve(1, epoch_data_list, None, None, 'roc_ood_all', f'ROC/{self.dataset_name}')
            if self.model_path is not None:
                # torch.save(best_model, self.model_path)
                torch.save(final_model, self.model_path)
            print("BEST EPOCH: ", best_epoch)

            best_results = [best_seen, best_unseen, best_H, best_seen_ood_acc, best_unseen_ood_acc, best_fsl_acc, best_zsl_acc]
            final_results = [final_seen, final_unseen, final_H, final_seen_ood_acc, final_unseen_ood_acc, final_fsl_acc, final_zsl_acc]
            # sys.exit()
        return best_results, final_results

    def val_gzsl(self, test_X, test_label, target_classes, thresh, type_classes, save_match=False): 

        predicted_label = torch.LongTensor(test_label.size()).cuda()
        entropy = []

        if (type_classes == 'unseen' and self.zsl_embedding is True):
            with torch.no_grad():
                audio_emb, video_emb, emb_cls = self.unseen_cls_model.get_embeddings(test_X[:,:512], test_X[:,512:], self.class_attr)
            dis = torch.cdist(video_emb, emb_cls, p=2)
            # dis += torch.cdist(audio_emb, emb_cls, p=2)
            dis[:,self.mask[1]] -= 99
            _, predicted_label = torch.min(dis, 1)
        elif (type_classes == 'seen' and self.fsl_embedding is True):
            with torch.no_grad():
                audio_emb, video_emb, emb_cls = self.seen_cls_model.get_embeddings(test_X[:,:512], test_X[:,512:], self.class_attr)
            dis = torch.cdist(video_emb, emb_cls, p=2)
            dis[:,self.mask[0]] -= 99
            _, predicted_label = torch.min(dis, 1)
        else:
            if type_classes == 'seen':
                pred = self.seen_cls_model(test_X)
                _, predicted_label = torch.max(pred.data, 1)
            elif type_classes == 'unseen':
                pred = self.unseen_cls_model(test_X)
                if self.mask is None:
                    _, predicted_label = torch.max(pred.data, 1)
                else:
                    dis = torch.cdist(pred, self.class_attr, p=2)
                    dis[:,self.mask] += 9999999
                    _, predicted_label = torch.min(dis, 1)

        with torch.no_grad():
            output = self.ood_model(test_X)
        entropy = self.criterion(output, batch=True).cpu().numpy()
        
        if not self.fsl_embedding and type_classes == 'seen':
            target_classes = [x for x in range(len(target_classes))]

        seen_mask = torch.ones(len(entropy))
        fsl_acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, seen_mask)
        
        if type_classes == 'seen':
            seen_mask = torch.Tensor(np.array(entropy)) < thresh
        elif type_classes == 'unseen':
            seen_mask = torch.Tensor(np.array(entropy)) > thresh
        od_acc = (seen_mask.sum()/len(seen_mask)).item()

        if save_match:
            match_idx = predicted_label.eq(test_label.int()).nonzero().flatten()
            unmatch_idx = predicted_label.ne(test_label.int()).nonzero().flatten()
            assert torch.all(torch.logical_not(torch.isin(match_idx, unmatch_idx))) and torch.all(torch.logical_not(torch.isin(unmatch_idx, match_idx)))
            with open(f'TSNE/{self.args.dataset.dataset_name}/match_idx_ood_{type_classes}.pkl', 'wb') as f:
                pickle.dump((match_idx, unmatch_idx), f)
            print(f'Saved match_idx_ood_{type_classes}.pkl')

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, seen_mask)
        return entropy, acc, od_acc, fsl_acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes, mask):
        # # Algorism1:
        # matched = (test_label == predicted_label) * mask.cuda()
        # if len(matched) == 0:
        #     acc_per_class = 0
        # else:
        #     acc_per_class = matched.sum().item() / len(matched)

        # # Algorism2:
        total_acc = 0
        mask = mask.cuda()
        for i in target_classes:  # 0-25
            idx = (test_label == i)  # (630)
            matched = (test_label[idx]==predicted_label[idx]) * mask[idx]
            total_acc += torch.sum(matched) / torch.sum(idx)
        acc_per_class = total_acc / len(target_classes)

        return acc_per_class
    
    # Batch Sampler for seen data              
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        endt = self.index_in_epoch
        if endt > self.ntrain-batch_size:
            # shuffle the data and reset start counter
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            start = 0
            endt = start + batch_size
        return self.train_X[start:endt], self.train_Y[start:endt]

    
    # Fetch next batch for Synthetic features
    def next_batch_syn(self, batch_size):
        start = self.index_in_epoch_syn
        ntrain = self.syn_feat.size(0)  # 30000
        self.index_in_epoch_syn += batch_size
        endt = self.index_in_epoch_syn
        if endt > ntrain-batch_size:
            # shuffle the data and reset start counter
            perm = torch.randperm(ntrain)
            self.syn_feat = self.syn_feat[perm]
            self.syn_label = self.syn_label[perm]
            start = 0
            endt = start + batch_size
        return self.syn_feat[start:endt], self.syn_label[start:endt]


    
class OOD_Detector(nn.Module):
    def __init__(self, num_classes, input_dim=1024, h_size1=512, h_size2=128):  
        super(OOD_Detector, self).__init__()
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.fc1 = nn.Linear(input_dim, h_size1) 
        self.fc2 = nn.Linear(h_size1, h_size2)
        self.classifier = nn.Linear(h_size2, num_classes)
        self.dropout = nn.Dropout(p=0.0)
        self.apply(model_addition.weights_init)
        self.ent_thresh = None
        
    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.dropout(h)
        h = self.lrelu(self.fc2(h))
        h = self.dropout(h)
        pred = self.classifier(h)
        return pred
    
    def def_entropy_thresh(self, ent_thresh):
        self.ent_thresh = ent_thresh

    def get_entropy_thresh(self):
        return self.ent_thresh

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, neg=True, batch=False):  # (512, 166)
        b = self.softmax(x) * self.logsoft(x)  # (512, 166)
        if batch:
            return -1.0 * b.sum(1)  # (512)
        if neg:
            return -1.0 * b.sum()/x.size(0)  # scalar
        else:
            return  b.sum()/x.size(0)  # scalar
