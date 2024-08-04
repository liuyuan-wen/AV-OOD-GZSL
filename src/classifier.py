import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import src.model_addition as model_addition
import copy
import sys
import src.model_addition as model_addition
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, args, train_dataset, test_dataset, type=None, model_path=None):
        self.train_X = train_dataset.all_data['cat'].to(args.device)
        self.train_Y = train_dataset.all_data['target_mapped'].to(args.device)
        
        self.test_unseen_feature = test_dataset.unseen_data['cat'].to(args.device)
        self.test_unseen_label = test_dataset.unseen_data['target_mapped'].to(args.device)
        self.unseenclasses = test_dataset.unseen_data['ids']
        self.test_seen_feature = test_dataset.seen_data['cat'].to(args.device)
        self.test_seen_label = test_dataset.seen_data['target_mapped'].to(args.device)
        self.seenclasses = test_dataset.seen_data['ids']

        self.type = type
        self.input_dim = self.train_X.size(1)
        if self.type == 'seen':
            self.model_args = args.clss
            self.nclass = len(self.seenclasses)
            self.model = CLASS_S(self.nclass, self.input_dim)
        elif self.type == 'unseen':
            self.model_args = args.clsu
            self.nclass = len(self.unseenclasses)
            self.model = CLASS_U(self.nclass, self.input_dim)
        self.batch_size = self.model_args.bs
        self.nepoch = self.model_args.epochs
        
        if args.device == 'cuda':
            self.cuda = True
        
        self.model_path = model_path

        self.model1 = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.criterion = nn.NLLLoss()
        self.criterion1 = nn.CrossEntropyLoss()
        self.input = torch.FloatTensor(self.model_args.bs, self.input_dim) 
        self.label = torch.LongTensor(self.model_args.bs) 

        self.lr = self.model_args.lr
        self.beta = self.model_args.beta
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, verbose=True)
        # self.optimizer1 = optim.Adam(self.model1.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.model1.cuda()
            self.criterion.cuda()
            self.criterion1.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        self.acc, self.best_model, self.final_acc, self.final_model = self.fit() 

    
    def fit(self):
        if self.model_path is not None and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            best_model = copy.deepcopy(self.model)
            final_model = best_model
            print("Model loaded")
            if self.type == 'seen':
                best_acc = self.val(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            elif self.type == 'unseen':
                best_acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            final_acc = best_acc
        else:
            best_acc = 0
            for epoch in range(self.nepoch):
                train_loss = 0.0
                # print("Epoch: ", epoch)
                self.model.train()
                for i in range(0, self.ntrain, self.batch_size):      
                    self.model.zero_grad()
                    batch_input, batch_label = self.next_batch(self.batch_size) 
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)
                    output = self.model(self.input)
                    loss = self.criterion1(output, self.label)
                    loss.backward()
                    train_loss += loss.item()/self.ntrain
                    self.optimizer.step()
                # print(train_loss)
                self.scheduler.step(train_loss)
                self.model.eval()
                if self.type == 'seen':
                    acc = self.val(self.test_seen_feature, self.test_seen_label, self.seenclasses)
                elif self.type == 'unseen':
                    acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    best_model = copy.deepcopy(self.model)
                if epoch == self.nepoch-1:
                    final_epoch = epoch
                    final_acc = acc
                    final_model = copy.deepcopy(self.model)
            if self.model_path is not None:
                # torch.save(best_model.state_dict(), self.model_path)
                torch.save(final_model.state_dict(), self.model_path)
            print(f"BEST EPOCH: {best_epoch}/{self.nepoch}")
        return best_acc, best_model, final_acc, final_model
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            # print("shuffle the data at the first epoch")
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data before beginning the next epoch
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]         
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    # test_label is integer 
    def val(self, test_feature, test_label, target_classes): 
        start = 0
        ntest = test_feature.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        predicted_val = torch.FloatTensor(test_label.size())

        with torch.no_grad():
            output = self.model(test_feature.cuda())  # (630, 51)

        predicted_val, predicted_label = torch.max(output.data, 1)   # (630), (630)

        acc = self.compute_per_class_acc(test_label, predicted_label, len(target_classes))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):

        # matched = (test_label == predicted_label)
        # acc_per_class = matched.sum().item() / len(matched)

        acc_per_class = 0
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= nclass

        return acc_per_class 


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
        
class CLASS_S(nn.Module):
    def __init__(self, nclass, input_dim, hidden_dim1=512, hidden_dim2=256):
        super(CLASS_S, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, nclass)
        self.apply(model_addition.weights_init)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        o = self.lrelu(self.fc1(x))
        o = self.dropout(o)
        o = self.lrelu(self.fc2(o))
        o = self.dropout(o)
        o = self.fc3(o)
        return o  
    
class CLASS_U(nn.Module):
    def __init__(self, nclass, input_dim, hidden_dim1=512, hidden_dim2=256):
        super(CLASS_U, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        hidden_dim1 = int(nclass/2)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, nclass)
        self.apply(model_addition.weights_init)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.0)
    def forward(self, x):
        # o = self.fc(x)
        o = self.lrelu(self.fc1(x))
        # o = self.dropout(o)
        o = self.lrelu(self.fc2(o))
        # o = self.dropout(o)
        # o = self.lrelu(self.fc3(o))
        # o = self.dropout(o)
        o = self.fc3(o)
        return o  