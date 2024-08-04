import numpy as np
import torch
import sys
from src.utils import evaluate_dataset_baseline


class EmbeddingMetric:
    def __init__(self, args, model, val_dataset, test_dataset):
        super(EmbeddingMetric, self).__init__()
        self.args = args
        self.model = model
        self.dataset = val_dataset
        self.test_dataset = test_dataset
        self.reset()

    def __call__(self):

        test_evaluation = evaluate_dataset_baseline(args=self.args, 
                                                    dataset=self.test_dataset, 
                                                    model=self.model, 
                                                    best_beta=None)

        self.test_seen.append(test_evaluation["seen"])
        self.test_unseen.append(test_evaluation["unseen"])
        self.test_hm.append(test_evaluation["hm"])
        self.test_recall.append(test_evaluation["recall"])
        self.test_beta.append(test_evaluation["beta"])
        self.test_zsl.append(test_evaluation["zsl"])
        self.test_fsl.append(test_evaluation["fsl"])

    def reset(self):

        self.test_seen = []
        self.test_unseen = []
        self.test_hm = []
        self.test_recall = []
        self.test_beta = []
        self.test_zsl=[]
        self.test_fsl=[]

    def value(self):
        return {
            "test_seen": np.mean(self.test_seen),
            "test_unseen": np.mean(self.test_unseen),
            "test_hm": np.mean(self.test_hm),
            "test_recall": np.mean(self.test_recall, axis=0),
            "test_beta": np.mean(self.test_beta),
            "test_zsl":np.mean(self.test_zsl),
            "test_fsl":np.mean(self.test_fsl),
        }
