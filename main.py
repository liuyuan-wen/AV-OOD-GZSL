import os
import sys
import yaml
import argparse
import pathlib
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from src.dataset import Dataset, ContrastiveDataset
import src.metrics
from src.sampler import SamplerFactory
import src.model_clsu_embedding
from src.utils import fix_seeds, setup_experiment
from src.train import *
from src.model_addition import generate_syn_feature
import src.model_generator
import src.classifier
import src.classifier_oodgzsl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    return parser.parse_args()

def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, value)
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
    return namespace

def main():
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = dict_to_namespace(config)
    args.dataset.root_set = pathlib.Path(args.dataset.root_set)
    args.dataset.feature_extraction_method = pathlib.Path(args.dataset.feature_extraction_method)

    fix_seeds(args.seed)
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")


    train_val_dataset = Dataset(args=args, dataset_split="train_val")
    test_dataset = Dataset(args=args, dataset_split="test")

    contrastive_train_val_dataset = ContrastiveDataset(train_val_dataset)
    train_val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train_val_dataset.target_to_indices.values()),
        batch_size=args.clsu.bs,
        n_batches=args.clsu.n_batches,
        alpha=1,
        kind='random'
    )
    train_val_loader = data.DataLoader(
        dataset=contrastive_train_val_dataset,
        batch_sampler=train_val_sampler,
        num_workers=args.clsu.num_workers
    )


    #========================clsu embedding pretrain========================
    print('\n--------clsu embedding pretraining--------')
    clsu_embedding = src.model_clsu_embedding.CLSU_Embedding(args)
    clsu_embedding.to(args.device)
    metrics = src.metrics.EmbeddingMetric(args=args,
                            model=clsu_embedding, 
                            val_dataset=None, 
                            test_dataset=test_dataset,)
    logger.info(clsu_embedding)
    model_zsl_embedding_path = f'pretrained_models/{args.dataset.dataset_name}/model_clsu_embedding_{args.dataset.dataset_name}.pth'

    if not args.clsu.scan_pretrain and os.path.exists(model_zsl_embedding_path):
        clsu_embedding.load_state_dict(torch.load(model_zsl_embedding_path))
        print(f'Loaded {model_zsl_embedding_path}')
    else:
        for epoch in range(args.clsu.epochs):
            print()
            train_step(args, train_val_loader, clsu_embedding, epoch=epoch, epochs=args.clsu.epochs, metrics=metrics, stats=train_stats, optimize=True, train_type='clsu')
            test_step(args, train_val_loader, clsu_embedding, epoch=epoch, epochs=args.clsu.epochs, writer=writer, metrics=metrics, stats=val_stats)
        if not args.clsu.scan_pretrain:
            torch.save(clsu_embedding.state_dict(), model_zsl_embedding_path)

    clsu_embedding.eval()
    # os.remove(model_zsl_embedding_path)
    if args.clsu.scan_pretrain:
        sys.exit()
    # sys.exit()

    #========================clss pretraining========================
    print('\n--------clss pretraining--------')
    clss_path = f'pretrained_models/{args.dataset.dataset_name}/model_clss_{args.dataset.dataset_name}.pth'
    clss = src.classifier.CLASSIFIER(args = args,
                                    train_dataset = train_val_dataset,
                                    test_dataset = test_dataset,
                                    type='seen',
                                    model_path=clss_path)
    print('FSL = %.4f/%.4f' % (clss.acc, clss.final_acc))

    # os.remove(clss_path)
    # sys.exit()

    #========================generator pretraining========================
    print('\n--------generator training--------')
    model_generator = src.model_generator.Generator(args)
    model_generator.to(args.device)
    model_generator_path = f'pretrained_models/{args.dataset.dataset_name}/model_gen_{args.dataset.dataset_name}.pth'
    if os.path.exists(model_generator_path):
        model_generator.load_state_dict(torch.load(model_generator_path))
        print(f'Loaded {model_generator_path}')
    else:
        for epoch in range(args.gen.epochs):
            print(epoch)
            train_step(args, train_val_loader, model_generator, epoch=epoch, epochs=args.gen.epochs, metrics=None, stats=train_stats, optimize=True, train_type='gen')
            if epoch == args.gen.epochs-1:
                torch.save(model_generator.state_dict(), model_generator_path)
    model_generator.eval()

    # os.remove(model_generator_path)
    # sys.exit()


    #========================ood pretraining, gzsl classifying========================
    print('\n--------ood pretraining, gzsl classifying--------')
    ood_path = f'pretrained_models/{args.dataset.dataset_name}/model_ood-ent_{args.dataset.dataset_name}.pth'

    syn_feature_unseen_test, syn_label_unseen_test = generate_syn_feature(model_generator.netG, test_dataset.unseen_class_ids, test_dataset.all_targets, args.ood.syn_num)    # (syn_num*55, 512), (syn_num*55)/276
    clsg = src.classifier_oodgzsl.CLASSIFIER(args = args,
                                            train_dataset = train_val_dataset,     # (55716, 512)
                                            test_dataset = test_dataset,     # (55716)
                                            syn_feature = syn_feature_unseen_test,     # (syn_num*55, 512)
                                            syn_label = syn_label_unseen_test,         # (syn_num*55)
                                            dataset_name=args.dataset.dataset_name,
                                            seen_classifier = clss, 
                                            unseen_classifier = clsu_embedding, 
                                            mask = (test_dataset.seen_class_ids, test_dataset.unseen_class_ids),
                                            zsl_embedding = True,
                                            fsl_embedding = False,
                                            model_path = ood_path)

    print('TPR(Recall):  {:.4f}/{:.4f}'.format(clsg.best_seen_ood_acc, clsg.final_seen_ood_acc), 'FPR:  {:.4f}/{:.4f}'.format(1-clsg.best_unseen_ood_acc, 1-clsg.final_unseen_ood_acc))
    print('Seen Classifier Acc: {:.4f}'.format(clsg.best_fsl_acc),     'Unseen Classifier Acc(ZSL): {:.4f}'.format(clsg.best_zsl_acc))
    # print('GZSL-OD_best:  S = %.4f             U = %.4f             HM = %.4f' % (clsg.best_acc_seen, clsg.best_acc_unseen, clsg.best_H))
    print('GZSL-OD_final: S = %.4f             U = %.4f             HM = %.4f' % (clsg.final_acc_seen, clsg.final_acc_unseen, clsg.final_H))


    logger.info(f"FINISHED. Unseen Classifier Run is stored at {log_dir}")


if __name__ == '__main__':
    main()
