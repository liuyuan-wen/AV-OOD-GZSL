import pathlib
import logging
import pickle
import socket
from datetime import datetime
from pathlib import Path
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.logger import PD_Stats, create_logger


def read_features(path):
    hf = h5py.File(path, 'r')
    # keys = list(hf.keys())
    data = hf['data']
    url = [str(u, 'utf-8') for u in list(hf['video_urls'])]

    return data, url


def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_experiment(args, *stats, exist_ok=False):
    if args.exp_name == "":
        exp_name = str(args.run_file_path) + f"/{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}"
    else:
        exp_name = str(args.run_file_path) + "/" + str(args.exp_name)
    log_dir = (pathlib.Path(".") / exp_name)
    log_dir.mkdir(parents=True, exist_ok=exist_ok)
    (log_dir / "checkpoints").mkdir(exist_ok=exist_ok)
    pickle.dump(args, (log_dir / "args.pkl").open("wb"))
    train_stats = PD_Stats(log_dir / "train_stats.pkl", stats)
    val_stats = PD_Stats(log_dir / "val_stats.pkl", stats)
    logger = create_logger(log_dir / "train.log")

    logger.info(f"Start experiment {exp_name}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"The experiment will be stored in {log_dir.resolve()}\n")
    logger.info("")
    if args.exp_name == "":
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(log_dir=exp_name)
    return logger, log_dir, writer, train_stats, val_stats


def setup_evaluation(args, *stats):
    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()
    # pickle.dump(args, (eval_dir / "args.pkl").open("wb"))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    logger = create_logger(eval_dir / "eval.log")

    logger.info(f"Start evaluation {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    return logger, eval_dir, test_stats


def save_best_model(epoch, best_metric, model, optimizer, log_dir, metric="", checkpoint=False):
    logger = logging.getLogger()
    logger.info(f"Saving model to {log_dir} with {metric} = {best_metric:.4f}")
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        # "optimizer": optimizer.state_dict(),
        "metric": metric
    }
    if checkpoint:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}_ckpt_{epoch}.pt"
        )
    else:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}.pt"
        )


def check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir):
    if not best_loss:
        save_best_model(epoch, val_loss, model, optimizer, log_dir, metric="loss")
        return val_loss
    if val_loss < best_loss:
        best_loss = val_loss
        save_best_model(epoch, best_loss, model, optimizer, log_dir, metric="loss")
    return best_loss


def check_best_score(epoch, best_score, hm_score, model, optimizer, log_dir):
    if not best_score:
        save_best_model(epoch, hm_score, model, optimizer, log_dir, metric="score")
        return hm_score
    if hm_score > best_score:
        best_score = hm_score
        save_best_model(epoch, best_score, model, optimizer, log_dir, metric="score")
    return best_score


def load_model_parameters(model, model_weights):
    logger = logging.getLogger()
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            logger.info("didnt load ", name)


def load_args(path):
    return pickle.load((path / "args.pkl").open("rb"))


def cos_dist(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res


def evaluate_dataset_baseline(args, dataset, model, best_beta=None):
    data_a = dataset.all_data["audio"].to(args.device)
    data_v = dataset.all_data["video"].to(args.device)
    data_t = dataset.all_data["text"].to(args.device)
    data_num = dataset.all_data["target"].to(args.device)

    all_targets = dataset.all_data["target"].to(args.device)
    model.eval()

    a_p, v_p, t_p = model.get_embeddings(data_a, data_v, data_t)
    evaluation = get_best_evaluation(args, dataset, all_targets, a_p, v_p, t_p, best_beta=best_beta)
    return evaluation



def get_best_evaluation(args, dataset, targets, a_p, v_p, t_p, best_beta=None):
    betas_dis_list = []
    seen_scores = []
    zsl_scores = []
    fsl_scores = []
    unseen_scores = []
    hm_scores = []
    per_class_recalls = []
    tpr_fpr = []
    start = 0
    end = 5
    steps = int((end - start) / 0.2) + 1
    # steps = 16
    betas = torch.tensor([best_beta], dtype=torch.float, device=args.device) if best_beta else torch.linspace(start, end, steps,
                                                                                                         device=args.device)
    seen_label_array = torch.tensor(dataset.seen_class_ids, dtype=torch.long, device=args.device)
    unseen_label_array = torch.tensor(dataset.unseen_class_ids, dtype=torch.long, device=args.device)
    seen_unseen_array = torch.tensor(np.sort(np.concatenate((dataset.seen_class_ids, dataset.unseen_class_ids))),
                                     dtype=torch.long, device=args.device)
    # print(a_p.shape)
    # print(v_p.shape)
    # print(t_p.shape)
    # print(seen_label_array.shape)
    # print(unseen_label_array.shape)
    # print(unseen_label_array)
    # print(seen_unseen_array.shape)
    # print()
    # sys.exit()
    with torch.no_grad():
        seen_class_ids = dataset.seen_class_ids.copy()
        unseen_class_ids = dataset.unseen_class_ids.copy()

        seen_targets = torch.isin(targets, torch.tensor(seen_class_ids).cuda())
        unseen_targets = torch.isin(targets, torch.tensor(unseen_class_ids).cuda())
        seen_num = seen_targets.sum().item()
        unseen_num = unseen_targets.sum().item()
        seen_indices = torch.nonzero(seen_targets).squeeze()
        unseen_indices = torch.nonzero(unseen_targets).squeeze()
        assert seen_num + unseen_num == targets.shape[0]
        targets_bin = np.array([1 if i.item() in seen_class_ids else 0 for i in targets])

        distance_mat = torch.zeros((a_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                    device=args.device) + 99999999999999
        distance_mat_zsl = torch.zeros((a_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                        device=args.device) + 99999999999999

        if args.clsu.evaluation_mode == "audio":
            distance_mat[:, seen_unseen_array] = torch.cdist(a_p, t_p)  # .pow(2)
            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=args.device)
            mask[seen_label_array] = 99999999999999
            distance_mat_zsl = distance_mat + mask
            if args.clsu.distance_fn == "SquaredL2Loss":
                distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
                distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
        elif args.clsu.evaluation_mode == "video":
            distance_mat[:, seen_unseen_array] = torch.cdist(v_p, t_p)  # .pow(2)
            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=args.device)
            mask[seen_label_array] = 99999999999999
            distance_mat_zsl = distance_mat + mask
            if args.clsu.distance_fn == "SquaredL2Loss":
                distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
                distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
        elif args.clsu.evaluation_mode == "both":
            # L2
            audio_distance = torch.cdist(a_p, t_p, p=2)  # .pow(2)
            video_distance = torch.cdist(v_p, t_p, p=2)  # .pow(2)
            if args.clsu.distance_fn == "SquaredL2Loss":
                audio_distance = audio_distance.pow(2)
                video_distance = video_distance.pow(2)

            # Sum
            distance_mat[:, seen_unseen_array] = (audio_distance + video_distance)

            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=args.device)
            mask[seen_label_array] = 99999999999999
            distance_mat_zsl = distance_mat + mask

        for beta in betas:
            #--------------------- evaluating gzsl ---------------------
            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=args.device) + beta
            mask[unseen_label_array] = 0
            distance_mat_gzsl = distance_mat + mask
            min_dis = torch.min(distance_mat_gzsl, dim=1)[0].cpu().numpy()
            betas_dis_list.append(min_dis)

            neighbor_batch = torch.argmin(distance_mat_gzsl, dim=1)
            assert (neighbor_batch.unsqueeze(1) == seen_unseen_array).any(dim=1).all()
            assert (targets.unsqueeze(1) == seen_unseen_array).any(dim=1).all()
            match_idx = neighbor_batch.eq(targets.int()).nonzero().flatten()
            unmatch_idx = neighbor_batch.ne(targets.int()).nonzero().flatten()

            if args.clsu.multi_evaluation is True:
                for sample_idx in range(neighbor_batch.shape[0]):
                    min_dis_idx = neighbor_batch[sample_idx].item()
                    distance_mat_gzsl[sample_idx, min_dis_idx] = 9999999999999
                second_neighbor_batch = torch.argmin(distance_mat_gzsl, dim=1)
                assert (second_neighbor_batch.unsqueeze(1) == seen_unseen_array).any(dim=1).all()
                
                for sample_idx in range(neighbor_batch.shape[0]):
                    min_dis_idx = second_neighbor_batch[sample_idx].item()
                    distance_mat_gzsl[sample_idx, min_dis_idx] = 9999999999999
                third_neighbor_batch = torch.argmin(distance_mat_gzsl, dim=1)
                assert (third_neighbor_batch.unsqueeze(1) == seen_unseen_array).any(dim=1).all()

                assert torch.all(torch.eq(second_neighbor_batch, neighbor_batch)).item() == 0
                assert torch.all(torch.eq(third_neighbor_batch, neighbor_batch)).item() == 0
                assert torch.all(torch.eq(third_neighbor_batch, second_neighbor_batch)).item() == 0
                second_match_idx = second_neighbor_batch.eq(targets.int()).nonzero().flatten()
                third_match_idx = third_neighbor_batch.eq(targets.int()).nonzero().flatten()

                match_idx = torch.cat((match_idx, second_match_idx, third_match_idx))


            match_counts = torch.bincount(neighbor_batch[match_idx], minlength=len(dataset.all_class_ids))[
                seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=args.device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            seen_recall_dict = per_class_recall[seen_label_array]
            unseen_recall_dict = per_class_recall[unseen_label_array]
            s = seen_recall_dict.mean()
            u = unseen_recall_dict.mean()

            # if save_performances:
            #     seen_dict = {k: v for k, v in zip(np.array(dataset.all_class_names)[seen_label_array.cpu().numpy()], seen_recall_dict.cpu().numpy())}
            #     unseen_dict = {k: v for k, v in zip(np.array(dataset.all_class_names)[unseen_label_array.cpu().numpy()], unseen_recall_dict.cpu().numpy())}
            #     save_class_performances(seen_dict, unseen_dict, dataset.dataset_name)

            hm = (2 * u * s) / ((u + s) + np.finfo(float).eps)
            tpr = 0
            tnr = 0
            for i in neighbor_batch[seen_indices]:
                if i.item() in seen_class_ids:
                    tpr += 1 / seen_num
            for i in neighbor_batch[unseen_indices]:
                if i.item() in unseen_class_ids:
                    tnr += 1 / unseen_num
            fpr = 1 - tnr
            # print(tpr, fpr)
            # sys.exit()
            
            #--------------------- evaluating zsl ---------------------
            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=args.device)
            mask[seen_label_array] = 99999999999999
            distance_mat_zsl = distance_mat + mask
            neighbor_batch_zsl = torch.argmin(distance_mat_zsl, dim=1)
            match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()

            if args.clsu.multi_evaluation is True:
                for sample_idx in range(neighbor_batch_zsl.shape[0]):
                    min_dis_idx = neighbor_batch_zsl[sample_idx].item()
                    distance_mat_zsl[sample_idx, min_dis_idx] = 9999999999999
                second_neighbor_batch = torch.argmin(distance_mat_zsl, dim=1)
                assert (second_neighbor_batch.unsqueeze(1) == seen_unseen_array).any(dim=1).all()
                
                for sample_idx in range(neighbor_batch_zsl.shape[0]):
                    min_dis_idx = second_neighbor_batch[sample_idx].item()
                    distance_mat_zsl[sample_idx, min_dis_idx] = 9999999999999
                third_neighbor_batch = torch.argmin(distance_mat_zsl, dim=1)
                assert (third_neighbor_batch.unsqueeze(1) == seen_unseen_array).any(dim=1).all()

                assert torch.all(torch.eq(second_neighbor_batch, neighbor_batch_zsl)).item() == 0
                assert torch.all(torch.eq(third_neighbor_batch, neighbor_batch_zsl)).item() == 0
                assert torch.all(torch.eq(third_neighbor_batch, second_neighbor_batch)).item() == 0
                second_match_idx = second_neighbor_batch.eq(targets.int()).nonzero().flatten()
                third_match_idx = third_neighbor_batch.eq(targets.int()).nonzero().flatten()

                match_idx = torch.cat((match_idx, second_match_idx, third_match_idx))

            match_counts = torch.bincount(neighbor_batch_zsl[match_idx], minlength=len(dataset.all_class_ids))[seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=args.device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            zsl = per_class_recall[unseen_label_array].mean()


            #--------------------- evaluating fsl ---------------------
            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=args.device)
            mask[unseen_label_array] = 99999999999999
            distance_mat_fsl = distance_mat + mask
            neighbor_batch_fsl = torch.argmin(distance_mat_fsl, dim=1)
            match_idx = neighbor_batch_fsl.eq(targets.int()).nonzero().flatten()

            if args.clsu.multi_evaluation is True:
                for sample_idx in range(neighbor_batch_fsl.shape[0]):
                    min_dis_idx = neighbor_batch_fsl[sample_idx].item()
                    distance_mat_fsl[sample_idx, min_dis_idx] = 9999999999999
                second_neighbor_batch = torch.argmin(distance_mat_fsl, dim=1)
                assert (second_neighbor_batch.unsqueeze(1) == seen_unseen_array).any(dim=1).all()
                
                for sample_idx in range(neighbor_batch_fsl.shape[0]):
                    min_dis_idx = second_neighbor_batch[sample_idx].item()
                    distance_mat_fsl[sample_idx, min_dis_idx] = 9999999999999
                third_neighbor_batch = torch.argmin(distance_mat_fsl, dim=1)
                assert (third_neighbor_batch.unsqueeze(1) == seen_unseen_array).any(dim=1).all()

                assert torch.all(torch.eq(second_neighbor_batch, neighbor_batch_fsl)).item() == 0
                assert torch.all(torch.eq(third_neighbor_batch, neighbor_batch_fsl)).item() == 0
                assert torch.all(torch.eq(third_neighbor_batch, second_neighbor_batch)).item() == 0
                second_match_idx = second_neighbor_batch.eq(targets.int()).nonzero().flatten()
                third_match_idx = third_neighbor_batch.eq(targets.int()).nonzero().flatten()

                match_idx = torch.cat((match_idx, second_match_idx, third_match_idx))

            match_counts = torch.bincount(neighbor_batch_fsl[match_idx], minlength=len(dataset.all_class_ids))[
                seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=args.device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            fsl = per_class_recall[seen_label_array].mean()

            zsl_scores.append(zsl.item())
            fsl_scores.append(fsl.item())
            seen_scores.append(s.item())
            unseen_scores.append(u.item())
            hm_scores.append(hm.item())
            per_class_recalls.append(per_class_recall.tolist())
            tpr_fpr.append((tpr, fpr))

        argmax_hm = np.argmax(hm_scores)
        max_seen = seen_scores[argmax_hm]
        max_zsl = zsl_scores[argmax_hm]
        max_fsl = fsl_scores[argmax_hm]
        max_unseen = unseen_scores[argmax_hm]
        max_hm = hm_scores[argmax_hm]
        max_recall = per_class_recalls[argmax_hm]
        best_beta = betas[argmax_hm].item()
        best_beta_dis_list = [betas_dis_list[argmax_hm]]
        best_beta_tpr_fpr = tpr_fpr[argmax_hm]

    return {
        "seen": max_seen,
        "unseen": max_unseen,
        "hm": max_hm,
        "recall": max_recall,
        "zsl": max_zsl,
        "beta": best_beta,
        "fsl": max_fsl,
    }

def cal_match_feature_label(data_feature, data_label, plot_ids, num_per_class):
    data_feature_cls = torch.FloatTensor()
    data_label_cls = torch.IntTensor()
    data_num = 0
    for _id in plot_ids:
        matched = data_feature[data_label == _id].cpu()
        if len(matched) < num_per_class:
            data_num += len(matched)
            indices = torch.randperm(matched.size(0))
            data_label_cls = torch.cat((data_label_cls, torch.ones(len(matched))*_id), dim=0)
        else:
            data_num += num_per_class
            indices = torch.randperm(matched.size(0))[:num_per_class]
            data_label_cls = torch.cat((data_label_cls, torch.ones(num_per_class)*_id), dim=0)
        data_feature_cls = torch.cat((data_feature_cls, matched[indices]), dim=0)
    return data_feature_cls, data_label_cls.int(), data_num



def draw_roc_curve(pos_label, epoch_data_list=None, betas_dis_list=None, targets_bin=None, save_name=None, save_path=None):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pickle

    color_map = cm.get_cmap('viridis')

    if epoch_data_list is not None:
        with open(f'{save_path}/{save_name}.pkl', 'wb') as f:
            pickle.dump(epoch_data_list, f)
        for i, epoch_data in enumerate(epoch_data_list):
            length = len(epoch_data_list)
            entropy_seen, entropy_unseen = epoch_data
            entropy = np.concatenate((entropy_seen, entropy_unseen))
            seen_bin = np.ones(len(entropy_seen))
            unseen_bin = np.zeros(len(entropy_unseen))
            bin_label = np.concatenate((seen_bin, unseen_bin))
            fpr, tpr, thresholds = roc_curve(bin_label, entropy, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            color = color_map(i / len(epoch_data_list)) 
            plt.plot(fpr, tpr, lw=2, color=color, label='ROC curve (epoch %d, area = %0.2f)' % (i+1, roc_auc))
            last_data = (fpr, tpr, roc_auc)
    else:
        with open(f'{save_path}/{save_name}.pkl', 'wb') as f:
            pickle.dump((betas_dis_list, targets_bin), f)
        for i, beta_dis in enumerate(betas_dis_list):
            length = len(betas_dis_list)
            beta_dis = 1/beta_dis
            fpr, tpr, thresholds = roc_curve(targets_bin, beta_dis, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            color = color_map(i / len(betas_dis_list))
            plt.plot(fpr, tpr, lw=2, color=color, label='ROC curve (epoch %d, area = %0.2f)' % (i+1, roc_auc))
            last_data = (fpr, tpr, roc_auc)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    # 添加最后一个epoch的图例
    if last_data is not None:
        fpr, tpr, roc_auc = last_data
        plt.legend(['ROC curve (epoch %d, AUC = %0.2f)' % (length, roc_auc)], loc='lower right')

    plt.grid(True)
    plt.savefig(f'{save_path}/{save_name}.png')
    plt.close()
    print(f">>>>> ROC curve saved: {save_name}.png, AUC={roc_auc:.4f} <<<<<")
    # plt.show()

def test_fpr_60(args, beta, test_beta, distance_mat_gzsl, min_dis, targets, targets_bin, threshold, seen_label_array, unseen_label_array, seen_unseen_array, all_class_num):
    if beta >= test_beta + 0.01 or beta <= test_beta - 0.01:
        return False
    print('___testing fpr_60___')
    pos_targets = np.where(targets_bin==1)[0]
    neg_targets = np.where(targets_bin==0)[0]
    pos_dis_idx = np.where(min_dis > threshold)[0]
    neg_dis_idx = np.where(min_dis <= threshold)[0]
    correct_pos_idx = pos_dis_idx[np.where(targets_bin[pos_dis_idx] == 1)[0]]
    correct_neg_idx = neg_dis_idx[np.where(targets_bin[neg_dis_idx] == 0)[0]]
    wrong_pos_idx = pos_dis_idx[np.where(targets_bin[pos_dis_idx] == 0)[0]]
    wrong_neg_idx = neg_dis_idx[np.where(targets_bin[neg_dis_idx] == 1)[0]]
    
    tpr = len(correct_pos_idx) / len(pos_targets)
    fpr = 1- len(correct_neg_idx) / len(neg_targets)
    mask_wrong = torch.zeros(distance_mat_gzsl.shape, dtype=torch.long, device=args.device)
    mask_wrong[wrong_pos_idx, targets[wrong_pos_idx]] = 99999999999999
    mask_wrong[wrong_neg_idx, targets[wrong_neg_idx]] = 99999999999999
    distance_mat_test = distance_mat_gzsl + mask_wrong

    mask = torch.zeros(all_class_num, dtype=torch.long, device=args.device)
    mask[seen_label_array] = 99999999999999
    distance_mat_zsl = distance_mat_test + mask
    neighbor_batch_zsl = torch.argmin(distance_mat_zsl, dim=1)
    match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()
    match_counts = torch.bincount(neighbor_batch_zsl[match_idx], minlength=all_class_num)[
        seen_unseen_array]
    target_counts = torch.bincount(targets, minlength=all_class_num)[seen_unseen_array]
    per_class_recall = torch.zeros(all_class_num, dtype=torch.float, device=args.device)
    per_class_recall[seen_unseen_array] = match_counts / target_counts
    zsl = per_class_recall[unseen_label_array].mean()

    mask = torch.zeros(all_class_num, dtype=torch.long, device=args.device)
    mask[unseen_label_array] = 99999999999999
    distance_mat_fsl = distance_mat_test + mask
    neighbor_batch_fsl = torch.argmin(distance_mat_fsl, dim=1)
    match_idx = neighbor_batch_fsl.eq(targets.int()).nonzero().flatten()
    match_counts = torch.bincount(neighbor_batch_fsl[match_idx], minlength=all_class_num)[
        seen_unseen_array]
    target_counts = torch.bincount(targets, minlength=all_class_num)[seen_unseen_array]
    per_class_recall = torch.zeros(all_class_num, dtype=torch.float, device=args.device)
    per_class_recall[seen_unseen_array] = match_counts / target_counts
    fsl = per_class_recall[seen_label_array].mean()

    hm = (2 * zsl * fsl) / ((zsl + fsl) + np.finfo(float).eps)
    print(f"TPR={tpr:.4f} FPR={fpr:.4f}\nS={fsl.item():.4f} U={zsl.item():.4f} HM={hm.item():.4f}")
    print('___test completed___')
    return True


def cal_cos_sim(a, b):
    dot_product = torch.mm(a, b.T)
    norm1 = torch.norm(a, dim=1)
    norm2 = torch.norm(b, dim=1)
    cosine_similarity = dot_product / (torch.outer(norm1, norm2))
    return cosine_similarity.mean().item()

def cal_cos_sim_mutual_max(a, features, labels, ids):
    max_cos_sim_mutual = 0
    for _id in ids:
        b = features[labels==_id]
        cos_sim_id__id = cal_cos_sim(a, b)
        if max_cos_sim_mutual < cos_sim_id__id:
            max_cos_sim_mutual = cos_sim_id__id
    return max_cos_sim_mutual

def get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes


def load_model_weights(weights_path, model):
    logging.info(f"Loading model weights from {weights_path}")
    load_dict = torch.load(weights_path)
    model_weights = load_dict["model"]
    epoch = load_dict["epoch"]
    logging.info(f"Load from epoch: {epoch}")
    load_model_parameters(model, model_weights)
    return epoch
    
def plot_hist_from_dict(dict):
    plt.bar(range(len(dict)), list(dict.values()), align="center")
    plt.xticks(range(len(dict)), list(dict.keys()), rotation='vertical')
    plt.tight_layout()
    plt.show()

def save_class_performances(seen_dict, unseen_dict, dataset_name):
    seen_path = Path(f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_seen.pkl")
    unseen_path = Path(f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_unseen.pkl")
    with seen_path.open("wb") as f:
        pickle.dump(seen_dict, f)
        logging.info(f"Saving seen class performances to {seen_path}")
    with unseen_path.open("wb") as f:
        pickle.dump(unseen_dict, f)
        logging.info(f"Saving unseen class performances to {unseen_path}")

def find_median(list_to_find):
    sorted_list = sorted(list_to_find)
    length = len(sorted_list)
    if length % 2 == 1:
        # 对于奇数长度的列表，中位数是中间的那个数
        median = sorted_list[length // 2]
    else:
        # 对于偶数长度的列表，中位数是中间两个数的平均值
        middle1 = sorted_list[length // 2 - 1]
        middle2 = sorted_list[length // 2]
        median = (middle1 + middle2) / 2

    return median