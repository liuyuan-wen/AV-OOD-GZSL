import logging
import torch
import sys


def train_step(args, data_loader, model, epoch, epochs, metrics, stats, optimize, train_type='clsu'):
    model.train()
    batch_loss = 0
    mean_lossD, mean_lossG = 0, 0
    mean_lossR, mean_lossC = 0, 0 

    for batch_idx, (data, target) in enumerate(data_loader):
        p = data["positive"]
        q = data["negative"]

        x_p_a = p["audio"].to(args.device)
        x_p_v = p["video"].to(args.device)
        x_p_t = p["text"].to(args.device)
        x_p_num = target["positive"].to(args.device)

        x_q_a = q["audio"].to(args.device)
        x_q_v = q["video"].to(args.device)
        x_q_t = q["text"].to(args.device)

        inputs = (x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v, x_q_t)
        
        if train_type=='clsu':
            if args.clsu.z_score_inputs: # False
                inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])
            loss = model.optimize_params(*inputs, optimize=optimize)
            batch_loss += loss.item()
            iteration = len(data_loader) * epoch + batch_idx

        elif train_type=='gen':
            gan_costs = model.optimize_params(*inputs)
            mean_lossD += gan_costs[0]  # D_cost
            mean_lossG += gan_costs[1]  # G_cost
            mean_lossC += gan_costs[3]  # embed_err
            mean_lossR += gan_costs[2]  # R_cost
            Wasserstein_dist_D = gan_costs[4]
            Wasserstein_dist_G = gan_costs[5]

    if train_type=='clsu':
        logger = logging.getLogger()
        metrics.reset()
        batch_loss /= (batch_idx + 1)
        stats.update((epoch, batch_loss, None))
        logger.info(
            f"TRAIN\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"Loss: {batch_loss:.4f}\t"
        )
    
    elif train_type=='gen':
        mean_lossD /= (batch_idx + 1)
        mean_lossG /= (batch_idx + 1)
        mean_lossC /= (batch_idx + 1)
        mean_lossR /= (batch_idx + 1)
        print('Loss_D: %.4f  Wasserstein_dist: %.4f, Loss_G: %.4f  Wasserstein_dist: %.4f' % (mean_lossD, Wasserstein_dist_D, mean_lossG, Wasserstein_dist_G))

    return
    


def test_step(args, data_loader, model, epoch, epochs, writer, metrics, stats):

    logger = logging.getLogger()
    model.eval()
    metrics.reset()

    with torch.no_grad():
        batch_loss = 0
        test_hm_score = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            p = data["positive"]
            q = data["negative"]

            x_p_a = p["audio"].to(args.device)
            x_p_v = p["video"].to(args.device)
            x_p_t = p["text"].to(args.device)
            x_p_num = target["positive"].to(args.device)

            x_q_a = q["audio"].to(args.device)
            x_q_v = q["video"].to(args.device)
            x_q_t = q["text"].to(args.device)

            inputs = (x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v, x_q_t)

            if args.clsu.z_score_inputs:
                inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])

            loss = model.optimize_params(*inputs)

            batch_loss += loss.item()
        
            iteration = len(data_loader) * epoch + batch_idx
            if iteration % len(data_loader) == 0:
                metrics()
                for key, value in metrics.value().items():
                    if "recall" in key:
                        continue
                    if key == "test_beta":
                        test_beta = value
                    if key == "test_hm":
                        test_hm_score = value
                    if key == "test_zsl":
                        test_zsl_score=value
                    if key == "test_fsl":
                        test_fsl_score=value
                    if key == "test_seen":
                        test_seen_score=value
                    if key == "test_unseen":
                        test_unseen_score=value
                    writer.add_scalar(
                        key, value, iteration
                    )

        batch_loss /= (batch_idx + 1)
        stats.update((epoch, batch_loss, test_hm_score))

        logger.info(
            f"TEST\t"
            f"Epoch: {epoch}/{epochs}  "
            f"Iteration={iteration}  "
            f"Loss={batch_loss:.4f}  "
            f"FSL={test_fsl_score:.4f}  "
            f"ZSL={test_zsl_score:.4f}  "
        )
    return
