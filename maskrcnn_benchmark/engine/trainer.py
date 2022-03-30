# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
# INFERENCE DURING TRAINING
import os
from .inference import evaluate_during_training

import copy
from maskrcnn_benchmark.utils.comm import get_world_size, ParamDict
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

fish_times = 0
def fish_step(meta_weights, inner_weights, meta_lr):
    global fish_times
    fish_times+=1
    meta_weights, weights = ParamDict(meta_weights), ParamDict(inner_weights)

    # fish all
    meta_weights += meta_lr * sum([weights - meta_weights], 0 * meta_weights)

    # restore da weights
    da_weights = {}
    for keys in weights:
        if 'da_head' in keys:
            da_weights[keys] = weights[keys]
    meta_weights.update(da_weights)
    del weights
    del da_weights

    return meta_weights

def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]

    model.train()

    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def save_model_only(model, name):
    data = {}
    data["model"] = model.state_dict()
    torch.save(data, name)

def do_da_train(
    model,
    source_data_loader,
    target_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    # Make an outer model
    model2=copy.deepcopy(model)
    model2.train()
    start_training_time = time.time()
    end = time.time()
    best_map = 0 # SAVE BEST MODEL PURPOSE
    global fish_times
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        # for name, m in model.named_modules():
        #     if name == "da_heads.grl_ins":
        #         hook = m.register_backward_hook(backward_hook)

        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        scheduler.step()

        if cfg.MODEL.NOISEGRADMATCH_ON: # Fish
            if iteration % cfg.MODEL.FISH_PERIOD == 0 or iteration == max_iter:
                meta_weights = fish_step(meta_weights=model2.state_dict(),
                                    inner_weights=model.state_dict(),
                                    meta_lr=0.01 / cfg.MODEL.EAGR_LR)
                reset_inner_weight = False
                if reset_inner_weight:
                    model.load_state_dict(copy.deepcopy(meta_weights))
                    model2=copy.deepcopy(model)
                else:
                    model2.load_state_dict(copy.deepcopy(meta_weights))


        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter: # logger
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                        "fish times: {fish_times:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    fish_times=int(fish_times),
                )
            )

        
        if cfg.MODEL.NOISEGRADMATCH_ON: # Fish
            if iteration % cfg.SOLVER.VAL_ITER == 0 and iteration > 0:
                logger.info('----------------------------- Evaluating MODEL2 -----------------------------')
                evaluate_during_training(cfg, model2)
                if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "result.txt")):
                    with open(os.path.join(cfg.OUTPUT_DIR, "result.txt"), "r") as f:
                        cur_map = float(f.readline()[5:]) # mAP: 
                if cur_map > best_map:
                    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model_best.pth")):
                        os.remove(os.path.join(cfg.OUTPUT_DIR, "model_best.pth"))
                    save_model_only(model2, os.path.join(cfg.OUTPUT_DIR, "model_best.pth"))
                    # checkpointer.save("model_best", **arguments)
                    best_map = cur_map
                    logger.info('----------------------------- BEST MODEL UPDATED -----------------------------')
                    logger.info('-----------------------------     mAP: {}      -----------------------------'.format(best_map))
                # model2.train()
            if iteration == max_iter-1:
                logger.info('---------------------- EVALUATING FINAL OUTER MODEL -----------------------')
                evaluate_during_training(cfg, model2)
                save_model_only(model2, os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
                # checkpointer.save("model_final", **arguments)
        else:
            if iteration % cfg.SOLVER.VAL_ITER == 0 and iteration > 0:
                logger.info('----------------------------- Evaluating MODEL1 -----------------------------')
                evaluate_during_training(cfg, model)
                if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "result.txt")):
                    with open(os.path.join(cfg.OUTPUT_DIR, "result.txt"), "r") as f:
                        cur_map = float(f.readline()[5:]) # mAP: 
                if cur_map > best_map:
                    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model_best.pth")):
                        os.remove(os.path.join(cfg.OUTPUT_DIR, "model_best.pth"))
                    checkpointer.save("model_best", **arguments)
                    best_map = cur_map
                    logger.info('----------------------------- BEST MODEL UPDATED -----------------------------')
                    logger.info('-----------------------------     mAP: {}      -----------------------------'.format(best_map))
                model.train()
            if iteration == max_iter-1:
                evaluate_during_training(cfg, model)
                checkpointer.save("model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )