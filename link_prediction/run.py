import os
import sys
import math
import pprint
import time

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util

separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data_list, valid_data_list, filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_loaders = []
    for train_data in train_data_list:
        train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
        train_loaders.append(torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler))

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    train_times = []
    inference_times = []
    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            train_start = time.time()
            losses = []
            sampler.set_epoch(epoch)
            for j in range(len(train_loaders)):
                train_loader = train_loaders[j]
                for batch in train_loader:
                    if hasattr(cfg.model, "edge_embed_dim"):
                        edge_embed = cfg.model.edge_embed_dim
                    else:
                        edge_embed = None
                    batch = tasks.negative_sampling(train_data_list[j], batch, cfg.task.num_negative, edge_embed_dim=edge_embed,
                                                    strict=cfg.task.strict_negative)
                    pred = parallel_model(train_data_list[j], batch)
                    target = torch.zeros_like(pred)
                    target[:, 0] = 1
                    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                    neg_weight = torch.ones_like(pred)
                    if cfg.task.adversarial_temperature > 0:
                        with torch.no_grad():
                            neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                    else:
                        neg_weight[:, 1:] = 1 / cfg.task.num_negative
                    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                    loss = loss.mean()

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                        logger.warning(separator)
                        logger.warning("binary cross entropy: %g" % loss)
                    losses.append(loss.item())
                    batch_id += 1

                if util.get_rank() == 0:
                    avg_loss = sum(losses) / len(losses)
                    logger.warning(separator)
                    logger.warning("Epoch %d end" % epoch)
                    logger.warning(line)
                    logger.warning("average binary cross entropy: %g" % avg_loss)

        train_end = time.time()
        epoch_train_time = train_end - train_start
        train_times.append(epoch_train_time)
        
        logger.warning(f"Epoch {epoch} training time: {epoch_train_time:.2f} seconds")

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")

        result = test_list(cfg, model, valid_data_list, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch
        
        logger.warning(separator)
        logger.warning("Evaluate on test")
        
        inference_start = time.time()
        test_result =  test_list(cfg, model, test_data_list, filtered_data=filtered_data)
        inference_end = time.time()
        epoch_inference_time = inference_end - inference_start
        inference_times.append(epoch_inference_time)
        logger.warning(f"Epoch {epoch} inference time: {epoch_inference_time:.2f} seconds")
        logger.warning(f"Test result: {test_result}")
        logger.warning(separator)
    
    logger.warning(separator)
    logger.warning("Training and Inference Time Summary:")
    logger.warning("Training times per epoch: %s" % ", ".join([f"{t:.2f}s" for t in train_times]))
    logger.warning("Inference times per epoch: %s" % ", ".join([f"{t:.2f}s" for t in inference_times]))
    logger.warning(f"Average training time per epoch: {sum(train_times)/len(train_times):.2f} seconds")
    logger.warning(f"Average inference time per epoch: {sum(inference_times)/len(inference_times):.2f} seconds")
    logger.warning(separator)

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()

def test_list(cfg, model, test_data_list, filtered_data=None):
    mrr_list = []
    for test_data in test_data_list:
        mrr = test(cfg, model, test_data, filtered_data)
        mrr_list.append(mrr)
    return sum(mrr_list) / len(mrr_list)

@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        if hasattr(cfg.model, "edge_embed_dim"):
            edge_embed = cfg.model.edge_embed_dim
        else:
            edge_embed = None
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch, edge_embed)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch, edge_embed)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
    mrr = (1 / all_ranking.float()).mean()

    return mrr

def rename_attribute(data, old_attr, new_attr):
    if hasattr(data, old_attr):
        setattr(data, new_attr, getattr(data, old_attr))
        delattr(data, old_attr)


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        print("Working dir: %s" % working_dir)
        logger.warning("Working dir: %s" % working_dir)
        util.git_logs(working_dir)
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Train categories: %s" % cfg["dataset"]["train_categories"])
        logger.warning("Test categories: %s" % cfg["dataset"]["test_categories"])
        logger.warning("Feature method: %s" % cfg["model"]["feature_method"])
        logger.warning("Config file: %s" % args.config)
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset_list, num_relations = util.build_dataset(cfg)

    cfg.model.num_relation = num_relations
    model = util.build_model(cfg)
    device = util.get_device(cfg)
    model = model.to(device)
    train_data_list, valid_data_list, test_data_list = dataset_list
    train_data_list = [train_data_list[i].to(device) for i in range(len(train_data_list))]
    valid_data_list = [valid_data_list[i].to(device) for i in range(len(valid_data_list))]
    test_data_list = [test_data_list[i].to(device) for i in range(len(test_data_list))]

    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None

    if args.checkpoint:
        if util.get_rank() == 0:
            logger.warning(f"Loading model from checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])

    train_and_validate(cfg, model, train_data_list, valid_data_list, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test_list(cfg, model, valid_data_list, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test_list(cfg, model, test_data_list, filtered_data=filtered_data)
