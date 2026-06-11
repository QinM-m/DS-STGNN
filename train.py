
import os
import time
import shutil
import yaml
import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt



import numpy as np
import cv2
import pandas as pd
import torch
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn.functional as F
from torch import optim as optim_module
# import imgaug as ia
# from imgaug import augmenters as iaa

from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torch_geometric.data import Data, Batch

from libs import datasets
from models.resnet import resnet50_fc256, load_pretrained_weights
from models.mpn import MOTMPNet

from torch_geometric.utils import to_networkx
import networkx as nx
from skimage.io import imread

from libs import utils
from sklearn.metrics.pairwise import paired_distances
from scipy.sparse.csgraph import connected_components
from sklearn import metrics


def apply_density_aware_sparsification(edge_ixs_np, data_df_g, prev_max_counter, reid_embeds_np=None, eta=0.5, k_min=2,
                                       k_max=10):

    if edge_ixs_np.shape[1] == 0:
        return edge_ixs_np


    xws = data_df_g['xw'].values
    yws = data_df_g['yw'].values
    coords = np.stack([xws, yws], axis=1)  # shape (N, 2)


    src_nodes = edge_ixs_np[0] - prev_max_counter
    dst_nodes = edge_ixs_np[1] - prev_max_counter


    if reid_embeds_np is not None:
        src_global = edge_ixs_np[0]
        dst_global = edge_ixs_np[1]
        reid_src = reid_embeds_np[src_global]
        reid_dst = reid_embeds_np[dst_global]


        norm_src = np.linalg.norm(reid_src, axis=1, keepdims=True) + 1e-12
        norm_dst = np.linalg.norm(reid_dst, axis=1, keepdims=True) + 1e-12


        reid_src_norm = reid_src / norm_src
        reid_dst_norm = reid_dst / norm_dst
        cos_sims = np.sum(reid_src_norm * reid_dst_norm, axis=1)


        cos_dists = 1.0 - cos_sims
    else:
        cos_dists = np.zeros(edge_ixs_np.shape[1])

    num_nodes = len(data_df_g)
    degrees = np.bincount(src_nodes, minlength=num_nodes)


    dynamic_k = np.ceil(eta * degrees).astype(int)
    dynamic_k = np.clip(dynamic_k, k_min, k_max)

    kept_edges_idx = []
    for i in range(num_nodes):
        k_i = dynamic_k[i]
        edge_mask = (src_nodes == i)
        edge_indices = np.where(edge_mask)[0]

        if len(edge_indices) <= k_i:
            kept_edges_idx.extend(edge_indices)
        else:
            dests = dst_nodes[edge_indices]
            dist = np.linalg.norm(coords[i] - coords[dests], axis=1)


            dist_norm = np.clip(dist / 10.0, 0, 1.0)

            if reid_embeds_np is not None:
                local_cos_dists = cos_dists[edge_indices]

                combined_score = 0.5 * dist_norm + 0.5 * (local_cos_dists / 2.0)
                top_k_local_idx = np.argsort(combined_score)[:k_i]
            else:
                top_k_local_idx = np.argsort(dist)[:k_i]

            kept_edges_idx.extend(edge_indices[top_k_local_idx])

    kept_edges_idx = np.array(kept_edges_idx)
    return edge_ixs_np[:, kept_edges_idx]

def compute_loss_acc(outputs, batch, criterion, criterion_no_reduction,  mode):
    # global num_edges, num_edges1
    # num_edges1 += np.int(positive_vals.cpu())
    # num_edges += np.int(labels.shape[0])

    # Define Balancing weight
    labels = batch.edge_labels.view(-1)

    # Compute Weighted BCE:
    loss = 0
    loss_class1 = 0
    loss_class0 = 0
    precision_class1 = list()
    precision_class0 = list()
    precision_all = list()

    list_pred_prob = list()
    num_steps = len(outputs['classified_edges'])

   # Compute loss of all the steps and sum them

    ## FOR CONSIDERING ONLY LAST 3 STEPS or less
    # step_ini = max(0,num_steps-3)
    # step_end = num_steps

    # comment FOR CONSIDERING ALL STEPS
    step_ini= 0
    step_end = num_steps

    for step in range(step_ini, step_end):
        preds = outputs['classified_edges'][step].view(-1)

        if mode == 'train':

            loss_per_sample = criterion_no_reduction(preds, labels)


            pos_mask = (labels == 1)
            neg_mask = (labels == 0)

            loss_pos = loss_per_sample[pos_mask]
            loss_neg = loss_per_sample[neg_mask]

            num_pos = loss_pos.size(0)

            num_hard_neg = min(loss_neg.size(0), max(num_pos * 3, 1))

            if num_hard_neg > 0 and loss_neg.size(0) > 0:

                hard_loss_neg, _ = torch.topk(loss_neg, num_hard_neg)

                step_loss = (loss_pos.sum() + hard_loss_neg.sum()) / (num_pos + num_hard_neg)
            else:

                step_loss = loss_per_sample.mean()

            loss += step_loss


            loss_class1 += torch.mean(loss_pos) if num_pos > 0 else torch.tensor(0.0).cuda()
            loss_class0 += torch.mean(loss_neg) if loss_neg.size(0) > 0 else torch.tensor(0.0).cuda()


        else:
            loss_per_sample = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
            loss_class1 += torch.mean(loss_per_sample[labels == 1])
            loss_class0 += torch.mean(loss_per_sample[labels == 0])

            loss += F.binary_cross_entropy_with_logits(preds, labels, reduction='mean')


        with torch.no_grad():
            sig = torch.nn.Sigmoid()
            preds_prob = sig(preds)
            list_pred_prob.append(preds_prob)


    # Precision is computed only with last step predictions
    with torch.no_grad():
        preds = outputs['classified_edges'][-1].view(-1)
        sig = torch.nn.Sigmoid()
        preds_prob = sig(preds)
        predictions = (preds_prob >= 0.5) * 1
        # Precision class 1
        index_label_1 = np.where(np.asarray(labels.cpu()) == 1)
        sum_successes_1 = np.sum(predictions.cpu().numpy()[index_label_1] == labels.cpu().numpy()[index_label_1])
        if sum_successes_1 == 0:
            precision_class1.append(0)
        else:
            precision_class1.append((sum_successes_1 / len(labels[index_label_1])) * 100.0)

        # Precision class 0
        index_label_0 = np.where(np.asarray(labels.cpu()) == 0)
        sum_successes_0 = np.sum(predictions.cpu().numpy()[index_label_0] == labels.cpu().numpy()[index_label_0])
        if sum_successes_0 == 0:
            precision_class0.append(0)
        else:
            precision_class0.append((sum_successes_0 / len(labels[index_label_0])) * 100.0)

        # Precision
        sum_successes = np.sum(predictions.cpu().numpy() == labels.cpu().numpy())
        if sum_successes == 0:
            precision_all.append(0)
        else:
            precision_all.append((sum_successes / len(labels) )* 100.0)
   #  end


   # Compute loss and precision only of the last step




    # preds = outputs['classified_edges'][-1].view(-1)
    # #
    # if mode == 'train':
    #
    #     loss_per_sample = criterion_no_reduction(preds, labels)
    #     loss = criterion(preds, labels)
    #
    #     loss_class1 = torch.mean(loss_per_sample[labels == 1])
    #     loss_class0 = torch.mean(loss_per_sample[labels == 0])
    #
    #
    # else:
    #     loss_per_sample = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
    #     loss_class1 = torch.mean(loss_per_sample[labels == 1])
    #     loss_class0 = torch.mean(loss_per_sample[labels == 0])
    #
    #     loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='mean')
    #
    #
    # with torch.no_grad():
    #     sig = torch.nn.Sigmoid()
    #     preds_prob = sig(preds)
    #     predictions = (preds_prob >= 0.5) * 1
    #     # Precision class 1
    #     index_label_1 = np.where(np.asarray(labels.cpu()) == 1)
    #     sum_successes_1 = np.sum(predictions.cpu().numpy()[index_label_1] == labels.cpu().numpy()[index_label_1])
    #     if sum_successes_1 == 0:
    #         precision_class1.append(0)
    #         # precision_class1 = 0
    #     else:
    #         precision_class1.append((sum_successes_1 / len(labels[index_label_1])) * 100.0)
    #         # precision_class1 = (sum_successes_1 / len(labels[index_label_1])) * 100.0
    #
    #
    #     # Precision class 0
    #     index_label_0 = np.where(np.asarray(labels.cpu()) == 0)
    #     sum_successes_0 = np.sum(predictions.cpu().numpy()[index_label_0] == labels.cpu().numpy()[index_label_0])
    #
    #     if sum_successes_0 == 0:
    #         precision_class0.append(0)
    #         # precision_class0 = (0)
    #
    #     else:
    #         precision_class0.append((sum_successes_0 / len(labels[index_label_0])) * 100.0)
    #         # precision_class0 = (sum_successes_0 / len(labels[index_label_0])) * 100.0
    #
    #
    #     # Precision
    #     sum_successes = np.sum(predictions.cpu().numpy() == labels.cpu().numpy())
    #     if sum_successes == 0:
    #         precision_all.append(0.5)
    #         # precision_all = 0
    #
    #     else:
    #         precision_all.append((sum_successes / len(labels)) * 100.0)
    #         # precision_all = (sum_successes / len(labels)) * 100.0
    #
    #     for step in range(num_steps):
    #         preds = outputs['classified_edges'][step].view(-1)
    #         preds_prob = sig(preds)
    #
    #         list_pred_prob.append(preds_prob)

    #     a=1
            ## end

    return loss, precision_class1, precision_class0, precision_all, loss_class1, loss_class0, list_pred_prob


def train(CONFIG, train_loader, cnn_model, mpn_model, epoch, optimizer,results_path,train_loss_in_history,
          train_prec1_in_history,train_prec0_in_history, train_prec_in_history, train_dataset, dataset_dir , criterion, criterion_no_reduction,list_mean_probs_history):

    train_losses = utils.AverageMeter('losses', ':.4e')
    train_losses1 = utils.AverageMeter('losses', ':.4e')
    train_losses0 = utils.AverageMeter('losses', ':.4e')

    train_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    train_precision_class1 = utils.AverageMeter('Precision_class1', ':6.2f')
    train_precision_class0 = utils.AverageMeter('Precision_class0', ':6.2f')
    train_precision = utils.AverageMeter('Precision', ':6.2f')
    mpn_model.train()
    n_steps = CONFIG['GRAPH_NET_PARAMS']['num_class_steps']
    list_mean_probs = {"0": {}, "1": {}}
    if n_steps >0:
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []

    else:
        n_steps = 1
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []


    for i, data in enumerate(train_loader):

        if i >= 0 :

            start_time = time.time()

            ########### Data extraction ###########

            [bboxes_t, data_df_t, bboxes_prev, data_df_prev, max_dist] = data
            len_graphs = [len(item) for item in data_df_t]

            with torch.no_grad():

                if len(bboxes_t) > 0:
                    bboxes_t_tensor = torch.cat(bboxes_t, dim=0).cuda() if isinstance(bboxes_t,
                                                                                      list) else bboxes_t.cuda()
                    if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                        node_embeds_t, reid_embeds_t = cnn_model(bboxes_t_tensor)
                    else:
                        node_embeds_t = cnn_model(bboxes_t_tensor)
                        reid_embeds_t = node_embeds_t
                else:
                    reid_embeds_t = torch.tensor([]).cuda()
                    node_embeds_t = torch.tensor([]).cuda()


                if len(bboxes_prev) > 0:
                    bboxes_prev_tensor = torch.cat(bboxes_prev, dim=0).cuda() if isinstance(bboxes_prev,
                                                                                            list) else bboxes_prev.cuda()
                    if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                        node_embeds_prev, reid_embeds_prev = cnn_model(bboxes_prev_tensor)
                    else:
                        node_embeds_prev = cnn_model(bboxes_prev_tensor)
                        reid_embeds_prev = node_embeds_prev
                else:
                    reid_embeds_prev = torch.tensor([]).cuda()


            if CONFIG['CNN_MODEL']['L2norm']:
                if reid_embeds_t.numel() > 0:
                    reid_embeds_t = F.normalize(reid_embeds_t, p=2, dim=1)
                if reid_embeds_prev.numel() > 0:
                    reid_embeds_prev = F.normalize(reid_embeds_prev, p=2, dim=1)


            reid_embeds = reid_embeds_t
            node_embeds = node_embeds_t
            data_df = data_df_t


            max_counter = 0
            prev_max_counter = 0


            global_prev_counter = 0


            edge_ixs = []
            node_label = []
            node_id_cam = []
            batch = []

            flag_visualize = False
            for g in range(len(len_graphs)):

                if flag_visualize:
                    pass
                else:
                    data_df[g] = data_df[g].assign(node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))
                    max_counter = max(data_df[g]['node'].values) + 1

                    node_embeds_g = torch.stack([node_embeds[i] for i in data_df[g]['node']])


                    prev_num_nodes = len(data_df_prev[g]) if len(data_df_prev) > 0 else 0
                    if prev_num_nodes > 0:
                        node_embeds_prev_g = node_embeds_prev[global_prev_counter: global_prev_counter + prev_num_nodes]
                        global_prev_counter += prev_num_nodes
                    else:
                        node_embeds_prev_g = torch.tensor([]).cuda()


                    edge_ixs_g = []
                    for id_cam in np.unique(data_df[g]['id_cam']):
                        ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                        ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                        edge_ixs_g.append(
                            torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                    edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                    edge_ixs_g_np = edge_ixs_g.cpu().numpy()


                    edge_ixs_g_np = apply_density_aware_sparsification(
                        edge_ixs_g_np, data_df[g], prev_max_counter,
                        reid_embeds_np=reid_embeds.cpu().numpy(),
                        eta=0.5, k_min=2, k_max=10
                    )
                    edge_ixs_g = torch.from_numpy(edge_ixs_g_np).cuda()
                    # =====================================================================

                    node_label_g = torch.from_numpy(data_df[g]['id'].values)

                    emb_dist_g = F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(-1, 1)
                    emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(
                        -1, 1)

                    xws_1 = np.expand_dims(
                        np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                        axis=1)
                    yws_1 = np.expand_dims(
                        np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                        axis=1)
                    xws_2 = np.expand_dims(
                        np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                        axis=1)
                    yws_2 = np.expand_dims(
                        np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                        axis=1)

                    points1 = np.concatenate((xws_1, yws_1), axis=1)
                    points2 = np.concatenate((xws_2, yws_2), axis=1)

                    spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))),
                                                     dim=1).cuda()
                    spatial_dist_g_norm = torch.from_numpy(spatial_dist_g.cpu().numpy() / max_dist[g]).cuda()

                    spatial_dist_manh_g = torch.unsqueeze(
                        (torch.from_numpy(paired_distances(points1, points2, metric='manhattan'))), dim=1).cuda()
                    spatial_dist_manh_g_norm = torch.from_numpy(spatial_dist_manh_g.cpu().numpy() / max_dist[g]).cuda()

                    if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                        edge_attr = torch.cat((emb_dist_g, emb_dist_g_cos), dim=1)
                    elif CONFIG['TRAINING']['ONLY_DIST']:
                        edge_attr = torch.cat(
                            (spatial_dist_g_norm.type(torch.float32), spatial_dist_manh_g_norm.type(torch.float32)),
                            dim=1)
                    else:
                        edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32),
                                               spatial_dist_manh_g_norm.type(torch.float32), emb_dist_g,
                                               emb_dist_g_cos), dim=1)

                    edge_labels_g = torch.from_numpy(
                        np.asarray(
                            [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                                   data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                             for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                    edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)


                    ids_t = data_df[g]['id'].values
                    ids_prev = data_df_prev[g]['id'].values if len(data_df_prev) > 0 else []

                    prev_idx_list, t_idx_list = [], []
                    if len(ids_prev) > 0:
                        for i_t, node_id in enumerate(ids_t):
                            if node_id in ids_prev:
                                i_prev = np.where(ids_prev == node_id)[0][0]
                                prev_idx_list.append(i_prev)
                                t_idx_list.append(i_t)

                    temporal_edge_index = torch.tensor([prev_idx_list, t_idx_list], dtype=torch.long).cuda()


                    data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_label_g, edge_attr=edge_attr,
                                edge_labels=edge_labels_g)


                    data.prev_x = node_embeds_prev_g
                    data.temporal_edge_index = temporal_edge_index

                    if temporal_edge_index.numel() > 0:
                        data.prev_edge_attr = torch.zeros((temporal_edge_index.size(1), edge_attr.size(1))).cuda()
                    else:
                        data.prev_edge_attr = torch.zeros((0, edge_attr.size(1))).cuda()


                    batch.append(data)
                    prev_max_counter = max_counter
            # TRAINING #

            data_batch = Batch.from_data_list(batch)

            ########### Forward ###########

            outputs = mpn_model(data_batch)

            ########### Loss ###########

            loss, precision1, precision0, precision,loss_class1, loss_class0, list_pred_probs = compute_loss_acc(outputs, data_batch, criterion, criterion_no_reduction, mode='train')
            #Fill dictionary with mean probabilities of each class at each step
            nsteps = len(list_pred_probs)
            for s in range(nsteps):
                if sum(sum([data_batch.edge_labels == 0])) == 0:
                    list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                else:
                    list_mean_probs["0"]["step" + str(s)].append(torch.mean(list_pred_probs[s][data_batch.edge_labels == 0]))
                if sum(sum([data_batch.edge_labels == 1])) == 0:
                    list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                else:
                    list_mean_probs["1"]["step" + str(s)].append(torch.mean(list_pred_probs[s][data_batch.edge_labels == 1]))


            train_losses.update(loss.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
            train_losses1.update(loss_class1.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
            train_losses0.update(loss_class0.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])


            train_precision_class1.update(np.sum(np.asarray([item for item in precision1])) / len(precision1),CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'] )
            train_precision_class0.update(np.sum(np.asarray([item for item in precision0])) / len(precision0),CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'] )
            train_precision.update(np.sum(np.asarray([item for item in precision])) / len(precision),CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'] )


            # accuracies.append()
            train_loss_in_history.append(train_losses.avg)
            train_prec1_in_history.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
            train_prec0_in_history.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))
            train_prec_in_history.append(np.sum(np.asarray([item for item in precision])) / len(precision))

            ########### Accuracy ###########

            ########### Optimizer update ###########

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lambda: float(loss))

            train_batch_time.update(time.time() - start_time)

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Train Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Train Acc 1 {acc.val:.3f} (avg: {acc.avg:.3f})\t'
                      'Train Acc 0 Act {acc2.val:.3f} (avg: {acc2.avg:.3f})\t'
                      '{et}<{eta}'.format(epoch, i, len(train_loader), batch_time=train_batch_time,  loss=train_losses,
                                          acc = train_precision_class1, acc2 =train_precision_class0, et=str(datetime.timedelta(seconds=int(train_batch_time.sum))),
                                          eta=str(datetime.timedelta(seconds=int(train_batch_time.avg * (len(train_loader) - i))))))

    plt.figure()
    for i in range(nsteps):
        list_mean_probs_history["0"]["step" + str(i)].append(
            np.mean(torch.stack(list_mean_probs["0"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history["0"]["step" + str(i)], '--', label="Class 0 Iter" + str(i))
        list_mean_probs_history["1"]["step" + str(i)].append(
            np.mean(torch.stack(list_mean_probs["1"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history["1"]["step" + str(i)], '-', label="Class 1 Iter" + str(i))
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Mean Probability per Class per Epoch Training.pdf', bbox_inches='tight')
    plt.close()



    plt.figure()
    plt.plot(train_loss_in_history, label='Loss')

    plt.ylabel('Loss'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Training Loss per Iteration.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(train_prec1_in_history,'g', label='Precision class 1')
    plt.plot(train_prec0_in_history, 'r', label='Precission class 0')

    plt.ylabel('Precision'), plt.xlabel('Iteration')
    plt.legend(loc= 'best')
    plt.savefig(results_path + '/images/Training Precision per Iteration.pdf', bbox_inches='tight')
    plt.close()

    return train_losses, train_losses1, train_losses0, train_precision_class1, train_precision_class0, train_loss_in_history,train_prec1_in_history,train_prec0_in_history,train_prec_in_history,list_mean_probs_history


def validate(CONFIG, val_loader, cnn_model, mpn_model, results_path, epoch, val_loss_in_history, val_prec1_in_history,
             val_prec0_in_history, val_prec_in_history, val_dataset, dataset_dir, list_mean_probs_history_val):


    torch.cuda.empty_cache()

    val_losses = utils.AverageMeter('losses', ':.4e')
    val_losses1 = utils.AverageMeter('losses', ':.4e')
    val_losses0 = utils.AverageMeter('losses', ':.4e')

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    val_precision_1 = utils.AverageMeter('Val prec class 1', ':6.2f')
    val_precision_0 = utils.AverageMeter('Val prec class 0', ':6.2f')
    val_precision = utils.AverageMeter('Val prec', ':6.2f')

    mpn_model.eval()
    cnn_model.eval()

    nsteps = CONFIG['GRAPH_NET_PARAMS']['num_class_steps']
    list_mean_probs = {"0": {}, "1": {}}
    for i in range(nsteps):
        list_mean_probs["0"]["step" + str(i)] = []
        list_mean_probs["1"]["step" + str(i)] = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0:
                start_time = time.time()

                ########### Data extraction ###########
                [bboxes_t, data_df_t, bboxes_prev, data_df_prev, max_dist] = data
                len_graphs = [len(item) for item in data_df_t]


                if len(bboxes_t) > 0:
                    bboxes_t_tensor = torch.cat(bboxes_t, dim=0).cuda() if isinstance(bboxes_t,
                                                                                      list) else bboxes_t.cuda()
                    if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                        node_embeds_t, reid_embeds_t = cnn_model(bboxes_t_tensor)
                    else:
                        node_embeds_t = cnn_model(bboxes_t_tensor)
                        reid_embeds_t = node_embeds_t
                else:
                    reid_embeds_t = torch.tensor([]).cuda()
                    node_embeds_t = torch.tensor([]).cuda()

                if len(bboxes_prev) > 0:
                    bboxes_prev_tensor = torch.cat(bboxes_prev, dim=0).cuda() if isinstance(bboxes_prev,
                                                                                            list) else bboxes_prev.cuda()
                    if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                        node_embeds_prev, reid_embeds_prev = cnn_model(bboxes_prev_tensor)
                    else:
                        node_embeds_prev = cnn_model(bboxes_prev_tensor)
                        reid_embeds_prev = node_embeds_prev
                else:
                    reid_embeds_prev = torch.tensor([]).cuda()

                if CONFIG['CNN_MODEL']['L2norm']:
                    if reid_embeds_t.numel() > 0:
                        reid_embeds_t = F.normalize(reid_embeds_t, p=2, dim=1)
                    if reid_embeds_prev.numel() > 0:
                        reid_embeds_prev = F.normalize(reid_embeds_prev, p=2, dim=1)

                reid_embeds = reid_embeds_t
                node_embeds = node_embeds_t
                data_df = data_df_t

                max_counter = 0
                prev_max_counter = 0


                global_prev_counter = 0


                edge_ixs = []
                node_label = []
                node_id_cam = []
                batch = []
                flag_visualize = False

                for g in range(len(len_graphs)):
                    if flag_visualize:
                        pass
                    else:
                        data_df[g] = data_df[g].assign(
                            node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))

                        max_counter = max(data_df[g]['node'].values) + 1
                        node_embeds_g = torch.stack([node_embeds[idx] for idx in data_df[g]['node']])


                        prev_num_nodes = len(data_df_prev[g]) if len(data_df_prev) > 0 else 0
                        if prev_num_nodes > 0:
                            node_embeds_prev_g = node_embeds_prev[
                                                 global_prev_counter: global_prev_counter + prev_num_nodes]
                            global_prev_counter += prev_num_nodes
                        else:
                            node_embeds_prev_g = torch.tensor([]).cuda()


                        edge_ixs_g = []
                        for id_cam in np.unique(data_df[g]['id_cam']):
                            ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                            ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                            edge_ixs_g.append(
                                torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                        edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                        edge_ixs_g_np = edge_ixs_g.cpu().numpy()



                        edge_ixs_g_np = apply_density_aware_sparsification(
                            edge_ixs_g_np, data_df[g], prev_max_counter,
                            reid_embeds_np=reid_embeds.cpu().numpy(),
                            eta=0.5, k_min=2, k_max=10
                        )
                        edge_ixs_g = torch.from_numpy(edge_ixs_g_np).cuda()


                        node_label_g = torch.from_numpy(data_df[g]['id'].values)

                        emb_dist_g = F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(
                            -1, 1)
                        emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]],
                                                             reid_embeds[edge_ixs_g[1]]).view(-1, 1)

                        xws_1 = np.expand_dims(
                            np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                            axis=1)
                        yws_1 = np.expand_dims(
                            np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                            axis=1)
                        xws_2 = np.expand_dims(
                            np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                            axis=1)
                        yws_2 = np.expand_dims(
                            np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                            axis=1)

                        points1 = np.concatenate((xws_1, yws_1), axis=1)
                        points2 = np.concatenate((xws_2, yws_2), axis=1)

                        spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))),
                                                         dim=1).cuda()
                        spatial_dist_g_norm = torch.from_numpy(spatial_dist_g.cpu().numpy() / max_dist[g]).cuda()
                        spatial_dist_manh_g = torch.unsqueeze(
                            (torch.from_numpy(paired_distances(points1, points2, metric='manhattan'))), dim=1).cuda()
                        spatial_dist_manh_g_norm = torch.from_numpy(
                            spatial_dist_manh_g.cpu().numpy() / max_dist[g]).cuda()

                        if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                            edge_attr = torch.cat((emb_dist_g, emb_dist_g_cos), dim=1)
                        elif CONFIG['TRAINING']['ONLY_DIST']:
                            edge_attr = torch.cat(
                                (spatial_dist_g_norm.type(torch.float32), spatial_dist_manh_g_norm.type(torch.float32)),
                                dim=1)
                        else:
                            edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32),
                                                   spatial_dist_manh_g_norm.type(torch.float32), emb_dist_g,
                                                   emb_dist_g_cos), dim=1)

                        edge_labels_g = torch.from_numpy(
                            np.asarray(
                                [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][j]] ==
                                       data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][j]]) else 0
                                 for j in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                        edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)


                        prev_idx_list, t_idx_list = [], []

                        prev_num_nodes = len(data_df_prev[g]) if len(data_df_prev) > 0 else 0
                        if prev_num_nodes > 0 and len(data_df[g]) > 0:
                            from scipy.spatial.distance import cdist
                            points_t = np.stack((data_df[g]['xw'].values, data_df[g]['yw'].values), axis=-1)
                            points_prev = np.stack((data_df_prev[g]['xw'].values, data_df_prev[g]['yw'].values),
                                                   axis=-1)

                            dist_matrix = cdist(points_prev, points_t, metric='euclidean')
                            temporal_k = min(2, len(points_t))

                            if temporal_k > 0:
                                for i_prev in range(len(points_prev)):
                                    closest_t_indices = np.argsort(dist_matrix[i_prev])[:temporal_k]
                                    for i_t in closest_t_indices:
                                        if dist_matrix[i_prev, i_t] < 0.5:
                                            prev_idx_list.append(i_prev)
                                            t_idx_list.append(i_t)

                        temporal_edge_index = torch.tensor([prev_idx_list, t_idx_list], dtype=torch.long).cuda()


                        data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_label_g, edge_attr=edge_attr,
                                    edge_labels=edge_labels_g)


                        data.prev_x = node_embeds_prev_g


                        data.temporal_edge_index = temporal_edge_index

                        if temporal_edge_index.numel() > 0:
                            data.prev_edge_attr = torch.zeros((temporal_edge_index.size(1), edge_attr.size(1))).cuda()
                        else:
                            data.prev_edge_attr = torch.zeros((0, edge_attr.size(1))).cuda()

                        batch.append(data)
                        prev_max_counter = max_counter


                # TRAINING #

                data_batch = Batch.from_data_list(batch)

                ########### Forward ###########

                outputs = mpn_model(data_batch)


                ########### Loss ###########

                # loss, acc_actives, acc_nonactives = compute_loss_acc(outputs, data_batch, mode = 'validate')
                loss, precision1, precision0, precision,loss_class1, loss_class0,list_pred_probs = compute_loss_acc(outputs, data_batch, criterion ='', criterion_no_reduction='', mode='validate')

                # Fill dictionary with mean probabilities of each class at each step
                nsteps = len(list_pred_probs)

                for s in range(nsteps):
                    if sum(sum([data_batch.edge_labels == 0])) == 0:
                        list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                    else:
                        list_mean_probs["0"]["step" + str(s)].append(
                            torch.mean(list_pred_probs[s][data_batch.edge_labels == 0]))
                    if sum(sum([data_batch.edge_labels == 1])) == 0:
                        list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                    else:
                        list_mean_probs["1"]["step" + str(s)].append(
                            torch.mean(list_pred_probs[s][data_batch.edge_labels == 1]))

                val_losses.update(loss.item(), CONFIG['TRAINING']['BATCH_SIZE']['VAL'])
                val_losses1.update(loss_class1.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
                val_losses0.update(loss_class0.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])


                val_precision_1.update(np.sum(np.asarray([item for item in precision1])) / len(precision1),
                                       CONFIG['TRAINING']['BATCH_SIZE']['VAL'])
                val_precision_0.update(np.sum(np.asarray([item for item in precision0])) / len(precision0),
                                       CONFIG['TRAINING']['BATCH_SIZE']['VAL'])

                val_precision.update(np.sum(np.asarray([item for item in precision])) / len(precision),
                                     CONFIG['TRAINING']['BATCH_SIZE']['VAL'])

                # accuracies.append()
                val_loss_in_history.append(val_losses.avg)
                val_prec1_in_history.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
                val_prec0_in_history.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))
                val_prec_in_history.append(np.sum(np.asarray([item for item in precision])) / len(precision))

                ########### Accuracy ###########

                ########### Optimizer update ###########



                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                          'Val Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                          'Val Precision class 1 {acc.val:.3f} (avg: {acc.avg:.3f})\t'
                          'Val Precision class 0 {acc2.val:.3f} (avg: {acc2.avg:.3f})\t'
                          '{et}<{eta}'.format(i, len(val_loader), batch_time=val_batch_time, loss=val_losses,
                                              acc=val_precision_1, acc2=val_precision_0,
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(
                                                  seconds=int(val_batch_time.avg * (len(val_loader) - i))))))

    plt.figure()
    for i in range(nsteps):
        list_mean_probs_history_val["0"]["step" + str(i)].append(np.mean(torch.stack(list_mean_probs["0"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history_val["0"]["step" + str(i)], '--', label="Class 0 Iter" + str(i))
        list_mean_probs_history_val["1"]["step" + str(i)].append( np.mean(torch.stack(list_mean_probs["1"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history_val["1"]["step" + str(i)], '-', label="Class 1 Iter" + str(i))
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Mean Probability per Class per Epoch Validation.pdf', bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.plot(val_loss_in_history, label='Loss')

    plt.ylabel('Loss'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Validation Loss per Iteration.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(val_prec1_in_history, 'g', label='Precision class 1')
    plt.plot(val_prec0_in_history, 'r', label='Precision class 0')
    plt.ylabel('Precision'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Validation Precision per Iteration.pdf', bbox_inches='tight')
    plt.close()

    return val_losses, val_losses1, val_losses0, val_precision_1, val_precision_0,val_loss_in_history,val_prec1_in_history, val_prec0_in_history,val_prec_in_history,list_mean_probs_history_val




# # CODIGO PINTAS DIST Y DIST NORM (va en el bucle)

    # spatial_dist_g_l = []
    #         spatial_dist_g_l_norm = []
    #         pets_dists = []
    #         pets_dists_norm = []
    #         terrace_dists = []
    #         terrace_dists_norm = []
    #         lab_dists = []
    #         lab_dists_norm = []
    # basket_dists  = []
    #         garden1_dists = []
    #         garden1_dists_norm = []


# spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))), dim=1).cuda()
#                 spatial_dist_g_l.append(spatial_dist_g.cpu().numpy())
#                 spatial_dist_g_norm.append(spatial_dist_g.cpu().numpy() / max_dist[g])
#                 spatial_dist_x = torch.abs(torch.from_numpy(xws_1 - xws_2)).cuda()
#                 spatial_dist_x_norm = spatial_dist_x / max_dist[g]
#                 spatial_dist_y = torch.abs(torch.from_numpy(yws_1 - yws_2)).cuda()
#                 spatial_dist_y_norm = spatial_dist_y / max_dist[g]
# #
# edge_labels_g = torch.from_numpy(
#     np.asarray([1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
#                       data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
#                 for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()
# if max_dist[g] == 26.56:  # PETS
#     pets_dists.append(
#         [n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if edge_labels_g.cpu().numpy()[pos] == 1])
#     pets_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                             edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 50.83:  # Terrace
#     terrace_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                           edge_labels_g.cpu().numpy()[pos] == 1])
#     terrace_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                                edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 44.23:  # Laboratory
#     lab_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                       edge_labels_g.cpu().numpy()[pos] == 1])
#     lab_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                            edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 85.23:  # Garden1 CAMPUS
#     garden1_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                           edge_labels_g.cpu().numpy()[pos] == 1])
#     garden1_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                                edge_labels_g.cpu().numpy()[pos] == 1])



# # CODIGO PINTAR DISTSN desopues de forwards
#  dists = np.concatenate(spatial_dist_g_l)
#             dists_norm = np.concatenate(spatial_dist_g_l_norm)
#
#             pets_dists = np.concatenate(pets_dists)
#             pets_dists_norm = np.concatenate(pets_dists_norm)
#             terrace_dists = np.concatenate(terrace_dists)
#             terrace_dists_norm = np.concatenate(terrace_dists_norm)
#             lab_dists = np.concatenate(lab_dists)
#             lab_dists_norm = np.concatenate(lab_dists_norm)
#             garden1_dists = np.concatenate(garden1_dists)
#             garden1_dists_norm = np.concatenate(garden1_dists_norm)
#
#
#             pets_dists_mean = np.mean(pets_dists)
#             pets_dists_norm_mean = np.mean(pets_dists_norm)
#             terrace_dists_mean = np.mean(terrace_dists)
#             terrace_dists_norm_mean = np.mean(terrace_dists_norm)
#             lab_dists_norm_mean = np.mean(lab_dists_norm)
#             lab_dists_mean = np.mean(lab_dists)
#             garden1_dists_mean = np.mean(garden1_dists)
#             garden1_dists_norm_mean = np.mean(garden1_dists_norm)
#
#
#             plt.figure()
#             plt.subplot(2, 1, 1)
#             plt.scatter(np.arange(len(dists)), dists, c=data_batch.edge_labels.cpu().numpy())
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * terrace_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * pets_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * lab_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * garden1_dists_mean)
#
#
#             plt.title('Distances. Mean(1) terrace = ' + str(int(terrace_dists_mean)) + ' Mean(1) pets = ' + str(
#                 int(pets_dists_mean)) + 'Mean(1) Lab = ' + str(int(lab_dists_mean)) + 'Mean(1) Garden1 = ' + str(int(garden1_dists_mean)) )
#             plt.show(block=False)
#             plt.subplot(2, 1, 2)
#             plt.scatter(np.arange(len(dists_norm)), dists_norm, c=data_batch.edge_labels.cpu().numpy())
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * terrace_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * pets_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * lab_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * garden1_dists_norm_mean)
#
#
#             plt.title(
#                 'Distances in meters. Mean(1) terrace = ' + str((terrace_dists_norm_mean)) + ' Mean(1) pets = ' + str(
#                     pets_dists_norm_mean) + 'Mean(1) Lab = ' + str(lab_dists_norm_mean) + 'Mean(1) Garden1 = ' + str(garden1_dists_norm_mean) )
#
#             plt.show(block=False)


# dists = np.concatenate(spatial_dist_g_l)
# dists_norm = np.concatenate(spatial_dist_g_l_norm)
# basket_dists = np.concatenate(basket_dists)
# basket_dists_norm = np.concatenate(basket_dists_norm)
# basket_dists_mean = np.mean(basket_dists)
# basket_dists_norm_mean = np.mean(basket_dists_norm)
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.scatter(np.arange(len(dists)), dists, c=data_batch.edge_labels.cpu().numpy())
# plt.plot(np.arange(len(dists)), np.ones(len(dists)) * basket_dists_mean)
#
#
# plt.title('Distances. Mean(1) Basketball = ' + str(int(basket_dists_mean))  )
# plt.subplot(2, 1, 2)
# plt.scatter(np.arange(len(dists_norm)), dists_norm, c=data_batch.edge_labels.cpu().numpy())
# plt.plot(np.arange(len(dists)), np.ones(len(dists)) * basket_dists_norm_mean)
#
# plt.title(  'Distances in meters. Mean(1) Basketball = ' + str((basket_dists_norm_mean)) )
#
# plt.show(block=False)