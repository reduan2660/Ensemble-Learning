import numpy as np
from statsmodels.stats.proportion import proportion_confint
import torch
import math


def array_split_bysize(a, size, axis):
    idx_split = np.arange(size, a.shape[axis], size)
    return np.split(a, idx_split, axis=axis)

def prepare_param(mode, preds, labels, num_poison, num_classes, overlap=0, scale=None):
    """
    Generate the convex optimization describing the collective certificate.
    Returns radius and pareto-point-reachability parameters for efficient repeated evaluation.
    Parameters
    ----------
    preds: torch.tensor_like (G, N)
        G is the num of sub-classifiers; N is the size of testset
    label: torch.tensor_like (N)
        True labels of the testset
    num_classes: int
        Number of classes
    Returns
    -------
    """
    numvotes = np.zeros((len(labels), num_classes))
    for preds_subclassifer in preds:
        numvotes += np.eye(num_classes)[preds_subclassifer]
    idxsort = np.argsort(-numvotes, axis=1, kind='stable')
    valsort = -np.sort(-numvotes, axis=1, kind='stable')
    rob = np.floor(
        ((valsort[:, 0]-valsort[:, 1] - (idxsort[:, 1] < idxsort[:, 0]))/(2*(overlap+1))))
    indices_attackable = (rob < num_poison)
    if mode == 'rob':
        # only need to consider the correct preds:
        indices_consider = indices_attackable
    elif mode == 'ca':
        # only need to consider the correct & attackable preds:
        indices_correct = (idxsort[:, 0] == labels)
        indices_consider = indices_attackable*indices_correct

    preds_consider = preds[:, indices_consider]
    # labels_consider = labels[indices_consider].squeeze()
    labels_consider = idxsort[:, 0][indices_consider].squeeze()
    numvotes_consider = numvotes[indices_consider, :]
    list_params = []
    if scale is not None:
        list_sub_preds_consider = array_split_bysize(
            preds_consider, scale, axis=1)
        list_sub_labels_consider = array_split_bysize(
            labels_consider, scale, axis=0)
        list_sub_numvotes_consider = array_split_bysize(
            numvotes_consider, scale, axis=0)
        for sub_preds_consider, sub_labels_consider, sub_numvotes_consider in zip(list_sub_preds_consider, list_sub_labels_consider, list_sub_numvotes_consider):
            # print(len(sub_labels_consider))
            list_params.append((len(sub_labels_consider), sub_numvotes_consider,
                                sub_labels_consider.squeeze(), sub_preds_consider))
    else:
        # print(len(labels_consider))
        list_params.append(
            (len(labels_consider), numvotes_consider, labels_consider, preds_consider))
    return list_params


def pA_minus_pB(counts, alpha):
    N = np.sum(counts)
    NA = counts[0]
    pAlower = proportion_confint(
        NA, N, alpha=alpha*2., method="beta")[0]
    return pAlower-(1-pAlower)


def rob_budget(num_trainset, num_poison, num_select):
    ra = 0
    for m in range(num_trainset-num_poison, num_trainset+num_poison+1):
        ra_m = (m/num_trainset)**num_select - \
            ((max(m, num_trainset)-num_poison)/num_trainset)**num_select*2+1
        if ra_m > ra:
            ra = ra_m
    return ra


def DPA_certify(mode, preds, labels, num_poison, num_classes, overlap):
    num_data = preds.shape[1]
    numvotes = np.zeros((num_data, num_classes))
    for preds_subclassifer in preds:
        numvotes += np.eye(num_classes)[preds_subclassifer]
    idxsort = np.argsort(-numvotes, axis=1, kind='stable')
    valsort = -np.sort(-numvotes, axis=1, kind='stable')
    rob = np.floor(
        ((valsort[:, 0]-valsort[:, 1] - (idxsort[:, 1] < idxsort[:, 0]))/(2*(overlap+1))))
    is_rob = (rob >= num_poison)

    if mode == 'ca':
        is_acc = (idxsort[:, 0] == labels)
        is_ca = is_acc*is_rob
        # return is_ca.sum().item()/num_data
        return is_ca.sum().item()
    else:
        # return is_rob.sum().item()/num_data
        return is_rob.sum().item()


def BAG_certify(mode, preds, labels, num_select, num_trainset, num_poison, num_classes, alpha=0.001):
    # num_select: the sub-trainset size
    # N: the trainset size
    num_data = preds.shape[1]
    numvotes = np.zeros((num_data, num_classes))
    for preds_subclassifer in preds:
        numvotes += np.eye(num_classes)[preds_subclassifer]
    idxsort = np.argsort(-numvotes, axis=1, kind='stable')
    valsort = -np.sort(-numvotes, axis=1, kind='stable')
    delta = np.zeros([num_data])
    for idx in range(num_data):
        numvotes = valsort[idx]
        delta[idx] = pA_minus_pB(numvotes, alpha/num_data)
    is_rob = (delta >= rob_budget(num_trainset, num_poison, num_select))
    if mode == 'ca':
        is_acc = (idxsort[:, 0] == labels)
        is_ca = is_acc*is_rob
        return is_ca.sum().item()/num_data
    else:
        return is_rob.sum().item()/num_data
