from copyreg import pickle
from utils import prepare_param, DPA_certify, BAG_certify, array_split_bysize
import numpy as np
import torch
import gurobipy as gp
import argparse
import csv
import time
import math
from scipy.io import savemat
import os
import pickle

def data_cb(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

        # Did objective value or best bound change?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._data.append([time.time() - model._start, cur_obj, cur_bd])


def gurobi_search(num_consider, num_classifiers, num_trainsamples, num_classes, num_poison, train_dict, numvotes, label, preds, time_limit, verbose):
    m = gp.Model("attack")

    X = m.addVars(num_classifiers, vtype=gp.GRB.BINARY, name="X")
    Y = m.addVars(num_consider, vtype=gp.GRB.BINARY, name="Y")
    Z = m.addVars(num_consider, num_classes-1, vtype=gp.GRB.BINARY, name="Z")
    W = m.addVars(num_trainsamples, vtype=gp.GRB.BINARY, name='W')

    m.update()

    m.setObjective(gp.quicksum(Y[i]
                   for i in range(num_consider)), gp.GRB.MAXIMIZE)

    m.addConstr(gp.quicksum(W[j] for j in range(num_trainsamples)) <= num_poison)

    for i in range(num_classifiers):
        m.addConstr(X[i] <= gp.quicksum(W[j] if i in train_dict[j] else 0 for j in range(num_trainsamples)))

    new_true_votes = []
    new_votes = [[] for _ in range(num_consider)]

    for i in range(num_consider):
        for j in range(num_classes):
            if j == label[i]:
                new_true_votes.append(
                    numvotes[i][j] - gp.quicksum(X[l] if preds[l][i] == j else 0 for l in range(num_classifiers)))
            else:
                new_votes[i].append(numvotes[i][j] + gp.quicksum(X[l]
                                    if preds[l][i] != j else 0 for l in range(num_classifiers)))

    for i in range(num_consider):
        for j in range(num_classes-1):
            if j < label[i]:
                m.addConstr(
                    (Z[i, j]-0.5)*(0.5-(new_votes[i][j] - new_true_votes[i] + 1)) <= 0)
            else:
                m.addConstr(
                    (Z[i, j]-0.5)*(0.5-(new_votes[i][j] - new_true_votes[i])) <= 0)

    for i in range(num_consider):
        m.addConstr((Y[i]-0.5)*(0.5-gp.quicksum(Z[i, j]
                    for j in range(num_classes-1))) <= 0)

    m.Params.LogToConsole = verbose
    m.Params.TimeLimit = time_limit
    # m.Params.MIPFocus = 3
    m._obj = None
    m._bd = None
    m._data = []
    m._start = time.time()
    m.optimize(callback=data_cb)
    obj = m._data[-1][1]
    upper_bound = m._data[-1][2]
    A = []
    for k, v in m.getAttr('X', X).items():
        if v == 1:
            A.append(k)
    return obj, upper_bound, A


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Certify Max Num of Poisoned Predictions.")
    parser.add_argument('mode', type=str, default='rob', choices=[
                        'rob', 'ca'], help='To certify robustness or to evaluate certified accuracy')
    parser.add_argument('dataset', default='cifar',
                        choices=['cifar', 'fashion_mnist'], help="Test dataset.")
    parser.add_argument('num_partition', type=int,
                        help='Test number of partitions.')
    parser.add_argument('--num_poison', default=1,
                        type=int, help='poison budget.')
    parser.add_argument('--scale', default=10000,
                        type=int, help='the maximum scale of the programming problem.')
    parser.add_argument('--t_persample', default=2.,
                        type=float, help="Gurobi's solution time limit.")
    parser.add_argument('--num_classes', default=10,
                        type=int, help='number of classes.')
    parser.add_argument('--verbose', action='store_true', help='print gurobi logs or not')
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    TRAIN_SIZE = 50000 if args.dataset == 'cifar' else 60000
    train_dict_file = open(f"train_dict/{args.dataset}/bag_partition_{args.num_partition}.pkl", 'rb')
    train_dict = pickle.load(train_dict_file)
    # print(train_dict)
    filein = torch.load(
        f'/home/crx/collective/evaluations/{args.dataset}/{args.dataset}_bag_baseline_partitions_{args.num_partition}.pth', map_location='cpu')
    labels = filein['labels']
    scores = filein['scores']
    num_classes = args.num_classes
    preds = scores.max(2).indices.T

    preds = preds.cpu().numpy() # [num_classifier, num_instance]
    labels = labels.cpu().numpy() # [num_instance]
    
    numvotes = np.zeros((preds.shape[1], num_classes)) # [num_instance, num_class]
    for preds_subclassifer in preds:
        numvotes += np.eye(num_classes)[preds_subclassifer]
    idxsort = np.argsort(-numvotes, axis=1, kind='stable')

    # if mode == 'ca', we only consider the correct predictions
    # if mode == 'rob', we consider all the predictions
    if args.mode == 'ca':
        indices_consider = idxsort[:, 0] == labels
        num_total = indices_consider.sum().item()  # correct
        labels_consider = idxsort[:, 0][indices_consider].squeeze()
        numvotes_consider = numvotes[indices_consider, :]
        preds_consider = preds[:, indices_consider]
    elif args.mode == 'rob':
        num_total = preds.shape[1]
        labels_consider = idxsort[:, 0].squeeze()
        numvotes_consider = numvotes
        preds_consider = preds
    
    # split by scale
    list_params = []
    scale = args.scale
    list_sub_preds_consider = array_split_bysize(
            preds_consider, scale, axis=1)
    list_sub_labels_consider = array_split_bysize(
        labels_consider, scale, axis=0)
    list_sub_numvotes_consider = array_split_bysize(
        numvotes_consider, scale, axis=0)
    for sub_preds_consider, sub_labels_consider, sub_numvotes_consider in zip(list_sub_preds_consider, list_sub_labels_consider, list_sub_numvotes_consider):
        if scale != 1:
            list_params.append((len(sub_labels_consider), sub_numvotes_consider,
                                sub_labels_consider.squeeze(), sub_preds_consider))
        else:
            list_params.append((len(sub_labels_consider), sub_numvotes_consider,
                                sub_labels_consider, sub_preds_consider))
    
    num_poison = args.num_poison

    print(f"================num_poison: {num_poison}==================")
    num_unattacked_col = num_total
    for i, (num_consider, new_numvotes, new_label, new_preds) in enumerate(list_params[:]):
        obj, upper_bound, A = gurobi_search(num_consider=num_consider, num_classifiers=args.num_partition, num_trainsamples=TRAIN_SIZE, num_classes=args.num_classes, 
                                            num_poison=num_poison, train_dict = train_dict, numvotes=new_numvotes, label=new_label, preds=new_preds,
                                            time_limit=args.t_persample*scale, verbose=args.verbose)
        num_attacked = math.floor(upper_bound)
        num_unattacked_col -= num_attacked
    print(f"num_consider:{num_total}, objective: {num_unattacked_col}, gap: {(num_total-math.floor(num_unattacked_col))/num_total:.4f}")

    # obj, upper_bound, A = gurobi_search(num_consider=num_total, num_classifiers=args.num_partition, num_trainsamples=TRAIN_SIZE, num_classes=args.num_classes, 
    #                                 num_poison=num_poison, train_dict=train_dict, numvotes=numvotes_consider, label=labels_consider, preds=preds_consider, 
    #                                 time_limit=20000, verbose=args.verbose)
    # print(f"num_consider:{num_total}, objective: {obj}, gap: {(num_total-math.floor(upper_bound))/num_total:.4f}")

if __name__ == '__main__':
    main(parse_arguments())
