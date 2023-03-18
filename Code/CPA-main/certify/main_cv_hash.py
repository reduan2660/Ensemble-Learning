from utils import prepare_param, DPA_certify, BAG_certify
import numpy as np
import torch
import gurobipy as gp
import argparse
import csv
import time
import math
from scipy.io import savemat
import os

def data_cb(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

        # Did objective value or best bound change?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._data.append([time.time() - model._start, cur_obj, cur_bd])


def gurobi_search(num_consider, num_classifiers, num_classes, num_poison, portion, numvotes, label, preds, time_limit, verbose):
    overlap = int(np.ceil(num_classifiers/(1/portion)))-1
    m = gp.Model("attack")

    X = m.addVars(num_classifiers, vtype=gp.GRB.BINARY, name="X")
    Y = m.addVars(num_consider, vtype=gp.GRB.BINARY, name="Y")
    Z = m.addVars(num_consider, num_classes-1, vtype=gp.GRB.BINARY, name="Z")

    m.update()

    m.setObjective(gp.quicksum(Y[i]
                   for i in range(num_consider)), gp.GRB.MAXIMIZE)

    for i in range(overlap+1):
        if i != overlap:
            m.addConstr(gp.quicksum(X[j] for j in range(i*int(1/portion), (i+1)*int(1/portion))) <= num_poison)
        else:
            m.addConstr(gp.quicksum(X[j] for j in range(overlap*int(1/portion),num_classifiers)) <= num_poison)

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
    parser.add_argument('--num_poison', default=None,
                        type=int, help='poison budget.')
    parser.add_argument('--scale', default=None,
                        type=int, help='the maximum scale of the programming problem.')
    parser.add_argument('--t_persample', default=2.,
                        type=float, help="Gurobi's solution time limit.")
    parser.add_argument('--num_classes', default=10,
                        type=int, help='number of classes.')
    parser.add_argument('--portion', default=0.005,
                        type=float, help='data for every partition')
    parser.add_argument('--out', type=str, default='/home/crx/collective/out')
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    out_dir = os.path.join(args.out, args.dataset+'partition'+str(args.num_partition),
                           args.mode)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    overlap = int(np.ceil(args.num_partition/int(1/args.portion)))-1
    filein = torch.load(
        f'/home/crx/collective/evaluations/{args.dataset}_nin_baseline_partitions_{args.num_partition}_portion_{args.portion}.pth', map_location='cpu')
    labels = filein['labels']
    scores = filein['scores']
    num_classes = args.num_classes
    preds = scores.max(2).indices.T

    preds = preds.cpu().numpy() # [num_classifier, num_instance]
    labels = labels.cpu().numpy() # [num_instance]

    # if mode == 'ca', we only consider the correct predictions
    # if mode == 'rob', we consider all the predictions
    if args.mode == 'ca':
        numvotes = np.zeros((preds.shape[1], num_classes)) # [num_instance, num_class]
        for preds_subclassifer in preds:
            numvotes += np.eye(num_classes)[preds_subclassifer]
        idxsort = np.argsort(-numvotes, axis=1, kind='stable')
        num_total = (idxsort[:, 0] == labels).sum().item()  # correct
    elif args.mode == 'rob':
        num_total = preds.shape[1]
    
    if args.num_poison is not None:
        num_poison = args.num_poison
        out_file = os.path.join(
        out_dir, f'poison{num_poison}_scale{args.scale}_t_persample{args.t_persample}.mat')

        scale = args.scale if args.scale is not None else 50+5*args.num_poison
        print(f"================num_poison: {num_poison}==================")
        num_unattacked_sam = DPA_certify(mode=args.mode,
                                         preds=preds, labels=labels, num_poison=num_poison, num_classes=args.num_classes, overlap=overlap)
        list_params = prepare_param(mode=args.mode, preds=preds, 
                                    labels=labels, num_poison=num_poison, num_classes=args.num_classes, overlap=args.overlap, scale=scale)

        num_unattacked_col = num_total
        for i, (num_consider, new_numvotes, new_label, new_preds) in enumerate(list_params[:]):
            obj, upper_bound, A = gurobi_search(num_consider=num_consider, num_classifiers=args.num_partition, num_classes=args.num_classes, num_poison=num_poison,
                                                portion=args.portion, numvotes=new_numvotes, label=new_label, preds=new_preds, time_limit=scale*args.t_persample, verbose=False)
            num_attacked = math.floor(upper_bound)
            num_unattacked_col -= num_attacked
            print(num_unattacked_col)

        print(f"samplewise: {num_unattacked_sam}, collective: {num_unattacked_col}")

        savemat(out_file, {'num_poison': num_poison,
                'samplewise': num_unattacked_sam, 'collective': num_unattacked_col,
                })
    else:
        start = (args.num_partition/2) // 10
        end = args.num_partition/2
        step = (args.num_partition/2) // 10
        out_file = os.path.join(
            out_dir, f'poison{start}to{end}_scale{args.scale}_t_persample{args.t_persample}.mat')
        x = np.arange(start, end+1, step)
        y1 = []
        y2 = []

        # x = [3, 5, 8, 10, 13, 15, 18, 20, 23, 25]
        out_file = os.path.join(
            out_dir, f'poison3to25_scale{args.scale}_t_persample{args.t_persample}.mat')

        for num_poison in x:
            scale = args.scale if args.scale is not None else 50+5*args.num_poison
            print(f"================num_poison: {num_poison}==================")
            num_unattacked_sam = DPA_certify(mode=args.mode,
                                             preds=preds, labels=labels, num_poison=num_poison, num_classes=args.num_classes, overlap=overlap)
            y1.append(num_unattacked_sam)
            list_params = prepare_param(mode=args.mode, preds=preds, 
                                    labels=labels, num_poison=num_poison, num_classes=args.num_classes, overlap=overlap, scale=scale)

            num_unattacked_col = num_total
            for i, (num_consider, new_numvotes, new_label, new_preds) in enumerate(list_params[:]):
                obj, upper_bound, A = gurobi_search(num_consider=num_consider, num_classifiers=args.num_partition, num_classes=args.num_classes, num_poison=num_poison,
                                                    portion=args.portion, numvotes=new_numvotes, label=new_label, preds=new_preds, time_limit=scale*args.t_persample, verbose=True)
                num_attacked = math.floor(upper_bound)
                num_unattacked_col -= num_attacked

            y2.append(num_unattacked_col)
            print(f"samplewise: {num_unattacked_sam}, collective: {num_unattacked_col}")

        savemat(out_file, {
                'num_poison': x,'samplewise': y1, 'collective': y2,
                })


if __name__ == '__main__':
    main(parse_arguments())
