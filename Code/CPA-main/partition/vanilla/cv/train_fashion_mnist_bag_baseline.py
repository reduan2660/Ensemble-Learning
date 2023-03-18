import sys
sys.path.append('./FeatureLearningRotNet/architectures')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from NetworkInNetwork import NetworkInNetwork
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import random
import pickle

NUM_TRAIN_SAMPLES = 60000
channels = 1

parser = argparse.ArgumentParser(description='Fashion MNIST Training')
parser.add_argument('--num_partitions', default=50, type=int, help='number of partitions')
parser.add_argument('--num_classifiers', default=50, type=int, help='number of classifiers')

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dirbase = 'fashion_mnist_bag_baseline'
checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_subdir = f'./{checkpoint_dir}/' + dirbase + f'_partitions_{args.num_partitions}'
if not os.path.exists(checkpoint_subdir):
    os.makedirs(checkpoint_subdir)
print("==> Checkpoint directory", checkpoint_subdir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_samples = int(NUM_TRAIN_SAMPLES / args.num_partitions)
data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
imgs, labels = zip(*data)
finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
train_dict = {i:[] for i in range(NUM_TRAIN_SAMPLES)}

for i in range(args.num_classifiers):
    print('\Classifier: %d' % i)
    seed = i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    curr_lr = 0.1

    idxgroup = np.random.choice(NUM_TRAIN_SAMPLES, num_samples, False)
    for j in idxgroup:  
        train_dict[j].append(i)
    part_indices = torch.tensor(idxgroup)
    mean = finalimgs[part_indices.unsqueeze(1)].permute(2,0,1,3,4).reshape(channels,-1).mean(dim=1).numpy()
    std = finalimgs[part_indices.unsqueeze(1)].permute(2,0,1,3,4).reshape(channels,-1).std(dim=1).numpy()

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    nomtestloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)
    print('here')
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,part_indices), batch_size=128, shuffle=True, num_workers=1)
    net  = NetworkInNetwork({'num_classes':10, 'num_inchannels': 1})
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0005, nesterov= True)

    # Training
    net.train()
    for epoch in range(200):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch in [60,120,160]):
            curr_lr = curr_lr * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
    
    net.eval()
    (inputs, targets)  = next(iter(nomtestloader)) #Just use one test batch
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
    acc = 100.*correct/total
    print('Accuracy: '+ str(acc)+'%') 
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'partition': i,
        'norm_mean' : mean,
        'norm_std' : std,
    }
    torch.save(state, checkpoint_subdir + '/partition_'+ str(i)+'.pth')

train_dict_file = f'train_dict/fashion_mnist/bag_partition_{args.num_partitions}.pkl'
with open(train_dict_file, 'wb') as file:
    pickle.dump(train_dict, file)


