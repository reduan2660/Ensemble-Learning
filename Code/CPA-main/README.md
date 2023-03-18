Official implementation for ICML22: **On Collective Robustness of Bagging Against Data Poisoning**.

For the environment, We use python3.7, PyTorch, and torchvision, and assume CUDA capabilities. In addition, we need **gurobipy** since we formulate the certification process as a BILP problem.

The code mainly contains two directories: `certify` (in which we run gurobi to calculate the collective robustness and certification accuracy) and `partition` (in which we pretrain models and get predictions). 

We implement the certification for Vanilla and Hash Bagging (as mentioned in the paper) separately. Specifically, we provide an example for running our code. Assume we want to calculate the collective robustness for 50 classifiers (each with 2% of training samples) using Hash Bagging on Cifar-10:

- First, we create the partition file for training:
    ```shell
    python ./partition/hash/cv/partition_data_norm_hash.py --dataset cifar --portion 0.02 --partitions 50
    ```
    This will create `partitions_hash_mean_cifar_50_0.02.pth` under `partition/hash/cv`.
- Then, we train the subclassifiers and make predictions on the test set:
    ```shell
    python ./partition/hash/cv/train_cifar_nin_baseline.py --num_partitions 50 --start_partition 0 --num_partitions_range 50 --portion 0.02
    python ./partition/hash/cv/evaluate_cifar_nin_baseline.py --models cifar_nin_baseline_partitions_50_portion_0.02
    ```
    This will create `cifar_nin_baseline_partitions_50_portion_0.02.pth` under `partition/hash/cv/evaluations`.
- We move the evaluation file under `./certify/evaluations` (create the directory beforehand), then we can run the certification:
    ```shell
    python ./certify/main_cv_hash.py rob cifar 50 --portion 0.02 --num_poison xx --scale xx 
    ```
    This will give the collective robustness we want based on Gurobi.