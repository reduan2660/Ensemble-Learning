import numpy as np
import pandas as pd
import argparse

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Partition data')
#     parser.add_argument('--dataset', default='bank', choices=['bank', 'electricity'])
#     parser.add_argument('--partitions', default=20, type=int)
#     parser.add_argument('--portion', default=0.05, type=float)
#     args = parser.parse_args()
#     return args

args = {
    'dataset': 'electricity',
    'partitions': 20,
    'portion': 0.05
}

# print(args)

def main(args):
    filename = f'data/{args["dataset"]}/train.csv'
    df = pd.read_csv(filename)

    # print(df.sample())
    overlap = int(np.ceil(args['partitions']/int(1/args['portion']))) - 1

    # use hash to partition data
    pre_hash_data = df.values[:,:-1]
    idxgroup_final = []
    for time in range(overlap+1): # update hash value
        hash_data = [hash(str(data)+str(time)) % int(1/args['portion']) for data in pre_hash_data]
        if time != overlap:
            idxgroup = [np.nonzero((hash_data == np.array(i)))[0] for i in range(int(1/args['portion']))]
        else:
            idxgroup = [np.nonzero((hash_data == np.array(i)))[0] for i in range(args['partitions'] - overlap*int(1/args['portion']))]
        idxgroup_final += idxgroup

    # check array length
    for i in range(args['partitions']):
        idxgroup_final[i] = idxgroup_final[i][:600] # Partition Length Same
        print(len(idxgroup_final[i]))

    save_name = f'''./partitions/{args['dataset']}/hash_portion{args['portion']}_partition{args['partitions']}.npy'''

    # print(idxgroup_final)
    np.save(save_name, idxgroup_final, allow_pickle=True)

if __name__ == '__main__':
    main(args)