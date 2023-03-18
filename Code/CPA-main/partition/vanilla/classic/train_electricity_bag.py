import numpy as np
import pandas as pd
import argparse
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

TRAIN_SIZE = 10000

def parse_arguments():
    parser = argparse.ArgumentParser(description='train bank classifiers')
    parser.add_argument('--num_partition', default=20, type=int)
    parser.add_argument('--portion', default=0.05, type=float)
    args = parser.parse_args()
    return args

def main(args):
    # load data
    train_filename = 'data/electricity/train.csv'
    train_df = pd.read_csv(train_filename)
    test_filename = 'data/electricity/test.csv'
    test_df = pd.read_csv(test_filename)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    idxgroup = np.random.choice(TRAIN_SIZE, size=(args.num_partition, int(TRAIN_SIZE*args.portion)))
    train_dict = {i:[] for i in range(TRAIN_SIZE)}
    
    for i in range(TRAIN_SIZE):
        for j in range(args.num_partition):
            if i in idxgroup[j]:
                train_dict[i].append(j)
    
    # preprocessing data
    scaler = StandardScaler()
    num_df = df.drop(columns='day').select_dtypes(include=np.number)
    num_df_arr = scaler.fit_transform(num_df)
    num_df = pd.DataFrame(num_df_arr, columns=num_df.columns)

    # apply ordinal label to ordinal categorical data
    cat_df = df.select_dtypes(include='object')
    binary_dict = {'UP':0, 'DOWN':1}
    cat_df['class'].replace(binary_dict, inplace=True)

    df2 = pd.concat([cat_df, num_df, df['day']], axis=1)

    # preparing and splitting training and test data
    X_total = df2.drop(columns='class')
    y_total = df2['class']
    X_test = df2[-10000:].drop(columns='class')
    y_test = df2[-10000:]['class']
    labels = y_test.values

    models = []
    y_preds = [labels]

    for i in range(args.num_partition):
        X = X_total.loc[idxgroup[i]]
        y = y_total.loc[idxgroup[i]]

        svc = SVC(probability=True).fit(X, y)
        ypred_svc = svc.predict(X_test)
        print('Support Vector Machine\n', classification_report(y_test, ypred_svc, digits=4))
        models.append(svc)
        y_preds.append(ypred_svc)
    
    preds = np.array(y_preds[1:])
    preds_onehot = np.eye(2)[preds]
    pred_votes = np.sum(preds_onehot, axis=0)
    pred_labels = np.argmax(pred_votes, axis=1)
    print(f"Ensemble acc: {sum(labels==pred_labels)/10000:.4f}")

    checkpoint_filename = f'checkpoints/electricity/bag_model_svm_partition_{args.num_partition}_portion_{args.portion}.pkl'
    evaluation_filename = f'evaluations/electricity/bag_model_svm_partition_{args.num_partition}_portion_{args.portion}.pkl'
    with open(checkpoint_filename, 'wb') as file:
        pickle.dump(models, file)
    with open(evaluation_filename, 'wb') as file:
        pickle.dump(y_preds, file)
        pickle.dump(train_dict, file)

if __name__ == '__main__':
    main(parse_arguments())




