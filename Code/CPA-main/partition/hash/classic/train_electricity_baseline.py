import numpy as np
import pandas as pd
import argparse
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='train bank classifiers')
#     parser.add_argument('--num_partition', default=20, type=int)
#     parser.add_argument('--portion', default=0.05, type=float)
#     parser.add_argument('--model', default='random', choices=['bayes', 'logistic', 'svm', 'random'])
#     args = parser.parse_args()
#     return args

args = {
    'dataset': 'electricity',
    'num_partition': 20,
    'portion': 0.05,
    'model': 'svm'
}

def main(args):
    # load data
    train_filename = 'data/electricity/train.csv'
    train_df = pd.read_csv(train_filename)
    test_filename = 'data/electricity/test.csv'
    test_df = pd.read_csv(test_filename)
    partition_name = f'''partitions/electricity/hash_portion{args['portion']}_partition{args['num_partition']}.npy'''
    idxgroup = np.load(partition_name, allow_pickle=True)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    # print(df.tail())

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

    for i in range(args['num_partition']):
        X = X_total.loc[idxgroup[i]]
        y = y_total.loc[idxgroup[i]]

        if args['model'] == 'logistic':
                logreg = LogisticRegression().fit(X, y)
                ypred_logreg = logreg.predict(X_test)
                print('Logistic Regression\n', classification_report(y_test, ypred_logreg, digits=4))
                models.append(logreg)
                y_preds.append(ypred_logreg)

        if args['model'] == 'svm':
            svc = SVC(probability=True).fit(X, y)
            ypred_svc = svc.predict(X_test)
            print('Support Vector Machine\n', classification_report(y_test, ypred_svc, digits=4))
            models.append(svc)
            y_preds.append(ypred_svc)

        if args['model'] == 'bayes':
            gauss = GaussianNB().fit(X, y)
            ypred_gauss = gauss.predict(X_test)
            print('Naive Bayes\'s Classifier\n', classification_report(y_test, ypred_gauss, digits=4))
            models.append(gauss)
            y_preds.append(ypred_gauss)

        if args['model'] == 'random':
            selection = np.random.choice(3)
            if selection == 0:
                logreg = LogisticRegression().fit(X, y)
                ypred_logreg = logreg.predict(X_test)
                print('Logistic Regression\n', classification_report(y_test, ypred_logreg, digits=4))
                models.append(logreg)
                y_preds.append(ypred_logreg)
            elif selection == 1:
                svc = SVC(probability=True).fit(X, y)
                ypred_svc = svc.predict(X_test)
                print('Support Vector Machine\n', classification_report(y_test, ypred_svc, digits=4))
                models.append(svc)
                y_preds.append(ypred_svc)
            else:
                gauss = GaussianNB().fit(X, y)
                ypred_gauss = gauss.predict(X_test)
                print('Naive Bayes\'s Classifier\n', classification_report(y_test, ypred_gauss, digits=4))
                models.append(gauss)
                y_preds.append(ypred_gauss)
    
    preds = np.array(y_preds[1:])
    preds_onehot = np.eye(2)[preds]
    pred_votes = np.sum(preds_onehot, axis=0)
    pred_labels = np.argmax(pred_votes, axis=1)
    print(f"Ensemble acc: {sum(labels==pred_labels)/10000:.4f}")

    checkpoint_filename = f'''checkpoints/electricity/model_{args['model']}_partition_{args['num_partition']}_portion_{args['portion']}.pkl'''
    evaluation_filename = f'''evaluations/electricity/model_{args['model']}_partition_{args['num_partition']}_portion_{args['portion']}.pkl'''
    with open(checkpoint_filename, 'wb') as file:
        pickle.dump(models, file)
    with open(evaluation_filename, 'wb') as file:
        pickle.dump(y_preds, file)

if __name__ == '__main__':
    main(args)

