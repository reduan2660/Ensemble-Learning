import numpy as np
import pandas as pd
import argparse
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def parse_arguments():
    parser = argparse.ArgumentParser(description='train bank classifiers')
    parser.add_argument('--num_partition', default=20, type=int)
    parser.add_argument('--portion', default=0.05, type=float)
    parser.add_argument('--model', default='random', choices=['random', 'bayes', 'logistic', 'svm'])
    args = parser.parse_args()
    return args

def main(args):
    # load data
    # data_filename = "./data/bank/bank-full"
    # rng = RandomState()
    train_filename = './data/bank/train.csv'
    train_df = pd.read_csv(train_filename)
    # train_df = df.sample(frac=0.7, random_state=rng)
    test_filename = './data/bank/test.csv'
    test_df = pd.read_csv(test_filename)
    # test_df = df.loc[~df.index.isin(train_df.index)]
    partition_name = f'partitions/bank/hash_portion{args.portion}_partition{args.num_partition}.npy'
    idxgroup = np.load(partition_name, allow_pickle=True)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    # print(df.tail())

    # preprocessing data
    i = df[(df['poutcome'] == 'unknown') & (df['pdays'] != -1)].index
    df['poutcome'].iloc[i] = 'failure'
    df['poutcome'].replace({'other':'failure'}, inplace=True)
    i = df[df['contact'] == 'unknown'].index
    df['contact'].iloc[i] = np.random.choice(a=['cellular', 'telephone'], size=len(i), p=(.774, .226))

    scaler = StandardScaler()
    num_df = df.select_dtypes(include=np.number)
    num_df_arr = scaler.fit_transform(num_df)
    num_df = pd.DataFrame(num_df_arr, columns=num_df.columns)

    # apply ordinal label to ordinal categorical data
    cat_df = df.select_dtypes(include='object')

    binary_dict = {'yes':1, 'no':0}
    calendar_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
                    'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    education_dict = {'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3}

    cat_df['deposit'].replace(binary_dict, inplace=True)
    cat_df['default'].replace(binary_dict, inplace=True)
    cat_df['housing'].replace(binary_dict, inplace=True)
    cat_df['loan'].replace(binary_dict, inplace=True)
    cat_df['month'].replace(calendar_dict, inplace=True)
    cat_df['education'].replace(education_dict, inplace=True)

    df2 = pd.concat([cat_df, num_df], axis=1)

    # apply one hot encoding to the rest of the categorical features
    df2 = pd.get_dummies(df2, drop_first=True)

    # preparing and splitting training and test data
    X_total = df2.drop(columns='deposit')
    y_total = df2['deposit']
    X_test = df2[-10000:].drop(columns='deposit')
    y_test = df2[-10000:]['deposit']
    labels = y_test.values

    models = []
    y_preds = [labels]

    for i in range(args.num_partition):
        X = X_total.loc[idxgroup[i]]
        y = y_total.loc[idxgroup[i]]

        if args.model == 'logistic':
            logreg = LogisticRegression().fit(X, y)
            ypred_logreg = logreg.predict(X_test)
            print('Logistic Regression\n', classification_report(y_test, ypred_logreg, digits=4))
            models.append(logreg)
            y_preds.append(ypred_logreg)

        if args.model == 'svm':
            svc = SVC(probability=True).fit(X, y)
            ypred_svc = svc.predict(X_test)
            print('Support Vector Machine\n', classification_report(y_test, ypred_svc, digits=4))
            models.append(svc)
            y_preds.append(ypred_svc)

        if args.model == 'bayes':
            gauss = GaussianNB().fit(X, y)
            ypred_gauss = gauss.predict(X_test)
            print('Naive Bayes\'s Classifier\n', classification_report(y_test, ypred_gauss, digits=4))
            models.append(gauss)
            y_preds.append(ypred_gauss)

        if args.model == 'random':
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

    checkpoint_filename = f'checkpoints/bank/model_{args.model}_partition_{args.num_partition}_portion_{args.portion}.pkl'
    evaluation_filename = f'evaluations/bank/model_{args.model}_partition_{args.num_partition}_portion_{args.portion}.pkl'
    with open(checkpoint_filename, 'wb') as file:
        pickle.dump(models, file)
    with open(evaluation_filename, 'wb') as file:
        pickle.dump(y_preds, file)

if __name__ == '__main__':
    main(parse_arguments())


