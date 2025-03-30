import pandas as pd
import numpy as np
import argparse
from sklearn.utils import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='biosnap', help='which dataset')
parser.add_argument('--split', type=str, default='up', help='which split type')
parser.add_argument('--rs', type=int, default=0, help='which random seed')
args = parser.parse_args()

split_type = args.split
dataset = args.ds
seed = args.rs
np.random.seed(seed)
print("Dataset: " + dataset + ", split type: " + split_type + ", random seed: " + str(seed))
df = pd.read_csv(dataset + '.csv')

# Define the split ratios for each protein type (proportional to counts)
train_ratio = 0.7
valid_ratio = 0.1
test_ratio = 0.2

train_set = pd.DataFrame(columns=df.columns)
valid_set = pd.DataFrame(columns=df.columns)
test_set = pd.DataFrame(columns=df.columns)
train_size, valid_size = int(train_ratio * len(df)), int(valid_ratio * len(df))
selected_columns = ['SMILES', 'Protein', 'Y']

# up denotes unseen protein; ud denotes unseen drug
if split_type == 'up':
    unique_proteins = df['Protein'].unique()
    np.random.shuffle(unique_proteins)
    protein_counts = df['Protein'].value_counts()

    train_protein, valid_protein, test_protein = [], [], []
    tmp_train, tmp_valid = 0, 0
    for pp in unique_proteins:
        if tmp_train < train_size:
            train_protein.append(pp)
            tmp_train += protein_counts[pp]
        elif tmp_train >= train_size and tmp_valid < valid_size:
            valid_protein.append(pp)
            tmp_valid += protein_counts[pp]
        else:
            test_protein.append(pp)
    train_set = df[(df['Protein'].isin(train_protein))].reset_index()
    valid_set = df[(df['Protein'].isin(valid_protein))].reset_index()
    test_set = df[(df['Protein'].isin(test_protein))].reset_index()

    train_set[selected_columns].to_csv('datas/' + dataset + '_train_' + split_type + str(seed) + '.csv', index=False)
    valid_set[selected_columns].to_csv('datas/' + dataset + '_valid_' + split_type + str(seed) + '.csv', index=False)
    test_set[selected_columns].to_csv('datas/' + dataset + '_test_' + split_type + str(seed) + '.csv', index=False)

elif split_type == 'ud':
    unique_drugs = df['SMILES'].unique()
    np.random.shuffle(unique_drugs)
    drug_counts = df['SMILES'].value_counts()

    train_drug, valid_drug, test_drug = [], [], []
    tmp_train, tmp_valid = 0, 0
    for dd in unique_drugs:
        if tmp_train < train_size:
            train_drug.append(dd)
            tmp_train += drug_counts[dd]
        elif tmp_train >= train_size and tmp_valid < valid_size:
            valid_drug.append(dd)
            tmp_valid += drug_counts[dd]
        else:
            test_drug.append(dd)
    train_set = df[(df['SMILES'].isin(train_drug))].reset_index()
    valid_set = df[(df['SMILES'].isin(valid_drug))].reset_index()
    test_set = df[(df['SMILES'].isin(test_drug))].reset_index()

    train_set[selected_columns].to_csv('datas/' + dataset + '_train_' + split_type + str(seed) + '.csv', index=False)
    valid_set[selected_columns].to_csv('datas/' + dataset + '_valid_' + split_type + str(seed) + '.csv', index=False)
    test_set[selected_columns].to_csv('datas/' + dataset + '_test_' + split_type + str(seed) + '.csv', index=False)

elif split_type == 'random':
    df = shuffle(df)
    train_set = df.iloc[:train_size].reset_index()
    valid_set = df.iloc[train_size:train_size + valid_size].reset_index()
    test_set = df.iloc[train_size + valid_size:].reset_index()

    train_set[selected_columns].to_csv('datas/' + dataset + '_train_' + split_type + str(seed) + '.csv', index=False)
    valid_set[selected_columns].to_csv('datas/' + dataset + '_valid_' + split_type + str(seed) + '.csv', index=False)
    test_set[selected_columns].to_csv('datas/' + dataset + '_test_' + split_type + str(seed) + '.csv', index=False)
