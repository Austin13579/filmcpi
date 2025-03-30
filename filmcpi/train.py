import torch
from utils import Basic_Encoder
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
import pandas as pd
import numpy as np
import random
from model import FilmCPI
import argparse
import copy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# training function at each epoch
def train(model, device, train_loader, optimizer):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    losses = []
    for batch_idx, (compound, protein, label) in enumerate(train_loader):
        compound, protein = compound.to(device), protein.to(device)
        optimizer.zero_grad()

        output = model(compound, protein)
        m = torch.nn.Sigmoid()
        score = torch.squeeze(m(output))
        loss = loss_fn(score, label.float().to(device))

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (compound, protein, label) in enumerate(loader):
            compound, protein = compound.to(device), protein.to(device)
            output = model(compound, protein)
            m = torch.nn.Sigmoid()
            score = torch.squeeze(m(output))
            total_preds = torch.cat((total_preds, score.cpu()), 0)
            total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default='bindingDB', help='which dataset')
    parser.add_argument('--model_type', type=str, default='both', help='which split type')
    parser.add_argument('--split', type=str, default='up', help='which split type')
    parser.add_argument('--rs', type=int, default=0, help='which random seed')
    args = parser.parse_args()

    setup_seed(42)
    batch_size = 256
    LR = 1e-4
    NUM_EPOCHS = 50

    print('Dataset: ' + args.ds + ', Random Seed: ' + str(
        args.rs) + ', Split: ' + args.split + ', Model Type: ' + args.model_type)
    print('Learning Rate: ' + str(LR) + ', Epochs: ' + str(NUM_EPOCHS))

    train_data = pd.read_csv('../datasets/datas/' + args.ds + '_train_' + args.split + str(args.rs) + '.csv')
    valid_data = pd.read_csv('../datasets/datas/' + args.ds + '_valid_' + args.split + str(args.rs) + '.csv')
    test_data = pd.read_csv('../datasets/datas/' + args.ds + '_test_' + args.split + str(args.rs) + '.csv')

    train_set = Basic_Encoder(train_data.index.values, train_data)
    valid_set = Basic_Encoder(valid_data.index.values, valid_data)
    test_set = Basic_Encoder(test_data.index.values, test_data)

    # Build dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Training the model
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = FilmCPI().to(device)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    

    best_valid_roc = -100
    best_result = []
    train_losses = []
    valid_roc_aucs, test_roc_aucs = [], []
    test_pr_aucs = []
    for epoch in range(NUM_EPOCHS):
        print("Epoch: ", epoch + 1)
        train_loss = train(model, device, train_loader, optimizer)
        train_losses.append(train_loss)
        print("Train Loss: ", train_loss)

        print('Validation')
        valid_true, valid_pred = predicting(model, device, valid_loader)
        valid_roc_auc = roc_auc_score(valid_true, valid_pred)
        valid_roc_aucs.append(valid_roc_auc)
        print("Val ROC_AUC: ", valid_roc_auc)

        test_true, test_pred = predicting(model, device, test_loader)
        test_roc_auc = roc_auc_score(test_true, test_pred)
        test_roc_aucs.append(test_roc_auc)
        test_pr_auc = average_precision_score(test_true, test_pred)
        test_pr_aucs.append(test_pr_auc)
        print("Test ROC_AUC: ", test_roc_auc)
        print("Test PR_AUC: ", test_pr_auc)
        print()

    dd = {"Loss": train_losses, "Valid_roc": valid_roc_aucs, "Test_roc": test_roc_aucs, "Test_pr": test_pr_aucs}
    df = pd.DataFrame(dd)
    df.to_csv('results/' + args.ds + '_' + args.split + str(args.rs) + '_' + args.model_type + '.csv')
