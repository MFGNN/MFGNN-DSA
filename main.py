
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.autograd import Variable
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score


from utils import multiomics_data
from model import MultiDeep
import torch.utils.data as Dataset

from datetime import datetime

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--batch', type=int, default=256, help='Number of batch size')
parser.add_argument('--hidden_GAT', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--hidden_MHSA', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--nb_heads_GAT', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--nb_heads_MHSA', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=5, help='Patience')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda:2")
    torch.cuda.manual_seed(args.seed)
    used_memory = torch.cuda.memory_allocated()  # 已使用的GPU内存量
    cached_memory = torch.cuda.memory_reserved()  # 缓存的GPU内存量
    # print(f"GPU success，服务器GPU已分配：{used_memory / 1024 ** 3:.2f} GB，已缓存：{cached_memory / 1024 ** 3:.2f} GB")
else:
    device = torch.device("cpu")

# Load data
se_HIN, se_SE, se_adj, drug_HIN, drug_PC, drug_adj, sample_set = multiomics_data()
se_HIN, se_SE, se_adj = Variable(se_HIN), Variable(se_SE), Variable(se_adj)
drug_HIN, drug_PC, drug_adj = Variable(drug_HIN), Variable(drug_PC), Variable(drug_adj)
se_HIN, se_SE, se_adj = se_HIN.to(device), se_SE.to(device), se_adj.to(device)
drug_HIN, drug_PC, drug_adj = drug_HIN.to(device), drug_PC.to(device), drug_adj.to(device)

used_memory = torch.cuda.memory_allocated()  # 已使用的GPU内存量
cached_memory = torch.cuda.memory_reserved()   #缓存的GPU内存量
# print(f"数据上传成功，服务器GPU已分配：{used_memory / 1024**3:.2f} GB，已缓存：{cached_memory / 1024**3:.2f} GB")

# Model and optimizer
model = MultiDeep(nse=se_SE.shape[0],
                  ndrug=drug_PC.shape[0],
                  nseHIN=se_HIN.shape[1],
                  nseSE=se_SE.shape[1],
                  ndrugHIN=drug_HIN.shape[1],
                  ndrugPC=drug_PC.shape[1],
                  nhid_GAT=args.hidden_GAT,
                  nheads_GAT=args.nb_heads_GAT,
                  nhid_MHSA=args.hidden_MHSA,
                  nheads_MHSA=args.nb_heads_MHSA,
                  alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss_func = nn.MSELoss()
loss_func.to(device)
used_memory = torch.cuda.memory_allocated()  # 已使用的GPU内存量
cached_memory = torch.cuda.memory_reserved()   #缓存的GPU内存量
# print(f"loss函数上传成功，服务器GPU已分配：{used_memory / 1024**3:.2f} GB，已缓存：{cached_memory / 1024**3:.2f} GB")
best_value = [0, 0, 1]
model_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def train(epoch, index_tra, y_tra, index_val, y_val):

    tra_dataset = Dataset.TensorDataset(index_tra, y_tra)
    train_dataset = Dataset.DataLoader(tra_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    model.train()

    for index_trian, y_train in train_dataset:
        y_train = y_train.to(device)
        y_tpred = model(se_HIN, se_SE, se_adj, drug_HIN, drug_PC, drug_adj, index_trian.numpy().astype(int), device)

        loss_train = loss_func(y_tpred, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    pred_train = y_tpred.cpu().detach().numpy()
    true_train = y_train.cpu().detach().numpy()

    RMSE_train = np.sqrt(loss_train.item(), out=None)
    MAE_train = mean_absolute_error(true_train, pred_train)
    PCC_train = pearsonr(true_train, pred_train)[0]
    R2_train = r2_score(true_train, pred_train)
    AUC_train = roc_auc_score(true_train, pred_train)
    AUPR_train = average_precision_score(true_train, pred_train)

    model.eval()
    val_dataset = Dataset.TensorDataset(index_val, y_val)
    valid_dataset = Dataset.DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    pred_valid, true_valid = [], []
    for index_valid, y_valid in valid_dataset:
        y_valid = y_valid.to(device)
        y_vpred = model(se_HIN, se_SE, se_adj, drug_HIN, drug_PC, drug_adj, index_valid.numpy().astype(int), device)
        loss_valid = loss_func(y_vpred, y_valid)
        pred_valid.extend(y_vpred.cpu().detach().numpy())
        true_valid.extend(y_valid.cpu().detach().numpy())


    RMSE_valid = np.sqrt(loss_valid.item(), out=None)
    MAE_valid = mean_absolute_error(true_valid, pred_valid)
    PCC_valid = pearsonr(true_valid, pred_valid)[0]
    R2_valid = r2_score(true_valid, pred_valid)
    AUC_valid = roc_auc_score(true_valid ,pred_valid)
    AUPR_valid = average_precision_score(true_valid,pred_valid)


    print('Epoch: {:04d}'.format(epoch + 1),
          '\n loss_train: {:.4f}'.format(loss_train.item()),
          'RMSE_train: {:.4f}'.format(RMSE_train),
          'MAE_train: {:.4f}'.format(MAE_train),
          'PCC_train: {:.4f}'.format(PCC_train),
          'R2_train: {:.4f}'.format(R2_train),
          'AUC_train: {:.4f}'.format(AUC_train),
          'AUPR_train: {:.4f}'.format(AUPR_train),
          '\n loss_valid: {:.4f}'.format(loss_valid),
          'RMSE_valid: {:.4f}'.format(RMSE_valid),
          'MAE_valid: {:.4f}'.format(MAE_valid),
          'PCC_valid: {:.4f}'.format(PCC_valid),
          'R2_valid: {:.4f}'.format(R2_valid),
          'AUC_valid: {:.4f}'.format(AUC_valid),
          'AUPR_valid: {:.4f}'.format(AUPR_valid ))

    if AUC_valid >= best_value[0] and AUPR_valid >= best_value[1]:
        best_value[0] = AUC_valid
        best_value[1] = AUPR_valid
        best_value[2] = epoch + 1
        torch.save(model.state_dict(), "./output/models_{}.pkl".format(model_date))
    return best_value[2], AUC_valid


def compute_test(index_test, y_test):
    model.eval()
    pred_test, true_test = [], []
    dataset = Dataset.TensorDataset(index_test, y_test)
    test_dataset = Dataset.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    test_results = []  # Create an empty list to store all test results
    for index_test, y_test in test_dataset:
        y_test = y_test.to(device)
        y_pred = model(se_HIN, se_SE, se_adj, drug_HIN, drug_PC, drug_adj, index_test.numpy().astype(int), device)
        loss_test = loss_func(y_pred, y_test)
        pred_test.extend(y_pred.cpu().detach().numpy())
        true_test.extend(y_test.cpu().detach().numpy())

        test_results.extend(list(zip(index_test.numpy(), y_test.cpu().detach().numpy(),
                                     y_pred.cpu().detach().numpy())))  # Add this batch's results to the list# Now test_results contains the results for the entire test set
    # for idx, true_val, pred_val in test_results:
    #     print(f"Index: {idx}, True Value: {true_val},    Predicted Value: {pred_val}")
    # Save the test results to a CSV file
    test_results_df = pd.DataFrame(test_results, columns=['Index', 'True Value', 'Predicted Value'])
    test_results_df.to_csv('5-FU.csv', index=False)

    RMSE_test = np.sqrt(loss_test.item(), out=None)
    MAE_test = mean_absolute_error(true_test, pred_test)
    PCC_test = pearsonr(true_test, pred_test)[0]
    R2_test = r2_score(true_test, pred_test)
    AUC_test = roc_auc_score(true_test,pred_test)
    AUPR_test = average_precision_score(true_test,pred_test)
    pred_test_binary = [1 if x >= 0.5 else 0 for x in pred_test]
    # Accuracy and Precision
    Accuracy_test = (np.array(pred_test_binary) == np.array(true_test)).mean()
    Precision_test = precision_score(true_test, pred_test_binary)  # Precision added here
    Recall_test = recall_score(true_test, pred_test_binary)
    F1_score_test = f1_score(true_test, pred_test_binary)

    with open("HINGNN-DPI", 'a') as f:
        f.write(str(model_date) + " " + str(RMSE_test) + " " + str(MAE_test) + " " + str(PCC_test) + " " + str(R2_test)
                + " " + str(AUC_test) + " " + str(AUPR_test)
                + " " + str( Precision_test) + " " + str(Recall_test) + " " + str(F1_score_test) + " "
                + str(Accuracy_test) + "\n")  # Accuracy and Precision are now logged

    print("Test set results:",
          "\n loss_test: {:.4f}".format(loss_test),
          "RMSE_test: {:.4f}".format(RMSE_test),
          'MAE_test: {:.4f}'.format(MAE_test),
          "PCC_test: {:.4f}".format(PCC_test),
          "R2_test: {:.4f}".format(R2_test),
          "AUC_test: {:.4f}".format(AUC_test),
          "AUPR_test: {:.4f}".format(AUPR_test),
          "Precision_test: {:.4f}".format(Precision_test),  # Printing Precision
          "Recall_test: {:.4f}".format(Recall_test),
          "F1_score_test: {:.4f}".format(F1_score_test),
          "Accuracy_test: {:.4f}".format(Accuracy_test)  # Printing Accuracy
          )


# Train model
time_begin = time.time()

train_set, test_set = train_test_split(np.arange(sample_set.shape[0]), test_size=0.1, random_state=np.random.randint(0, 1000), stratify=sample_set[:, 2])
train_set, valid_set = train_test_split(train_set, test_size=1 / 9, random_state=np.random.randint(0, 1000), stratify=sample_set[train_set, 2])

index_train, y_train = sample_set[train_set[:], :2], sample_set[train_set[:], 2]
index_valid, y_valid = sample_set[valid_set[:], :2], sample_set[valid_set[:], 2]
index_test, y_test = sample_set[test_set[:], :2], sample_set[test_set[:], 2]
y_train, y_test, y_valid = Variable(y_train, requires_grad=True), Variable(y_test, requires_grad=True), Variable(y_valid, requires_grad=True)

model.to(device)
auc_valid = [0]
bad_counter = 0
for epoch in range(args.epochs):

    best_epoch, avg_auc_valid = train(epoch, index_train, y_train, index_valid, y_valid)
    auc_valid.append(avg_auc_valid)

    if abs(auc_valid[-1] - auc_valid[-2]) < 0.0005:
        bad_counter += 1
    else:
        bad_counter = 0

    if bad_counter >= args.patience:
        break

model.load_state_dict(torch.load('./output/models_{}.pkl'.format(model_date)))

# Testing
compute_test(index_test, y_test)
time_total = time.time() - time_begin
print("Total time: {:.4f}s".format(time_total))
