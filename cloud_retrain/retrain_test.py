import torch
import torch.nn as nn
import torch.optim as optim
import psycopg2
import numpy as np
from retrain_dataset import retrain_dataset
from CustomDataset_aug import CustomDataset
from model import CNN_LSTM, CBiLSTM, Transformer, Transformer_rev_Complex, GRUClassifier, CRNN
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import json
import os
import pandas as pd
import time  # 引入time模組

# Database connection parameters
db_host = "100.68.188.72"
db_name = "gesture"
db_user = "t1204"
db_password = "t1204"
num = 9

# 連接到資料庫
conn = psycopg2.connect(
    host=db_host,
    dbname=db_name,
    user=db_user,
    password=db_password
)
cur = conn.cursor()
table_name = "gesture_1_test"
query = f"SELECT * FROM {table_name} WHERE username = 'default';"
cur.execute(query)
# 使用fetchone()逐行讀取資料
data = cur.fetchall()
# 關閉連接
cur.close()
conn.close()
data_np = np.array([list(map(float, row[0:99])) for row in data])  # 手勢資料
labels_np = np.array([int(row[-3]) for row in data])             # 標籤
# load data
train_dataset = retrain_dataset(window_size=30, data=data_np, labels=labels_np)

# check the hyperparameters.json exist or not
# if not, create the alarm json to alarm no parameter file.
if not os.path.exists('hyperparameters.json'):
    hyperparameters = {"window_size": 30, "batch_size": 10, "learning_rate": 0.001, "epoch": 1,
                       "loss": "CrossEntropyLoss", "optimizer": "adam", "neuralNetwork": "Transformer",
                       "model": "gesture_Federated.pth"}
    with open("hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f)
else:
    with open('hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)

"""
{"learningRate": 0.001, 
"epoch": 10, 
"batchSize": 1,  // 將批次大小設置為1以減慢訓練速度
"lossFunction": "CrossEntropyLoss", 
"neuralNetwork": "Transformer", 
"optimizer": "adam",
 "model": "gesture_Federated.pth"}
"""

# get the hyperparameters
batch_size = hyperparameters["batchSize"]
print("batch_size:", batch_size)
learning_rate = hyperparameters["learningRate"]
epoch = hyperparameters["epoch"]
loss = hyperparameters["lossFunction"]
optimizer_name = hyperparameters["optimizer"]
model_name = hyperparameters["neuralNetwork"]
trained_model = hyperparameters["model"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model_name == "Transformer":
    model = Transformer_rev_Complex(99, num, 32, 2).to(device)
elif model_name == "CNN_LSTM":
    model = CNN_LSTM(99, num, 64, 32).to(device)
elif model_name == "CBiLSTM":
    model = CBiLSTM(99, 64, 32).to(device)
elif model_name == "GRUClassifier":
    model = GRUClassifier(99, 64, 32).to(device)
elif model_name == "CRNN":
    model = CRNN(99, 64, 32).to(device)
if loss == "CrossEntropyLoss":
    loss_func = nn.CrossEntropyLoss()
elif loss == "NLLLoss":
    loss_func = nn.NLLLoss()
if optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 加載預訓練模型
# model = Transformer(99, 64, 32, 2)
model.load_state_dict(torch.load(f"./model/{trained_model}", map_location=device))

#
# # 修改模型（如果需要）
# # 例如，更換分類層以適應新的任務
# # model.fc = nn.Linear(...)
#
# # 損失函數和優化器
# loss_func = nn.CrossEntropyLoss()  # 根據任務選擇適當的損失函數
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 數據加載器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#
# # 訓練模型
model.train()
Train_total_loss = 0
Train_correct_predictions = 0
all_labels = []
all_preds = []


# 在訓練開始前初始化 JSON 文件
def initialize_progress_file():
    initial_progress = {"epoch": 0, "batch": 0, "loss": 0, "progress": 0, "accuracy": 0}
    with open("progress.json", "w") as f:
        json.dump(initial_progress, f)


# 更新 JSON 文件中的進度
def update_progress_file(epoch, batch, loss, progress, accuracy):
    progress_dict = {"epoch": epoch, "batch": batch, "loss": loss, "progress": progress, "accuracy": accuracy}
    with open("progress.json", "w") as f:
        json.dump(progress_dict, f)


def initialize_metric_file():
    initial_metric = {"epoch": 0, "avg_train_loss": 0, "Train_accuracy": 0, "accuracy": 0,
                      "precision": 0, "recall": 0, "f1": 0, "confusion": []}
    with open("metric_process.json", "w") as f:
        json.dump(initial_metric, f)


initialize_progress_file()
initialize_metric_file()


def update_metric_file(epoch, avg_train_loss, Train_accuracy, accuracy, precision, recall, f1):
    metric_dict = {"epoch": epoch, "avg_train_loss": avg_train_loss, "Train_accuracy": Train_accuracy,
                   "accuracies": accuracy, "precisions": precision, "recall": recall, "f1": f1,
                   }
    with open("metric_process.json", "w") as f:
        json.dump(metric_dict, f)


def test(model_name, test_loader, device, loss_func):
    model_name.eval()
    Test_total_loss = 0
    Test_correct_predictions = 0
    true_labels = []
    predicted_labels = []
    predicted_probabilities = []

    with torch.no_grad():
        for x_test, y_test in tqdm(test_loader, desc='Testing'):
            x_test = x_test.to(device)
            y_test = y_test.to(device).long()
            y_predict = model_name(x_test)
            probabilities = torch.softmax(y_predict, dim=1)
            loss = loss_func(y_predict, y_test.squeeze(dim=-1))
            Test_total_loss += loss.item()
            _, indices = torch.max(y_predict.data, dim=1, keepdim=True)
            Test_correct_predictions += (indices == y_test).sum().item()
            true_labels.append(y_test.cpu())
            predicted_labels.append(indices.cpu())
            predicted_probabilities.append(probabilities.cpu())

    true_labels = torch.cat(true_labels, dim=0).numpy()
    predicted_labels = torch.cat(predicted_labels, dim=0).numpy()
    predicted_probabilities = torch.cat(predicted_probabilities, dim=0).numpy()
    avg_test_loss = Test_total_loss / len(test_loader)
    Test_accuracy = 100. * Test_correct_predictions / len(test_loader.dataset)

    print(
        f'Test set: Average loss: {avg_test_loss:.4f}, Accuracy: {Test_accuracy:.2f}%, Test loss: {Test_total_loss:.4f}')

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    confusion = confusion_matrix(true_labels, predicted_labels)

    # save confusion matrix to csv
    confusion_matrix_path = 'result/confusion_matrix.csv'
    df_cm = pd.DataFrame(confusion, index=[i for i in range(len(confusion))],
                         columns=[i for i in range(len(confusion))])
    # Add FN in the end of the table
    df_cm['FN'] = df_cm.sum(axis=1) - np.diag(df_cm)
    df_cm.to_csv(confusion_matrix_path, index=False)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    # print('Confusion Matrix:', confusion)

    result = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }

    # Calculate Precision-Recall and ROC curves for each class
    precision_vals = []
    recall_vals = []
    fpr_vals = []
    tpr_vals = []
    roc_aucs = []

    for i in range(predicted_probabilities.shape[1]):
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels == i, predicted_probabilities[:, i])
        fpr, tpr, _ = roc_curve(true_labels == i, predicted_probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        # Resampling to ensure consistent lengths for averaging
        precision_interp = np.interp(np.linspace(0, 1, 100), recall_curve[::-1], precision_curve[::-1])
        recall_interp = np.linspace(0, 1, 100)
        fpr_interp = np.interp(np.linspace(0, 1, 100), fpr, fpr)
        tpr_interp = np.interp(np.linspace(0, 1, 100), fpr, tpr)

        precision_vals.append(precision_interp)
        recall_vals.append(recall_interp)
        fpr_vals.append(fpr_interp)
        tpr_vals.append(tpr_interp)
        roc_aucs.append(roc_auc)

    # Calculate average precision-recall and ROC curves
    mean_precision = np.mean(precision_vals, axis=0)
    mean_recall = np.mean(recall_vals, axis=0)
    mean_fpr = np.mean(fpr_vals, axis=0)
    mean_tpr = np.mean(tpr_vals, axis=0)
    mean_roc_auc = np.mean(roc_aucs)

    # Save Precision-Recall curve data to CSV
    df_pr = pd.DataFrame({
        'Recall': mean_recall,
        'Precision': mean_precision
    })
    df_pr.to_csv("./precision_recall_curve.csv", index=False)

    # Save ROC curve data to CSV
    df_roc = pd.DataFrame({
        'FPR': mean_fpr,
        'TPR': mean_tpr,
        'AUC': [mean_roc_auc] * len(mean_fpr)
    })
    df_roc.to_csv("./roc_curve.csv", index=False)

    # Save the result to JSON
    with open('test_results.json', 'w') as f:
        result_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
        json.dump(result_to_save, f)

    return result


# 訓練模型
# 建立測試資料集
test_dataset = CustomDataset("test_dataset/test_gesture_1.csv", 30, augment=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
for epoch in range(epoch):
    Train_total_loss = 0.0
    Train_correct_predictions = 0
    all_labels = []
    all_preds = []

    with tqdm(train_loader, unit="batch", disable=False) as tepoch:
        for batch_idx, (x_train, y_train) in enumerate(tepoch):
            x_train = x_train.to(device)
            y_train = y_train.to(device).long()
            y_predict = model(x_train)

            loss = loss_func(y_predict, y_train.squeeze(dim=-1))
            Train_total_loss += loss.item()

            values, indices = torch.max(y_predict.data, dim=1, keepdim=True)
            Train_correct_predictions += (indices == y_train).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probabilities = torch.softmax(y_predict, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)
            all_labels.append(y_train.squeeze(dim=-1).cpu().numpy())
            all_preds.append(predicted_labels.cpu().numpy())
            tepoch.set_description(f"Epoch: {epoch}")
            tepoch.update(1)

            progress_percentage = (tepoch.n / tepoch.total) * 100
            update_progress_file(epoch, batch_idx, loss.item(),
                                 progress_percentage,
                                 Train_correct_predictions / ((batch_idx + 1) * len(x_train)))

            time.sleep(0.1)  # 添加0.1秒的延遲

    avg_train_loss = Train_total_loss / len(train_loader.dataset)
    Train_accuracy = 100. * Train_correct_predictions / len(train_loader.dataset)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    accuracies = accuracy_score(all_labels, all_preds)
    precisions = precision_score(all_labels, all_preds, labels=np.arange(num), average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, labels=np.arange(num), average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, labels=np.arange(num), average='macro', zero_division=0)
    confusion = confusion_matrix(all_labels, all_preds, labels=np.arange(num))

    update_metric_file(epoch, avg_train_loss, Train_accuracy, accuracies, precisions, recall, f1)

    os.makedirs('result', exist_ok=True)
    results = {
        'loss': [Train_total_loss],
        'accuracy': [Train_accuracy],
        'precision': [precisions],
        'recall': [recall],
        'f1_score': [f1]
    }

    # 建立DataFrame並儲存到CSV
    df = pd.DataFrame(results)
    df.to_csv('result/training_results.csv', index=False)

    # 儲存混淆矩陣
    # confusion_matrix_path = 'result/confusion_matrix.csv'
    # train_confusion_matrix = confusion_matrix(all_labels, all_preds, labels=np.arange(7))
    # df_cm = pd.DataFrame(train_confusion_matrix, index=[i for i in range(len(train_confusion_matrix))],
    #                      columns=[i for i in range(len(train_confusion_matrix))])
    # # Add FN in the end of the table
    # df_cm['FN'] = df_cm.sum(axis=1) - np.diag(df_cm)
    # df_cm.to_csv(confusion_matrix_path, index=False)
# print('Train Epoch: {} Average loss: {:.6f}, Accuracy: {:.2f}%'
#       .format(epoch, avg_train_loss, Train_accuracy))
# print('roc_auc: {:.2f}, accuracy: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(roc_auc, accuracies,
#                                                                                                 precisions, recall,
#                                                                                                 f1))
# write to json
model.eval()
results = test(model, test_loader, device, loss_func)

# overall_precision = precision_score(np.concatenate(true_labels), np.concatenate(predicted_labels), average='macro', task='multiclass', threshold=0.5,
#                               num_classes=7).item()
# overall_recall = recall(np.concatenate(true_labels), np.concatenate(predicted_labels), average='macro', task='multiclass', threshold=0.5,
#                         num_classes=7).item()
# overall_f1 = f1_score(true_labels, predicted_labels, average='macro', task='multiclass', threshold=0.5,
#                       num_classes=7).item()
# overall_specificity = specificity(true_labels, predicted_labels, average='macro', task='multiclass', threshold=0.5,
#                                   num_classes=7).item()

# print('Finished Training')
#
# search the largest index of model in folder
model_list = os.listdir('model')
model_index = []

for i in model_list:
    try:
        index = int(i.split('.')[0].split('_')[1])
        model_index.append(index)
    except ValueError:
        continue

if model_index:
    model_index = max(model_index)
else:
    model_index = 0  # 设置默认值为0

# print(model_index)
# save the model
# torch.save(model.state_dict(), f"./model/model_{trained_model}.pth".format(model_index+1))

# save the model by the model name with index
torch.save(model.state_dict(), f"./model/model_{model_index + 1}_{model_name}.pth")
# # 保存新訓練的模型
# torch.save(model.state_dict(), "model_1.pth")