""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides helpful functions for other python files in the repository.
# Update paths to processed datasets


import pandas as pd
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pytorchtools import EarlyStopping, EarlyStopping_v2

def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
    # print("Distribution of classes: \n", get_class_distribution(natural_img_dataset))
    return count_dict


def dummy_data(BATCH_SIZE, train_compose, test_compose):
    print("Loading CIFAR10 as dummy data.")
    train_set = datasets.CIFAR10('data', train=True, transform=train_compose, download=True)
    test_set = datasets.CIFAR10('data', train=False, transform=test_compose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def convert_batch_list(lst_of_lst):
    return sum(lst_of_lst, [])

# rows to be the “true class” and the columns to be the “predicted class.”
def make_cm(targets_list, predictions_list, classes):
    # labels = [‘True Neg’,’False Pos’,’False Neg’,’True Pos’]
    cm = confusion_matrix(targets_list, predictions_list)
    confusion_matrix_df = pd.DataFrame(cm, columns=classes, index=classes)
    fig = plt.figure(figsize=(12, 10))
    fig = sns.heatmap(confusion_matrix_df, annot=True, fmt= "d", cmap= "coolwarm")
    fig.set(ylabel = "True", xlabel="Predicted", title='DKL predictions' )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    return cm

def make_pretty_cm(cf, group_names=None, categories='auto', count=True,
                   percent=True, cbar=True, xyticks=True, xyplotlabels=True, sum_stats=True,
                   figsize=None,cmap='Blues', title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]
    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))
        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

def train_cnn(model, criterion, optimizer, train_loader, epoch):
    model.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        try:
            output, _, _ = model(data)
        except:
            output, _ = model(data)

        _, preds = torch.max(output, 1)
        loss = criterion(output, target)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        correct += preds.eq(target.view_as(preds)).cpu().sum()
    train_accuracy = (100. * correct / float(len(train_loader.dataset))).cpu().detach().numpy()
    train_loss = np.average(train_losses)
    # print("Training accuracy: {0:.3f} %\n".format(train_accuracy))
    return train_accuracy, train_loss

def val_cnn(model, early_stopping, timestamp, test_loader, epoch, path, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        stop = False
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output, _ = model(data)
            _, preds = torch.max(output, 1)
            # Convert to probabilities if output is logsoftmax
            #  ps = torch.exp(log_ps)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
            # Calculate accuracy
            # equals = pred == targets
            # accuracy = torch.mean(equals)
            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(test_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model, epoch, timestamp, path)
        if early_stopping.early_stop:
            # stop_flag_count += 1
            if early_stopping.counter >= early_stopping.patience:
                stop = True
        print("Validation accuracy: {0:.3f} %\n".format(val_accuracy))
        return val_accuracy, valid_loss, stop

def epoch_test_cnn(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            try:
                output, _ = model(data)
            except:
                output = model(data)

            _, preds = torch.max(output, 1)

            loss = criterion(output, target)

            try:
                valid_losses.append(loss.item())
            except:
                valid_losses.append(loss)

            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        return val_accuracy, valid_loss

def predict_cnn(data_loader, model):
    model.eval()
    y_pred, y_true, probs, feature = [], [], [], []
    for data, target in data_loader:
    # for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output, ftr = model(data)
        un, preds = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)
        probs.append(list(probabilities.detach().cpu().numpy()))
        y_pred.append(list(preds.cpu().numpy()))
        y_true.append(list(target.cpu().numpy()))
        feature.append(list(ftr.detach().cpu().numpy()))
    return y_pred, y_true, probs, feature

def extract_features(DenseNetModel, dataloader):
    f = []
    feature = DenseNetModel.features
    # features = torch.nn.Sequential(*list(DenseNetModel.children())[:-1])
    for data, target in dataloader:
        # for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        out = feature(data)
        out = F.relu(out, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1) # only works for inputs of 32 x 32
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1)).view(len(data), -1)
        f.append(list(out.detach().cpu().numpy()))
    return f

class custom_classifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.2):
        super(custom_classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # fully connected layer 1
        x = self.dropout(x)
        feat = F.relu(self.fc2(x)) # fully connected layer 2
        x = self.dropout(x)
        x = self.fc3(feat)   #fully connected layer 3
        return x, feat


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            # at = self.alpha.gather(0, target.data.view(-1))
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def visualize_imgs(dataloader, artifact):
    data_iter = iter(dataloader)
    MEAN = torch.tensor([0.5, 0.5, 0.5])
    STD = torch.tensor([0.25, 0.25, 0.25])
    images, labels = next(data_iter)
    plt.clf()
    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    for idx in np.arange(12):
        ax = fig.add_subplot(3, 4, idx+1)
        unnorm_img = images[idx] * STD[:, None, None] + MEAN[:, None, None]
        unnorm_img = unnorm_img.numpy().transpose(1, 2, 0)
        ax.imshow(unnorm_img, cmap='gray')
        # plt.imshow(np.transpose(images[idx], (1,2,0)).astype('uint8'))
        ax.set_title(labels[idx].numpy())
        ax.set_axis_off()

    plt.savefig(f"RGB to Gray Scale for {artifact}.png")