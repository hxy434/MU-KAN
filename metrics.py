import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def iou_score_s(output, target):
    """Independent IoU calculation function"""
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def dice_coef_s(output, target):
    """Independent Dice coefficient calculation function"""
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = output_.sum() + target_.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice

def recall_s(output, target):
    """Independent Recall calculation function"""
    smooth = 1e-5

    # Compress output to range [0, 1] using sigmoid function
    output = torch.sigmoid(output)

    # Flatten output and target to 1D arrays, add detach() to separate gradients
    output = output.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    # Calculate number of true positives
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))

    # Calculate number of actual positives
    actual_positives = np.sum(np.round(np.clip(target, 0, 1)))

    # Calculate recall
    recall = true_positives / (actual_positives + smooth)

    return recall

def precision_s(output, target):
    """Independent Precision calculation function"""
    smooth = 1e-5

    # Compress output to range [0, 1] using sigmoid function
    output = torch.sigmoid(output)

    # Flatten output and target to 1D arrays, add detach() to separate gradients
    output = output.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    # Calculate number of true positives
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))

    # Calculate number of predicted positives
    predicted_positives = np.sum(np.round(np.clip(output, 0, 1)))

    # Calculate precision
    precision = true_positives / (predicted_positives + smooth)

    return precision

def accuracy_s(output, target):
    """Independent Accuracy calculation function"""
    # Compress output to range [0, 1] using sigmoid function and round to 0 or 1
    predicted = torch.round(torch.sigmoid(output))

    # Convert predictions and targets to numpy arrays on CPU, add detach() to separate gradients
    predicted = predicted.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    # Calculate accuracy
    accuracy = np.mean(predicted == target)

    return accuracy

def jaccard_coef(output, target):
    """Independent Jaccard coefficient calculation function"""
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)

def specificity_s(output, target):
    """Independent Specificity calculation function"""
    smooth = 1e-5

    # Compress output to range [0, 1] using sigmoid function and round to 0 or 1
    predicted = torch.round(torch.sigmoid(output))

    # Convert predictions and targets to numpy arrays on CPU, add detach() to separate gradients
    predicted = predicted.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    # Calculate number of true negatives
    true_negatives = np.sum(np.round(np.clip((1 - predicted) * (1 - target), 0, 1)))

    # Calculate number of actual negatives
    actual_negatives = np.sum(np.round(np.clip(1 - target, 0, 1)))

    # Calculate specificity
    specificity = true_negatives / (actual_negatives + smooth)

    return specificity

def auc_s(output, target):
    """Independent AUC calculation function"""
    # Force binarization of target labels (assuming original labels are 0/255 or continuous values)
    target = (target > 0).float()

    # Convert to probabilities and flatten, add detach() to separate gradients
    output_prob = torch.sigmoid(output).squeeze(1).view(-1).detach().cpu().numpy()
    target = target.squeeze(1).view(-1).detach().cpu().numpy()

    # Handle multi-label case (calculate directly for single-label)
    if len(target.shape) > 1 and target.shape[1] > 1:
        # Multi-label: set average='micro' or 'macro'
        auc = roc_auc_score(target, output_prob, average='micro')
    else:
        # Single-label binary classification
        auc = roc_auc_score(target, output_prob)

    return auc

def f1_score_s(output, target):
    """Independent F1 score calculation function"""
    smooth = 1e-5

    # Calculate precision
    precision = precision_s(output, target)

    # Calculate recall
    recall = recall_s(output, target)

    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall + smooth)

    return f1

def mcc_s(output, target):
    """Independent MCC calculation function"""
    # Ensure target labels are binary (assuming target labels are in range 0 to 255)
    target = (target > 0).float()  # Convert target labels to binary 0 and 1 labels

    # Convert model output to probability values
    output_prob = torch.sigmoid(output)

    # Convert probability values to binary labels
    predicted = (output_prob > 0.5).float()  # Use 0.5 as threshold

    # Convert tensors to numpy arrays, add detach() to separate gradients
    target_array = target.squeeze(1).view(-1).detach().cpu().numpy()
    predicted = predicted.squeeze(1).view(-1).detach().cpu().numpy()

    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(target_array, predicted)
    return mcc

def sensitivity_s(output, target):
    """Independent Sensitivity calculation function (same as recall)"""
    smooth = 1e-5

    # Compress output to range [0, 1] using sigmoid function and round to 0 or 1
    predicted = torch.round(torch.sigmoid(output))

    # Convert predictions and targets to numpy arrays on CPU, add detach() to separate gradients
    predicted = predicted.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()

    # Calculate number of true positives
    true_positives = np.sum(np.round(np.clip(predicted * target, 0, 1)))

    # Calculate number of actual positives
    actual_positives = np.sum(np.round(np.clip(target, 0, 1)))

    # Calculate sensitivity
    sensitivity = true_positives / (actual_positives + smooth)

    return sensitivity

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    # Basic metrics
    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    
    # Additional metrics
    # Accuracy
    accuracy_ = accuracy_score(target_.flatten(), output_.flatten())
    
    # AUC (Area Under Curve)
    try:
        auc_ = roc_auc_score(target_.flatten(), output.flatten())
    except:
        auc_ = 0.0
    
    # F1 Score
    f1_ = f1_score(target_.flatten(), output_.flatten())
    
    # Matthews Correlation Coefficient (MCC)
    mcc_ = matthews_corrcoef(target_.flatten(), output_.flatten())
    
    # Jaccard (alternative name for IoU)
    jaccard_ = iou_  # IoU and Jaccard are the same metric
    
    # Sensitivity (same as recall)
    sensitivity_ = recall_
    
    # Hausdorff Distance (if available)
    try:
        hd_ = hd(output_, target_)
        hd95_ = hd95(output_, target_)
    except:
        hd_ = -1
        hd95_ = -1

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_, auc_, f1_, mcc_, jaccard_, sensitivity_
