import numpy as np
import torch
import torch.nn as nn
from uq360.metrics.classification_metrics import area_under_risk_rejection_rate_curve

REDUCE_FN_MAPPINGS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'none': lambda x: x
}

def hinge_loss(logit, target, margin, reduction='sum'):
    """
    Args:
        logit (torch.Tensor): (N, C, d_1, d_2, ..., d_K)
        target (torch.Tensor): (N, d_1, d_2, ..., d_K)
        margin (float):
    """
    target = target.unsqueeze(1)
    tgt_logit = torch.gather(logit, dim=1, index=target)
    loss = logit - tgt_logit + margin
    loss = torch.masked_fill(loss, loss < 0, 0)
    loss = torch.scatter(loss, dim=1, index=target, value=0)
    reduce_fn = REDUCE_FN_MAPPINGS[reduction]
    return reduce_fn(loss)

def brier_loss(logit, target):
    one_hot = torch.zeros(logit.size()[0], logit.size()[1]).cuda()
    one_hot[torch.arange(target.size()[0]), target] = 1
    pred_softmax = nn.Softmax(dim=1)(logit)
    return torch.mean(torch.sum((pred_softmax - one_hot)**2, dim=1))

def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-10)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def risk_function(y_true, y_pred):
    loss = np.sum(y_true)/y_true.shape[0]
    return loss


def ROC_OOD(test, ood):
    print('OOD Detection Task:')
    entropy = torch.cat((test[0], ood[0]), 0).cpu().numpy()
    confidence = torch.cat((test[1], ood[1]), 0).cpu().numpy()
    binary_label = np.concatenate((np.ones((test[0].size()[0])),  np.zeros((ood[0].size()[0]))), 0)

    entropy_rrrc = area_under_risk_rejection_rate_curve(binary_label, None, y_pred=binary_label, selection_scores=-entropy, risk_func=risk_function, num_bins=10)
    print('AURRRC score of Entropy is', entropy_rrrc)
    confidence_aurrc = area_under_risk_rejection_rate_curve(binary_label, None, y_pred=binary_label, selection_scores=-confidence, risk_func=risk_function, num_bins=10)
    print('AURRRC score of Confidence is', confidence_aurrc)

    return [entropy_rrrc, confidence_aurrc]


def ROC_selective(test, index):
    print('Selective Classification Task:')
    entropy = test[0].cpu().numpy()
    confidence = test[1].cpu().numpy()
    binary_label = (1-index).numpy()  # 1 for correct class, 0 for wrong class

    entropy_rrrc = area_under_risk_rejection_rate_curve(binary_label, None, y_pred=binary_label,selection_scores=-entropy, risk_func=risk_function, num_bins=10)
    print('AURRRC score of Entropy is', entropy_rrrc)
    confidence_aurrc = area_under_risk_rejection_rate_curve(binary_label, None, y_pred=binary_label, selection_scores=-confidence,  risk_func=risk_function, num_bins=10)
    print('AURRRC score of Confidence is', confidence_aurrc)

    return [entropy_rrrc, confidence_aurrc]



