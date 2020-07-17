from __future__ import print_function
import numpy as np
from sklearn import metrics

def auroc(in_out_label, score):
    auroc = metrics.roc_auc_score(in_out_label,score)
    print('auroc : {0}'.format(auroc))
    return auroc

def tnr_tpr95(in_out_label, score):
    fpr, tpr, thresholds = metrics.roc_curve(in_out_label, score, pos_label= 1)
    idx_tpr_95 = np.argmin(np.abs(tpr -0.95))
    tnr = 1 - fpr
    tnr_at_tpr95 = tnr[idx_tpr_95]
    print('tnr at 95% tpr : {0}'.format(tnr_at_tpr95))

    return tnr_at_tpr95

def met(in_out_label, score):
    result = dict()
    auroc = metrics.roc_auc_score(in_out_label,score)
    print('auroc : {0}'.format(auroc))
    result['auroc'] = auroc

    fpr, tpr, thresholds = metrics.roc_curve(in_out_label, score, pos_label= 1)
    idx_tpr_95 = np.argmin(np.abs(tpr -0.95))
    tnr = 1 - fpr
    tnr_at_tpr95 = tnr[idx_tpr_95]
    print('tnr at 95% tpr : {0}'.format(tnr_at_tpr95))
    result['tnr_at_tpr_95'] = tnr_at_tpr95

    return result

