

def get_metric(output, target):
    '''Get metrics by t(f)p(n) from binary output and target.'''
    TP = (output * target).sum()
    TN = ((1 - output) * (1 - target)).sum()
    FP = (output * (1 - target)).sum()
    FN = ((1 - output) * target).sum()

    N = TP + TN + FP + FN

    PCC = (TP + TN) / N
    PRE = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (N * N)
    Kappa = (PCC - PRE) / (1 - PRE) if 1-PRE else N-N
    Pr = TP / (TP + FP) if TP+FP else N-N
    Re = TP / (TP + FN) if TP+FN else N-N
    F1 = 2 * Pr * Re / (Pr + Re) if Pr+Re else N-N

    keys = ["TP", "TN", "FP", "FN", "PCC", "Kappa", "Pr", "Re", "F1"]
    values = [TP, TN, FP, FN, PCC, Kappa, Pr, Re, F1]

    return dict(zip(keys, values))

def update_metric(metric, new_metric):
    assert metric.keys()==new_metric.keys()
    num_update = 0
    for k,v in metric.items():
        if v < new_metric[k]:
            metric[k] = new_metric[k]
            num_update += 1
        
    return num_update, metric
        