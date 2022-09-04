import copy


class Metric(object):
    def __init__(
        self,
        init_metric={'f1': 0.0, 'iou': 0.0},
    ):
        super().__init__()
        self.init_metric = init_metric

        self.local_tp = 0
        self.local_fp = 0
        self.local_tn = 0
        self.local_fn = 0

        self.reset()
        self.reset_best()

    def reset(self):
        self.global_tp = 0
        self.global_fp = 0
        self.global_tn = 0
        self.global_fn = 0
    
    def reset_best(self):
        self.best_metric = copy.deepcopy(self.init_metric)

    def print(self, local=True, sep0=" | ", sep1=":", with_best=False):
        if with_best:
            bm = self.best_metric
            return sep0.join([f"{k:3s}{sep1}{v:.3f}({bm[k]:.3f})" for k, v in
                           self.calculate(local=local).items()])
        else:
            return sep0.join([f"{k:3s}{sep1}{v:.3f}" for k, v in
                           self.calculate(local=local).items()])

    def __str__(self):
        return " | ".join([f"{k:3s}:{v:.3f}" for k, v in
                           self.calculate(local=True).items()])

    def __call__(self, out, gt, set_to_1=True):

        if set_to_1:
            gt[gt > 0] = 1
            out[out > 0] = 1

        self.local_tp = (out * gt).sum()
        self.local_fp = (out * (1 - gt)).sum()
        self.local_tn = ((1 - out) * (1 - gt)).sum()
        self.local_fn = ((1 - out) * gt).sum()
        self.global_tp += self.local_tp
        self.global_fp += self.local_fp
        self.global_tn += self.local_tn
        self.global_fn += self.local_fn

    def _get_tfpn(self, local=True):
        if local == True:
            return self.local_tp, self.local_fp, self.local_tn, self.local_fn
        else:
            return self.global_tp, self.global_fp, self.global_tn, self.global_fn

    def oa(self, local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return (tp + tn) / (tp + fp + tn + fn)

    def kappa(self, local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        n = tp + fp + tn + fn
        po = self.oa(local=local)
        pc = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        return (po - pc) / (1 - pc) if 1-pc else 1 - pc

    def pr(self, local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return tp / (tp + fp) if tp + fp else tp + fp

    def re(self, local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return tp / (tp + fn) if tp + fn else tp + fn

    def f1(self, local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        pr = self.pr(local=local)
        re = self.re(local=local)
        return 2 * pr * re / (pr + re) if pr + re else pr + re

    def iou(self, local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return tp / (tp + fn + fp) if tp + fn + fp else tp + fn + fp

    def miou(self, local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        piou = tp / (tp + fn + fp) if tp + fn + fp else tp + fn + fp
        niou = tn / (tn + fn + fp) if tn + fn + fp else tn + fn + fp
        return (piou + niou) / 2

    def update_best(self, res):
        update = []
        for k, v in res.items():
            if v > self.best_metric[k]:
                self.best_metric[k] = v
                update.append(k)

        return update

    def calculate(self, local=True):
        res = {}
        for key in self.best_metric.keys():
            res[key] = getattr(self, key)(local=local)

        if not local:
            self.update_best(res)

        return res
