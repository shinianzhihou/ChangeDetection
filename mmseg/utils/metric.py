
class Metric(object):
    def __init__(self, name='metric'):
        super().__init__()
        self.name = name
        self.global_tp = 0
        self.global_fp = 0
        self.global_tn = 0
        self.global_fn = 0
        self.local_tp = 0
        self.local_fp = 0
        self.local_tn = 0
        self.local_fn = 0
    
    def __call__(self,out,gt,set_to_1=False):

        if set_to_1:
            gt[gt>0] = 1
            out[out>0] = 1
            
        self.local_tp = (out * gt).sum()
        self.local_fp = (out * (1 - gt)).sum()
        self.local_tn = ((1 - out) * (1 - gt)).sum()
        self.local_fn = ((1 - out) * gt).sum()
        self.global_tp += self.local_tp
        self.global_fp += self.local_fp
        self.global_tn += self.local_tn
        self.global_fn += self.local_fn
        
        
    def _get_tfpn(self,local=True):
        if local==True:
            return self.local_tp, self.local_fp, self.local_tn, self.local_fn
        else:
            return self.global_tp, self.global_fp, self.global_tn, self.global_fn
        
    def oa(self,local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return (tp + tn) / (tp + fp + tn + fn)
    
    def kappa(self,local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        n = tp + fp + tn + fn
        po = self.oa(local=local)
        pc =  ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        return (po - pc) / (1 - pc) if 1-pc else 1 - pc
    
    def pr(self,local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return tp / (tp + fp) if tp + fp else tp + fp
    
    def re(self,local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return tp / (tp + fn) if tp + fn else tp + fn
    
    
    def f1(self,local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        pr = self.pr(local=local)
        re = self.re(local=local)
        return 2 * pr * re / (pr +re) if pr + re else pr + re
        
    def iou(self,local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        return tp / (tp + fn + fp) if tp + fn + fp else tp + fn + fp

        
    def miou(self,local=True):
        tp, fp, tn, fn = self._get_tfpn(local=local)
        piou = tp / (tp + fn + fp) if tp + fn + fp else tp + fn + fp
        niou = tn / (tn + fn + fp) if tn + fn + fp else tn + fn + fp        
        return (piou + niou) / 2
    
    def calculate(self,local=True):
        self.res = {
            "oa" : self.oa(local=local),
            "kappa" : self.kappa(local=local),
            "pr" : self.pr(local=local),
            "re" : self.re(local=local),
            "f1" : self.f1(local=local),
            "iou" : self.iou(local=local),
            "miou" : self.miou(local=local),
            
        }
        return self.res