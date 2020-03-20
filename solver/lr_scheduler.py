from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


# In fact, it is nearly same as `torch.optim.lr_scheduler.MultiStepLR`
# It can be replaced when torch is updated.
class MultiStepLR(_LRScheduler):
    def __len__(self,optimizer,milestones,gamma=0.1,last_epoch=-1):
        assert list(milestones)==sorted(milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepLR,self).__init__(optimizer,last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]
