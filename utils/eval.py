import torch

from .metric import get_metric


def eval_model(
    model,
    data_loader,
    cfg,
    writer=None,
    step=0,
    criterion = None,
    save_img = False,
    save_name = ""
):
    device = cfg.MODEL.DEVICE
    model = model.to(device).eval()
    metric_all = dict(zip(
        cfg.EVAL.METRIC,
        [0.0] * len(cfg.EVAL.METRIC)
    ))
    with torch.no_grad():
        loss_all = 0.0
        for batch, data in enumerate(data_loader):
            img1, img2, gt = [img.to(device) for img in data]
            output = model(img1, img2)
            if criterion:
                loss = criterion(output,gt)
                loss_all += loss.item()
            out_tensor = torch.argmax(output, dim=1, keepdim=True).type_as(output)
            gt_tensor = gt[:, 1, :, :]
            metric = get_metric(out_tensor, gt_tensor)
            metric_all = add_metric(metric_all,metric)
            metric_avg = {k : v/data_loader.__len__() for k,v in metric_all.items()}
    if writer:
        writer.add_scalars("test/metric", metric_avg, step)
        if criterion:
            writer.add_scalar("test/loss", loss_all/data_loader.__len__(), step)
    model = model.train()
    return metric_avg


def add_metric(ma,mb):
    """ma = ma + mb"""
    assert set(ma.keys()).issubset(mb.keys())
    for k,v in ma.items():
        ma[k] = mb[k] + v
    return ma
