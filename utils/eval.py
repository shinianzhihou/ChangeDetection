import torch

from .metric import get_metric


def eval_model(
    model,
    data_loader,
    cfg,
    writer=None,
    step=0,
):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    for batch, data in enumerate(data_loader):
        img1, img2, gt = [img.to(device) for img in data]
        output = model(img1, img2)

        if writer:
            images = {
                "image1": img1,
                "image2": img2,
                "gt": gt[:, 1:2, :, :],
                "output": torch.argmax(output, dim=1, keepdim=True)
            }
            for key, value in images.items():
                if "image" in key and step > 1:
                    continue
                prefix = "test/"
                writer.add_images(prefix+key, value, step)

            out_tensor = torch.argmax(output, dim=1, keepdim=True).type_as(output)
            gt_tensor = gt[:, 1, :, :]
            metric = get_metric(out_tensor, gt_tensor)
            metric_values = [metric.get(key).item() for key in cfg.EVAL.METRIC]
            metric_scalar = dict(zip(cfg.EVAL.METRIC, metric_values))
            writer.add_scalars("test/metric", metric_scalar, step)
    model.train()
    return metric_scalar
