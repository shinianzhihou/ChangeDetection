import torch.optim as optim


def build_optimizer(cfg, model, choice='', **kwargs):
    ocfg = cfg.build.optimizer
    choice = choice if choice else ocfg.choice

    if hasattr(model, 'parameters'):
        model = model.parameters()

    if hasattr(optim, choice):
        optimizer = getattr(optim, choice)(
            params=filter(lambda p: p.requires_grad, model),
            lr=kwargs.get('lr', ocfg.base_lr),
            weight_decay=kwargs.get('weight_decay', ocfg.weight_decay),
            **kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return optimizer
