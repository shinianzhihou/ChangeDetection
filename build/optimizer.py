import torch.optim as optim


def build_optimizer(model, choice, lr, weight_decay, **kwargs):

    if hasattr(model, 'parameters'):
        model = model.parameters()

    if hasattr(optim, choice):
        optimizer = getattr(optim, choice)(
            params=filter(lambda p: p.requires_grad, model),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return optimizer
