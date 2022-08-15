import torchvision.models as tvmodels

import model as mymodels


def build_model(cfg, choice='', **kwargs):
    mcfg = cfg.build.model
    choice = choice if choice else mcfg.choice
    
    if hasattr(mymodels, choice):
        model = getattr(mymodels, choice)(**kwargs)
    elif hasattr(tvmodels, choice):
        model = getattr(tvmodels, choice)(**kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return model
