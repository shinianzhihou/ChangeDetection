import torchvision.models as tvmodels

import model as mymodels


def build_model(choice, **kwargs):
    
    if hasattr(mymodels, choice):
        model = getattr(mymodels, choice)(**kwargs)
    elif hasattr(tvmodels, choice):
        model = getattr(tvmodels, choice)(**kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return model
