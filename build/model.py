import torchvision.models as tvmodels
import change_detection_pytorch as cdpmodels
import model as mymodels


def build_model(choice, **kwargs):
    
    if hasattr(mymodels, choice):
        model = getattr(mymodels, choice)(**kwargs)
    elif hasattr(tvmodels, choice.replace("tv_","")):
        choice = choice.replace("tv_","")
        model = getattr(tvmodels, choice)(**kwargs)
    elif hasattr(cdpmodels, choice.replace("cdp_","")):
        choice = choice.replace("cdp_","")
        model = getattr(cdpmodels, choice)(**kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return model
