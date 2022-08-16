import dataset as mydatasets


def build_dataset(cfg, choice='', **kwargs):
    dcfg = cfg.build.dataset
    choice = choice if choice else dcfg.choice
    
    if hasattr(mydatasets,choice):
        dataset = getattr(mydatasets,choice)(**kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return dataset