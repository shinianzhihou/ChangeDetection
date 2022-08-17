import dataset as mydatasets


def build_dataset(choice, **kwargs):    
    if hasattr(mydatasets,choice):
        dataset = getattr(mydatasets,choice)(**kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return dataset