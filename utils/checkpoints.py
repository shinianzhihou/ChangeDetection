import torch

def save_checkpoint_with_optimizer(checkpoint_path, model, optimizer=None):
    '''Save model and optimizer.
    
    If  optimizer=None, just load model from checkpoints.

    Args:
        checkpoint_path: string
            the path to save the model and optimizer.
        model: pytorch model
            the model to be saved
        optimizer: pytorch optimizer
            such as torch.optim.Adam()

    Return:
        None
    '''
    if optimizer is None:
        state = {
            'state_dict': model.state_dict(),
        }
    else:
        state = {
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}

    torch.save(state, checkpoint_path)


def load_checkpoint_with_optimizer(checkpoint_path, model, optimizer=None):
    '''Load model and optimizer from checkpoint.

    If  optimizer=None, just load model from checkpoints.
    
    Args:
        checkpoint_path: string
            the path to load the model and optimizer.
        model: pytorch model
            the model to load checkpoint
        optimizer: pytorch optimizer
            the optimizer to load checkpoint

    Return:
        return the optimizer and model that have been loaded from checkpoint.
    '''
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer is None:
        return model
    else:
        optimizer.load_state_dict(state['optimizer'])
        return model, optimizer
    
    
def scwo(checkpoint_path, model, optimizer=None):
    save_checkpoint_with_optimizer(checkpoint_path, model, optimizer)
    
def lcwo(checkpoint_path, model, optimizer=None):
    return load_checkpoint_with_optimizer(checkpoint_path, model, optimizer)