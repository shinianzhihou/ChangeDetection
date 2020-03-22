from model.siamese_unet_conc import Siamese_unet_conc

# Q1 Why not directly import the models and their configs like `Siamese_unet_conc(3,2,0.0)`?
# A1 When the configs about models need to be changed, you can easily 
#  change them in `.yaml` files and don't need to open the script here.

# TODO(SNian) : test whether the `map` will increase the used memory.
# TODO(SNian) : add some other models

def build_model(cfg):

    mcfg = cfg.BUILD.MODEL

    model_map = {
        "Siamese_unet_conc" : Siamese_unet_conc(mcfg.IN_CHANNEL,mcfg.OUT_CHANNEL,mcfg.P_DROPOUT),
    }

    assert mcfg.CHOICE in model_map.keys()
    
    return model_map[mcfg.CHOICE]

# def build_model():
#     '''Even you can directly change the model here.'''
#     return Siamese_unet_conc(3,2,0.0)
