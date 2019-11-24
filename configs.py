#-*-coding:utf-8-*-

class Config:
    
    learning_rate = 1e-3

    momentum = 0.9
    weight_decay = 5e-4

    num_epochs = 1000
    batch_size = 4
    use_gpu = True

    num_workers = 0
    show_every = 100
    save_every = 3000
    test_every = 1000
    image_every = 100

    save_path = './models/Heterogeneous/'
    tensorboard_path = './tensorboard/Heterogeneous/'



