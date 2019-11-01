#-*-coding:utf-8-*-

class Config:
    
    learning_rate = 1e-3

    momentum = 0.9
    weight_decay = 5e-4

    num_epochs = 1000
    batch_size = 8
    use_gpu = True

    num_workers = 0
    show_every = 100
    save_every = 100
    test_every = 100

    save_path = './models/'



