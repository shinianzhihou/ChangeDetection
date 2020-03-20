from yacs.config import CfgNode as CN


_C = CN()

# Build
_C.BUILD = CN()
## Dataloader
_C.BUILD.DATALOADER = CN()
_C.BUILD.DATALOADER.CHOICE = "OSCD"
_C.BUILD.DATALOADER.USE_PART = "all" # "all", "change", "unchange"
## Loss
_C.BUILD.LOSS = CN()
_C.BUILD.LOSS.CHOICE = "BCELoss"
_C.BUILD.LOSS.REDUCTION = "mean"
### class balancing
_C.BUILD.LOSS.USE_POS_WEIGHT = True # BCEWithLogitsLoss
_C.BUILD.LOSS.POS_WEIGHT = 1.0
## Model
_C.BUILD.MODEL = CN()
_C.BUILD.MODEL.CHOICE = "Siamese_unet_conc"
_C.BUILD.MODEL.IN_CHANNEL = 3
_C.BUILD.MODEL.OUT_CHANNEL = 2
_C.BUILD.MODEL.P_DROPOUT = 0.0
_C.BUILD.MODEL.CHANNEL_ATTENTION = True # some tricks
## Optimizer
_C.BUILD.OPTIMIZER = CN()
_C.BUILD.OPTIMIZER.CHOICE = "SGD"
## Learning rate scheduler
_C.BUILD.LR_SCHEDULER = CN()
_C.BUILD.LR_SCHEDULER.CHOICE = "CosineAnnealingLR"
## Tools
_C.BUILD.USE_CHECKPOINT = False
_C.BUILD.USE_TENSORBOARD = False

# Checkpoint
_C.CHECKPOINT = CN()
_C.CHECKPOINT.PATH = "logs/checkpoints/"
_C.CHECKPOINT.PERIOD = 1000

# Dataset
_C.DATASETS = CN()
_C.DATASETS.TRAIN_CSV = "train.csv"
_C.DATASETS.TEST_CSV = "test.csv"

# Dataloader
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.NUM_WORKERS = 0
_C.DATALOADER.SHUFFLE = False

# Eval
_C.EVAL = CN()
## Metric
_C.EVAL.METRIC = ["TP","TN","FP","FN","F1","Re","Pr","PCC","Kappa"]
_C.EVAL.INITIAL_METRIC = [0.0]*len(_C.EVAL.METRIC)
## Checkpoints for evaluation 
_C.EVAL.CHECKPOINTS_PATH = ""
_C.EVAL.SAVE_PATH = ""
_C.EVAL.SAVE_NAME = ""

# Model
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

# Solver
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.NUM_EPOCH = 100
## SGD
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
## Tools
_C.SOLVER.LOG_PERIOD = 100
## Test when model training the model
_C.SOLVER.TEST_WHEN_TRAIN = False
_C.SOLVER.TEST_PERIOD = 200
_C.SOLVER.TEST_BETTER_SAVE = False

# Tensorboard
_C.TENSORBOARD = CN()
_C.TENSORBOARD.PATH = "logs/tensorboard/"

