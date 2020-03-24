from torch.utils.data import DataLoader

from data.isprs import ISPRS
from data.normal import Normal
# TODO(SNian) : add some other dataloaders

def build_dataloader(cfg, test=False):
    dcfg = cfg.BUILD.DATALOADER

    csv_path = cfg.DATASETS.TEST_CSV if test else cfg.DATASETS.TRAIN_CSV
    datasets_map = {
        "ISPRS": ISPRS(csv_path),
        "Szada": Normal(csv_path,test=test),
    }

    assert dcfg.CHOICE in datasets_map.keys()

    datasets = datasets_map[dcfg.CHOICE]
    bs = cfg.DATALOADER.TEST_BATCH_SIZE if test else cfg.DATALOADER.BATCH_SIZE
    shuffle = cfg.DATALOADER.TEST_SHUFFLE if test else cfg.DATALOADER.SHUFFLE
    dataloader = DataLoader(dataset=datasets,
                            batch_size=bs,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            shuffle=shuffle)

    return dataloader
