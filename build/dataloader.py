from torch.utils.data import DataLoader

from data.isprs import ISPRS


def build_dataloader(cfg, test=False):
    dcfg = cfg.BUILD.DATALOADER

    csv_path = cfg.DATASETS.TEST_CSV if test else cfg.DATASETS.TRAIN_CSV
    datasets_map = {
        "ISPRS": ISPRS(csv_path)
    }

    assert dcfg.CHOICE in datasets_map.keys()

    datasets = datasets_map[dcfg.CHOICE]

    dataloader = DataLoader(dataset=datasets,
                            batch_size=cfg.DATALOADER.BATCH_SIZE,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            shuffle=cfg.DATALOADER.SHUFFLE)

    return dataloader
