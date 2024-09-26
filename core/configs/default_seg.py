from yacs.config import CfgNode as CN

_C = CN()

_C.TASK_NAME = "202204_"
_C.OUTPUT_DIR = "./"

_C.MODELS = CN()
_C.MODELS.IN_CHANNELS = 3
# _C.MODELS.CLASSES = 6
_C.MODELS.CLASSES = 2
_C.MODELS.BACKBONE = "DeepLabV3+"
_C.MODELS.ENCODER = "resnet34"
_C.MODELS.PRETRAIN = "imagenet"
_C.MODELS.DEVICE = "cuda"
_C.MODELS.OUT_ADV = True

_C.DATASETS = CN()
_C.DATASETS.SIZE = 112
_C.DATASETS.SOURCE_DATASET_PATH = ""
_C.DATASETS.SOURCE_PART = "all"
_C.DATASETS.TARGET_DATASET_PATH = ""
_C.DATASETS.TARGET_PART = "all"
# _C.DATASETS.VAL_DATASET_PATH = "./datasets/Vaihingen"
_C.DATASETS.VAL_DATASET_PATH = "./datasets/our"
_C.DATASETS.VAL_PART = "test"
# _C.DATASETS.EVL_DATASET_PATH = "./datasets/Vaihingen"
_C.DATASETS.EVL_DATASET_PATH = "./datasets/our"
_C.DATASETS.EVL_PART = "train"
_C.DATASETS.EVL_GENERATE = True
_C.DATASETS.EVL_BATCH = 8


_C.LOSS = CN()
_C.LOSS.ADV = 0.02

_C.TRAIN = CN()
_C.TRAIN.EPOCH = 0
_C.TRAIN.TOTAL_EPOCH = 101
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LR = 0.0002
_C.TRAIN.LR_D = 0.0002
_C.TRAIN.N_CPU = 4
_C.TRAIN.LOSS = "focal"


