export output_dir="./res/train_on_our"
export datasets_path="./datasets"
python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 2 DATASETS.EVL_BATCH 1 DATASETS.SOURCE_DATASET_PATH $datasets_path/our DATASETS.SOURCE_PART train DATASETS.EVL_PART test OUTPUT_DIR $output_dir
