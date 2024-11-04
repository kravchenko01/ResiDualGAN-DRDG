export output_dir="./res/rdgv3"
export datasets_path="./datasets"
python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 8 DATASETS.EVL_BATCH 8 DATASETS.SOURCE_DATASET_PATH $datasets_path/potsdam OUTPUT_DIR $output_dir
