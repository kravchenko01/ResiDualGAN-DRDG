export output_dir="./res/rdgv3"
export datasets_path="./datasets"
python train_residualgan.py -cfg ./configs/residualgan.yaml -opts LOSS.DEPTH 0.0 LOSS.DEPTH_CYCLE 0.0 LOSS.ADV 1 LOSS.CYCLE 10 OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False DATASETS.SOURCE_PATH $datasets_path/potsdam DATASETS.TARGET_PATH $datasets_path/our &&
python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 8 DATASETS.EVL_BATCH 8 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir
