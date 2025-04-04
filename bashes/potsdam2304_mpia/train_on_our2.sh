export output_dir="./res/train_on_our"
export datasets_path="./datasets"
python train_seg.py -cfg ./bashes/potsdam2304_mpia/segmentation.yaml -opts DATASETS.SOURCE_DATASET_PATH $datasets_path/our_mpia DATASETS.SOURCE_PART train DATASETS.EVL_PART test OUTPUT_DIR $output_dir
