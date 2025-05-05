export output_dir="./res/mpia2our"
export datasets_path="./datasets"
python train_seg.py -cfg ./bashes/mpia2our/segmentation.yaml -opts DATASETS.SOURCE_DATASET_PATH $datasets_path/our_mpia OUTPUT_DIR $output_dir
