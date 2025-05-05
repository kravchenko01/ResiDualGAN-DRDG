export output_dir="./res/our2mpia"
export datasets_path="./datasets"
python train_seg.py -cfg ./bashes/our2mpia/segmentation.yaml -opts DATASETS.SOURCE_DATASET_PATH $datasets_path/our OUTPUT_DIR $output_dir
