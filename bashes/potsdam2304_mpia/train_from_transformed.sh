export output_dir="./res/rdgv3"
export datasets_path="./datasets"
python train_seg.py -cfg ./bashes/potsdam2304_mpia/segmentation.yaml -opts DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir
