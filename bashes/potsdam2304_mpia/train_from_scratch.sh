export output_dir="./res/rdgv3_scratch"
export datasets_path="./datasets"
python train_seg.py -cfg ./bashes/potsdam2304_mpia/segmentation.yaml -opts DATASETS.SOURCE_DATASET_PATH $datasets_path/potsdam_2304 OUTPUT_DIR $output_dir
