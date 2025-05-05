export output_dir="./res/mpi2our"
export datasets_path="./datasets"
python train_seg.py -cfg ./bashes/mpi2our/segmentation.yaml -opts DATASETS.SOURCE_DATASET_PATH $datasets_path/our_mpi OUTPUT_DIR $output_dir
