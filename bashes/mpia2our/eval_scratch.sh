export model_path="./res/mpia2our/model/model.pt"
export output_dir="./res/mpia2our"
export evl_data_path="./datasets/our"
export part="all"
python evaluate.py --model_path $model_path -cfg ./bashes/mpia2our/segmentation.yaml -opts DATASETS.EVL_BATCH 32 OUTPUT_DIR $output_dir DATASETS.EVL_DATASET_PATH $evl_data_path DATASETS.EVL_PART $part