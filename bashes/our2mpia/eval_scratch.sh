export model_path="./res/our2mpia/model/model.pt"
export output_dir="./res/our2mpia"
export evl_data_path="./datasets/our_mpia"
export part="all"
python evaluate.py --model_path $model_path -cfg ./bashes/our2mpia/segmentation.yaml -opts DATASETS.EVL_BATCH 32 OUTPUT_DIR $output_dir DATASETS.EVL_DATASET_PATH $evl_data_path DATASETS.EVL_PART $part