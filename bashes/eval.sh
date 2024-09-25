export model_path="./res/rdgv3/model/model.pt"
export output_dir="./res"
export evl_data_path="./datasets/our"
export part="all"
python evaluate.py --model_path $model_path -cfg ./configs/segmentation.yaml -opts DATASETS.EVL_BATCH 32 OUTPUT_DIR $output_dir DATASETS.EVL_DATASET_PATH $evl_data_path DATASETS.EVL_PART $part