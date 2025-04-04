export model_path="./res/train_on_our/model/model.pt"
export output_dir="./res/train_on_our"
export evl_data_path="./datasets/our_mpia"
export part="test"
python evaluate.py --model_path $model_path -cfg ./bashes/potsdam2304_mpia/segmentation.yaml -opts DATASETS.EVL_BATCH 1 OUTPUT_DIR $output_dir DATASETS.EVL_DATASET_PATH $evl_data_path DATASETS.EVL_PART $part