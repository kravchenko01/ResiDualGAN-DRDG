export output_dir="./res/rdgv3"
export datasets_path="./datasets"
python train_residualgan.py -cfg ./bashes/potsdam672_mpia/residualgan.yaml -opts OUTPUT_DIR $output_dir DATASETS.SOURCE_PATH $datasets_path/potsdam_672 DATASETS.TARGET_PATH $datasets_path/our_mpia
