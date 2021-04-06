#!/bin/bash
# python main_IRT.py --max_epoch 50 --input demo/colorization/goat_input --processed demo/colorization/goat_processed --model colorization --with_IRT 1 --IRT_initialization 1 --output result/colorization
# python main_IRT.py --max_epoch 25 --input demo/Enhancement/bike-packing-input --processed demo/Enhancement/bike-packing-processed --model enhancement --output result/enhancement
# python main_IRT.py --max_epoch 25 --input demo/Dehazing/Haze2_input --processed demo/Dehazing/Haze2_processed/ --model dehazing --output result/dehazing
# python main_IRT.py --max_epoch 25 --input demo/spatial_white_balance/Bedroom_input --processed demo/spatial_white_balance/Bedroom_processed --model spatial_white_balance --output result/spatial_white_balance --with_IRT 1

python main_IRT.py --max_epoch 25 --input demo/jk/enhancement/input --processed demo/jk/enhancement/processed --model enhancement --output result/jk/enhancement --format png
# python main_IRT.py --max_epoch 25 --input demo/jk/spatial_white_balance/input --processed demo/jk/spatial_white_balance/processed --model spatial_white_balance --output result/jk/spatial_white_balance
# python main_IRT.py --max_epoch 25 --input demo/jk/dehazing/input --processed demo/jk/dehazing/processed --model dehazing --output result/jk/dehazing
# python main_IRT.py --max_epoch 25 --input demo/jk/colorization/input --processed demo/jk/colorization/processed --model colorization --output result/jk/colorization
