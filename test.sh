python main_IRT.py --max_epoch 50 --input demo/colorization/goat_input --processed demo/colorization/goat_processed --model colorization --with_IRT 1 --IRT_initialization 1 --output ./result/colorization
python main_IRT.py --max_epoch 25 --input demo/Enhancement/bike-packing-input --processed demo/Enhancement/bike-packing-processed --model enhancement --output ./result/enhancement
python main_IRT.py --max_epoch 25 --input demo/Dehazing/Haze2_input --processed demo/Dehazing/Haze2_processed/ --model dehazing --output ./result/dehazing
python main_IRT.py --max_epoch 25 --input demo/spatial_white_balance/Bedroom_input --processed demo/spatial_white_balance/Bedroom_processed --model spatial_white_balance --output ./result/spatial_white_balance --with_IRT 1
