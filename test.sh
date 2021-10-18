#!/bin/bash
python dvp_video_consistency.py --max_epoch 25 --input demo/consistency/Enhancement/bike-packing-input --processed demo/consistency/Enhancement/bike-packing-processed --task enhancement --format png
# python dvp_video_consistency.py --max_epoch 50 --input demo/consistency/colorization/goat_input --processed demo/consistency/colorization/goat_processed --task colorization --with_IRT 1 --IRT_initialization 1 --output result/colorization
# python dvp_video_consistency.py --max_epoch 25 --input demo/consistency/colorization/goat_input --processed demo/consistency/colorization/goat_processed --task colorization  --output result/colorization

# python dvp_video_consistency.py --max_epoch 25 --input demo/consistency/Dehazing/Haze2_input --processed demo/consistency/consistency/Dehazing/Haze2_processed/ --task dehazing --output result/dehazing
# python dvp_video_consistency.py --max_epoch 25 --input demo/consistency/consistency/spatial_white_balance/Bedroom_input --processed demo/consistency/consistency/spatial_white_balance/Bedroom_processed --task spatial_white_balance --output result/spatial_white_balance --with_IRT 1

