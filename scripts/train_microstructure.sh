#!/bin/bash
export RESULT_FOLDER="/data1/fjx/las-models/model4"
/usr/bin/python3 train.py --results_folder $RESULT_FOLDER --data_class microstructure --name model --batch_size 128 --new True --continue_training False --image_size 128 --training_epoch 2000 --ema_rate 0.999 --base_channels 32 --save_last False --save_every_epoch 500 --with_attention True --use_text_condition False --use_sketch_condition False --split_dataset False  --lr 1e-4 --optimizier adamw --sdf_folder /home/fjx/bulk_rand/