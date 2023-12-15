#!/bin/bash

/usr/bin/python3 generate.py --step 50 --generate_method generate_based_on_bdr_uncond --model_path /data1/fjx/las-models/model6/model/epoch=1999.ckpt --output_path /data1/fjx/las-models/model6/output_uncond/output --bdr_path /home/fjx/bdr --num_generate 6400