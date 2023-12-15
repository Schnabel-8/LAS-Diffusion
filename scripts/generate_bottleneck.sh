# unconditional generation
# replace airplane with chair, car, rifle, table etc. to generate other categories

python generate.py --model_path ./checkpoints/airplane/epoch\=3999.ckpt --generate_method generate_unconditional --num_generate 16 --steps 50


# category-conditional generation

#python generate.py --model_path ./checkpoints/shape_five/epoch\=3999.ckpt --generate_method generate_based_on_class --data_class chair --num_generate 16 --steps 50
