
#train
python mainsefgn.py train --dataset_name 'Chikusei' --epochs 70 --batch_size 16  --n_scale 4 --gpus "0,1"

#test
python mainsefgn.py test --dataset_name 'Chikusei'  --n_scale 4  --gpus "0,1"

