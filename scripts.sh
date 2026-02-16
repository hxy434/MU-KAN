dataset=busi
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UKAN 

dataset=glas
input_size=512
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UKAN 

dataset=cvc
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UKAN 

dataset=lizi
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UKAN


python val.py --name lizi_UKAN --output_dir outputs


python train.py --arch UKAN --dataset lizi --input_w 256 --input_h 256 --name lizi_UKAN  --data_dir ./inputs
python train.py --arch UKAN --dataset glas --input_w 256 --input_h 256 --name glas_UKAN  --data_dir ./inputs
#-data_dir /root/autodl-tmp/seg/inputs