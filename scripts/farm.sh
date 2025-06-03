seq_len=96
batch_size=16 # 32

num_nodes=9


# pred_len 96
pred_len=96
learning_rate=1e-4
channel=128 # 128 Num Encoder dimensions
e_layer=1 #1
d_layer=2
dropout_n=0.01 # 0.7
data_path=Farm$num_nodes


log_path="./Results/${data_path}/"
mkdir -p $log_path
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes $num_nodes \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 100 \
  --seed 2025 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --target "CO2" > $log_file