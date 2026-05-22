if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=MixF
root_path='./'
data_path='./dataset/categories.xlsx'
model_id_name=arxiv
random_seed=2025
data_name=arxiv
seq_len=48
pred_len=6

python -u run.py \
    --random_seed $random_seed \
    --is_training 0 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --input_channels_len 1 \
    --output_channels_len 1 \
    --dropout 0.2 \
    --head_dropout 0 \
    --d_model 16 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 500 \
    --itr 1 \
    --batch_size 32 \
    --gpu 0 \
    --n_heads  2 \
    --e_layers 5 \
    --des 'Exp' \
    --loss_flag 2 \
    --patience 40 \
    --alpha 0.5 \
    --do_predict \
    --learning_rate 0.005 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log