#! /bin/bash

echo `date "+%Y-%m-%d %H:%M:%S"`
start=$(date +%s)

if [ ! -f $OpenNMT_py/preprocess.py ]; then
    print "OpenNMT_py environment variable should be set"
    exit 1
fi


# 1.通用 【1-3】
# 2.训练设置 【4-7】
# 3.GPU  【8-9】
# 4.模型结构 【10-17】
# 5.模型其他设置 【18-21】
# 6.参数设置 【22-31】

cd $OpenNMT_py
python3 train.py \
	-data $data_path/final \
	-save_model $data_path/final-model \
	-save_checkpoint_steps 5000 \
	-batch_size 16 \
	-train_steps 20000 \
	-valid_steps 2000 \
	-valid_batch_size 8 \
	-gpu_ranks 0 1 \
	-world_size 2 \
	-encoder_type transformer \
	-enc_layers 2 \
	-decoder_type transformer \
	-dec_layers 2 \
	-rnn_size 256 \
	-transformer_ff 1024 \
	-word_vec_size 256 \
	-bridge \
	-position_encoding \
	-global_attention general \
	-copy_attn \
	-reuse_copy_attn \
	-param_init 0 \
	-param_init_glorot \
	-normalization tokens \
	-optim adam \
	-adam_beta2 0.998 \
	-learning_rate 2 \
	-warmup_steps 2000 \
	-decay_method noam \
	-max_grad_norm 0 \
	-label_smoothing 0.1 \
	-dropout 0.1 \
	# -attention_dropout 0.1 \
	# -dropout_steps 0 \
	> $data_path/train.final.out
echo "train.sh" >> $data_path/train.out

end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.

