python train_translation.py \
    --vqgan_ckpt_path path/to/vqgan_checkpoint.ckpt \
    --text_model bert-base-uncased \
    --train_data path/to/train_data.json \
    --val_data path/to/val_data.json \
    --output_dir path/to/output_dir \
    --batch_size 16 \
    --max_seq_length 256 \
    --learning_rate 1e-4 \
    --num_epochs 10