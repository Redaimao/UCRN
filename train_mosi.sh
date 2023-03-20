CUDA_VISIBLE_DEVICES=7 python main.py \
--model=UCRN --lonly --aonly --vonly \
--name='UCRN_bs32_jsd_mosi' \
--dataset='mosi' --data_path='UCRN/data/' \
--batch_size=32 \
--seed=1234 --num_epochs=40 --when=20 \
--optim='Adam' --jsd \
--lr=0.001