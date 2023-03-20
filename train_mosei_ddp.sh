CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 \
main.py --model=UCRN --lonly --aonly --vonly \
--ddp \
--name='ucrn_mosei_ddp' \
--dataset='mosei_senti' --data_path='UCRN/data/' \
--batch_size=32 \
--seed=1234 --num_epochs=80 --when=40 \
--optim='Adam' \
--lr=0.001