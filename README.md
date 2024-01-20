1. pip install -r requirements.txt
2. huggingface-cli login (optional, PRIV dataset 쓰려면 필요)
3. wandb login (optional, 안 쓸 거면 --no_wandb)
4. accelerate config
5. accelerate launch train.py