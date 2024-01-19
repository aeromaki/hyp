import argparse
import torch
import os
from transformers import AutoConfig

from model import Hypformer
from dataset import Dataset
from trainer import Trainer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--project", type=str, default="hyp")
    parser.add_argument("--name", type=str, default=None)

    parser.add_argument("--device_e", type=str, default="cuda:0")
    parser.add_argument("--device_d", type=str, default="cuda:1")

    parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--d_eh", type=int, default=None)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--d_v", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=6)

    parser.add_argument("--dataset", type=str, default="WOS_L")
    parser.add_argument("--mask_label", action="store_true")
    parser.add_argument("--use_weight", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-05)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--n_bb", type=int, default=1)
    parser.add_argument("--n_print", type=int, default=100)

    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--n_save", type=int, default=2000)
    parser.add_argument("--n_iter", type=int, default=5000)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints")

    parser.add_argument("--detect_anomaly", action="store_true")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    n_label, max_depth = dataset.n_label, dataset.max_depth

    d_encoder = AutoConfig.from_pretrained(args.encoder_name).hidden_size

    if args.d_eh is None:
        args.d_eh = d_encoder * 4
    if args.d_v is None:
        args.d_v = args.d_k
    if args.d_ff is None:
        args.d_ff = args.d_model * 4

    model = Hypformer(
        d_encoder,
        args.d_eh,
        args.d_model,
        args.d_k,
        args.d_v,
        args.n_head,
        args.d_ff,
        args.n_layer,
        n_label,
        max_depth
    ).to(args.device_d)

    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    n_ckpt = args.ckpt if args.ckpt is not None else 0
    def save_path(x: int) -> str:
        alter_name = "@".join(args.encoder_name.split("/"))
        save_path = f"{args.ckpt_path}/{args.dataset}-{alter_name}-{args.d_eh}-{args.d_model}-{args.d_k}-{args.d_v}-{args.n_head}-{args.d_ff}-{args.n_layer}-ckpt-{x+n_ckpt}.tar"
        return save_path

    if args.ckpt is not None:
        ckpt = torch.load(save_path(0))
        model.load_state_dict(ckpt["model"], strict=False)

    trainer = Trainer(
        model=model,
        encoder_name=args.encoder_name,
        device_e=args.device_e,
        device_d=args.device_d
    )

    if args.detect_anomaly:
        torch.autograd.detect_anomaly()

    if args.name is None:
        mask_label = "-mask" if args.mask_label else ""
        use_weight = "-weight" if args.use_weight else ""
        args.name = f"{args.dataset}-{args.encoder_name}-{args.d_eh}-{args.d_model}-{args.d_k}-{args.d_v}-{args.n_head}-{args.d_ff}-{args.n_layer}"
        args.name += f"{mask_label}{use_weight}"

    config_wandb = {
        "project": args.project,
        "name": args.name,
        "config": {
            "dataset": args.dataset,
            "encoder": args.encoder_name,
            "d_eh": args.d_eh,
            "d_model": args.d_model,
            "d_k": args.d_k,
            "d_v": args.d_v,
            "n_head": args.n_head,
            "d_ff": args.d_ff,
            "n_layer": args.n_layer,
            "lr": args.lr,
            "batch_size": args.batch_size * args.n_bb
        }
    } if not args.no_wandb else None

    trainer.train(
        dataset=dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        n_bb=args.n_bb,
        n_print=args.n_print,
        n_val=args.n_val,
        n_save=args.n_save,
        n_iter=args.n_iter,
        save_path=save_path,
        mask_label=args.mask_label,
        use_weight=args.use_weight,
        config_wandb=config_wandb
    )