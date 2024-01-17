import argparse

from model import Hypformer
from dataset import Dataset
from utils import save_path
from trainer import Trainer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_e", type=str, default="cuda:0")
    parser.add_argument("--device_d", type=str, default="cuda:1")
    parser.add_argument("--d_encoder", type=int, default=768)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--d_v", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=6)

    parser.add_argument("--dataset", type=str, default="WOS_S")
    parser.add_argument("--lr", type=float, default=1e-06)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bb", type=int, default=1)
    parser.add_argument("--n_print", type=int, default=500)
    parser.add_argument("--n_save", type=int, default=5000)
    parser.add_argument("--checkpoint", type=int, default=None)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    n_label, max_depth = dataset.n_label, dataset.max_depth

    model = Hypformer(
        args.d_encoder,
        args.d_model,
        args.d_k,
        args.d_v,
        args.n_head,
        args.n_layer,
        n_label,
        max_depth
    ).to(args.device_d)

    if args.checkpoint is not None:
        checkpoint = torch.load(save_path(args.checkpoint))
        model.load_state_dict(checkpoint["model"], strict=False)

    trainer = Trainer(
        model=model,
        device_e=args.device_e,
        device_d=args.device_d
    )

    trainer.train(
        dataset=dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        n_bb=args.n_bb,
        n_print=args.n_print,
        n_save=args.n_save
    )