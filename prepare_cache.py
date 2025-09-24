from argparse import ArgumentParser, Namespace

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data.luna_dataset import LunaDataset


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--num-workers",
        help="the number of workers used for data loading",
        default=6,
        type=int,
    )

    parser.add_argument(
        "--batch-size",
        help="the number of samples in a training batch",
        default=32,
        type=int,
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    luna_dataset = LunaDataset()

    luna_dl = DataLoader(
        dataset=luna_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for _ in tqdm(luna_dl, desc="Preparing cache", total=len(luna_dl)):
        pass


if __name__ == "__main__":
    main()
