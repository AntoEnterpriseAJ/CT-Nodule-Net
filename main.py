from argparse import ArgumentParser, Namespace

import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from data.luna_dataset import LunaDataset
from model.nodule_net import NoduleNet


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--epochs",
        help="the number of inference steps",
        default=10,
        type=int,
    )

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


def training_loop(
    model: NoduleNet,
    n_epochs: int,
    device: str,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    train_dl: DataLoader,
) -> None:
    model.train()

    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for batch_ndex, batch_tuple in enumerate(train_dl):
            inputs, labels, _, _ = batch_tuple

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)

            optimizer.zero_grad()
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch{epoch}/{n_epochs}, Batch{batch_ndex}/{len(train_dl)}: loss={loss.item()}")

        print(f"Epoch{epoch}/{n_epochs}: avg loss={total_loss / len(train_dl.dataset)}")  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stride = 10
    train_dataset = LunaDataset(validate_stride=stride, validate=False)

    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = NoduleNet()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    training_loop(
        model=model,
        n_epochs=args.epochs,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        train_dl=train_dl,
    )


if __name__ == "__main__":
    main()
