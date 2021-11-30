import torch
import torch.utils.data
import deep_trainer
import deep_trainer.pytorch.metric

import HandGestureDataset
import HandGestureClassifier


def train():
    # trainset
    dataset = HandGestureDataset.HandGestureDataset()
    train_size = int(len(dataset) * 0.75)
    val_size = len(dataset) - train_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(trainset, 20, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, len(valset), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandGestureClassifier.HandGestureClassifier(21 * 3, len(dataset.classes))
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=len(dataset) * 20, gamma=0.1
    )  # Decay by 10 every 50 epochs

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()  # For classification for instance
    metrics = deep_trainer.pytorch.metric.MetricsHandler(
        [
            deep_trainer.pytorch.metric.Accuracy(train=False),
            deep_trainer.pytorch.metric.BalancedAccuracy(train=False),
        ]
    )

    # Training
    trainer = deep_trainer.PytorchTrainer(
        model,
        optimizer,
        scheduler,
        metrics,
        save_mode="none",
        device=device,
    )
    trainer.train(150, train_loader, criterion, val_loader=val_loader)


train()
