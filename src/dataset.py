"""
Shared Dataset class for style/artist/era tasks. One class, parameterized
by which metadata column is the label, so the three tasks reuse identical
image-loading and splitting logic.
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

TASK_TO_COLUMN = {
    "style": "style",
    "artist": "artist",
    "era": "era_label",
    "genre": "genre",
}


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            # RandomResizedCrop (instead of a plain Resize) gives real scale/
            # crop variation per epoch, which matters a lot for a dataset
            # this size — a plain resize means the model sees each image in
            # basically the same framing every epoch, which encourages
            # memorization rather than generalization.
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


class ArtworkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, label_to_idx: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.label_to_idx = label_to_idx
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row[self.label_col]]
        return image, label


def load_splits(metadata_csv: str, task: str):
    """
    Returns (train_df, val_df, test_df, label_to_idx) for the given task,
    with rows missing that task's label dropped and a stratified split.
    """
    if task not in TASK_TO_COLUMN:
        raise ValueError(f"Unknown task '{task}', expected one of {list(TASK_TO_COLUMN)}")
    label_col = TASK_TO_COLUMN[task]

    df = pd.read_csv(metadata_csv)
    df = df.dropna(subset=["path", label_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No rows have a usable '{label_col}' label. "
            f"Check metadata.csv — this task isn't trainable with current data."
        )

    labels = sorted(df[label_col].unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}

    train_df, temp_df = train_test_split(
        df, test_size=(config.VAL_FRAC + config.TEST_FRAC),
        stratify=df[label_col], random_state=config.RANDOM_SEED,
    )
    relative_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df[label_col], random_state=config.RANDOM_SEED,
    )

    print(f"[{task}] classes: {len(labels)} | "
          f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    return train_df, val_df, test_df, label_to_idx