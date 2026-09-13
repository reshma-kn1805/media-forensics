from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ============================================================
# IMPORT MODEL
# ============================================================

ML_SERVICE_DIR = Path(__file__).resolve().parents[1]

if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from model import create_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml-service"
    / "models"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_model.pt"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.01

VALIDATION_SPLIT = 0.20

RANDOM_SEED = 42

NUM_WORKERS = 0

RESUME_TRAINING = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA TRANSFORMS
# ============================================================

TRAIN_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


VAL_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# CUSTOM IMAGE DATASET
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples

        self.transform = transform


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.samples[index]
        )

        # --------------------------------------------
        # Open image
        # --------------------------------------------

        image = Image.open(
            image_path
        )


        # --------------------------------------------
        # Convert every image to RGB
        # --------------------------------------------

        image = image.convert(
            "RGB"
        )


        # --------------------------------------------
        # Apply transform
        # --------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )


        # --------------------------------------------
        # Return tensor + label
        # --------------------------------------------

        return image, label


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_existing_checkpoint(
    model,
    optimizer
):

    if not RESUME_TRAINING:

        print()

        print(
            "Starting a completely new training run."
        )

        return 0, 0.0


    if not BEST_MODEL_PATH.exists():

        print()

        print(
            "No previous checkpoint found."
        )

        print(
            "Starting from ImageNet-pretrained weights."
        )

        return 0, 0.0


    print()
    print("=" * 60)
    print(
        "EXISTING CHECKPOINT FOUND"
    )
    print("=" * 60)


    print()

    print(
        "Checkpoint:"
    )

    print(
        BEST_MODEL_PATH
    )


    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu"
    )


    # ========================================================
    # LOAD MODEL WEIGHTS
    # ========================================================

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        isinstance(
            checkpoint,
            dict
        )
        and
        "state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

    else:

        # Handle a checkpoint that is itself a state dict

        if isinstance(
            checkpoint,
            dict
        ):

            model.load_state_dict(
                checkpoint
            )

        else:

            raise RuntimeError(
                "The existing checkpoint does not contain "
                "valid model weights."
            )


    # ========================================================
    # LOAD OPTIMIZER STATE
    # ========================================================

    optimizer_restored = False


    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "optimizer_state_dict" in checkpoint
    ):

        try:

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

            optimizer_restored = True

        except Exception as error:

            print()

            print(
                "Optimizer state could not be restored."
            )

            print(
                "Continuing with a fresh optimizer."
            )

            print(
                "Reason:",
                error
            )


    # ========================================================
    # LOAD EPOCH
    # ========================================================

    if (
        isinstance(
            checkpoint,
            dict
        )
    ):

        saved_epoch = checkpoint.get(
            "epoch",
            2
        )

    else:

        saved_epoch = 2


    # ========================================================
    # LOAD VALIDATION ACCURACY
    # ========================================================

    if (
        isinstance(
            checkpoint,
            dict
        )
    ):

        best_val_accuracy = checkpoint.get(
            "val_accuracy",
            checkpoint.get(
                "best_val_accuracy",
                0.0
            )
        )

    else:

        best_val_accuracy = 0.0


    print()

    print(
        "Previous epoch:",
        saved_epoch
    )


    print(
        "Previous best validation accuracy:",
        f"{float(best_val_accuracy) * 100:.2f}%"
    )


    if optimizer_restored:

        print(
            "Optimizer state:",
            "Restored"
        )

    else:

        print(
            "Optimizer state:",
            "Fresh optimizer"
        )


    print()


    return (
        int(saved_epoch),
        float(best_val_accuracy)
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_accuracy,
    class_names
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            epoch,

        "val_accuracy":
            val_accuracy,

        "class_names":
            class_names,

        "model_name":
            "EfficientNet-B0",

        "image_size":
            IMAGE_SIZE
    }


    torch.save(
        checkpoint,
        BEST_MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # SET SEED
    # ========================================================

    set_seed(
        RANDOM_SEED
    )


    # ========================================================
    # HEADER
    # ========================================================

    print()

    print("=" * 60)

    print(
        "MEDIA FORENSICS - DEEPFAKE MODEL TRAINING"
    )

    print("=" * 60)


    # ========================================================
    # CONFIGURATION
    # ========================================================

    print()

    print(
        "Configuration:"
    )


    print(
        "Image size       :",
        IMAGE_SIZE
    )


    print(
        "Batch size       :",
        BATCH_SIZE
    )


    print(
        "Epochs           :",
        EPOCHS
    )


    print(
        "Learning rate    :",
        LEARNING_RATE
    )


    print(
        "Validation split :",
        f"{int(VALIDATION_SPLIT * 100)}%"
    )


    print(
        "Resume training  :",
        RESUME_TRAINING
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device           :",
        device
    )


    # ========================================================
    # DATASET
    # ========================================================

    print()

    print("=" * 60)

    print(
        "LOADING DATASET"
    )

    print("=" * 60)


    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            "Processed dataset directory was not found:\n"
            f"{DATASET_DIR}"
        )


    base_dataset = datasets.ImageFolder(
        root=DATASET_DIR
    )


    # ========================================================
    # CLASS INFORMATION
    # ========================================================

    class_names = list(
        base_dataset.classes
    )


    class_to_index = dict(
        base_dataset.class_to_idx
    )


    print()

    print(
        "Classes:"
    )

    print(
        class_names
    )


    print()

    print(
        "Class mapping:"
    )

    print(
        class_to_index
    )


    # ========================================================
    # VERIFY CLASSES
    # ========================================================

    if (
        "fake" not in class_to_index
        or
        "real" not in class_to_index
    ):

        raise RuntimeError(
            "Dataset must contain both "
            "'fake' and 'real' classes."
        )


    print()

    print(
        "IMPORTANT CLASS ORDER:"
    )


    for class_name, index in (
        class_to_index.items()
    ):

        print(
            f"  Output {index} -> {class_name}"
        )


    # ========================================================
    # DATASET SIZE
    # ========================================================

    total_images = len(
        base_dataset
    )


    print()

    print(
        "Total images:"
    )

    print(
        total_images
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    validation_size = int(
        total_images
        * VALIDATION_SPLIT
    )


    training_size = (
        total_images
        - validation_size
    )


    generator = torch.Generator()


    generator.manual_seed(
        RANDOM_SEED
    )


    indices = torch.randperm(
        total_images,
        generator=generator
    ).tolist()


    train_indices = indices[
        :training_size
    ]


    val_indices = indices[
        training_size:
    ]


    # ========================================================
    # CREATE SAMPLE LISTS
    # ========================================================

    train_samples = [
        base_dataset.samples[index]
        for index in train_indices
    ]


    val_samples = [
        base_dataset.samples[index]
        for index in val_indices
    ]


    # ========================================================
    # CREATE DATASETS
    # ========================================================

    train_dataset = DeepfakeDataset(
        samples=train_samples,
        transform=TRAIN_TRANSFORM
    )


    val_dataset = DeepfakeDataset(
        samples=val_samples,
        transform=VAL_TRANSFORM
    )


    print()

    print(
        "TRAINING SET:",
        len(train_dataset)
    )


    print(
        "VALIDATION SET:",
        len(val_dataset)
    )


    print(
        "TOTAL:",
        len(train_dataset)
        + len(val_dataset)
    )


    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()

    print("=" * 60)

    print(
        "CREATING EFFICIENTNET-B0 MODEL"
    )

    print("=" * 60)


    model = create_model()


    model = model.to(
        device
    )


    print()

    print(
        "Model:"
    )

    print(
        "EfficientNet-B0"
    )


    print()

    print(
        "Device:"
    )

    print(
        device
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    (
        start_epoch,
        best_val_accuracy
    ) = load_existing_checkpoint(
        model,
        optimizer
    )


    # ========================================================
    # CHECK COMPLETION
    # ========================================================

    if start_epoch >= EPOCHS:

        print()

        print("=" * 60)

        print(
            "TRAINING ALREADY COMPLETED"
        )

        print("=" * 60)


        print()

        print(
            "Saved checkpoint already contains",
            EPOCHS,
            "epochs."
        )


        print()

        print(
            "Best validation accuracy:",
            f"{best_val_accuracy * 100:.2f}%"
        )


        return


    # ========================================================
    # START TRAINING
    # ========================================================

    print()

    print("=" * 60)

    print(
        "STARTING TRAINING"
    )

    print("=" * 60)


    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        # ====================================================
        # EPOCH HEADER
        # ====================================================

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            "-" * 60
        )


        # ====================================================
        # TRAINING
        # ====================================================

        model.train()


        running_loss = 0.0

        correct_predictions = 0

        total_predictions = 0


        for images, labels in train_loader:

            images = images.to(
                device
            )


            labels = labels.to(
                device
            )


            # ------------------------------------------------
            # CLEAR GRADIENTS
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            outputs = model(
                images
            )


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # UPDATE WEIGHTS
            # ------------------------------------------------

            optimizer.step()


            # ------------------------------------------------
            # LOSS STATISTICS
            # ------------------------------------------------

            running_loss += (
                loss.item()
                * images.size(0)
            )


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            _, predictions = torch.max(
                outputs,
                1
            )


            correct_predictions += (
                (predictions == labels)
                .sum()
                .item()
            )


            total_predictions += (
                labels.size(0)
            )


        # ====================================================
        # TRAINING METRICS
        # ====================================================

        train_loss = (
            running_loss
            / total_predictions
        )


        train_accuracy = (
            correct_predictions
            / total_predictions
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        validation_loss = 0.0

        validation_correct = 0

        validation_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device
                )


                labels = labels.to(
                    device
                )


                outputs = model(
                    images
                )


                loss = criterion(
                    outputs,
                    labels
                )


                validation_loss += (
                    loss.item()
                    * images.size(0)
                )


                _, predictions = torch.max(
                    outputs,
                    1
                )


                validation_correct += (
                    (predictions == labels)
                    .sum()
                    .item()
                )


                validation_total += (
                    labels.size(0)
                )


        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        val_loss = (
            validation_loss
            / validation_total
        )


        val_accuracy = (
            validation_correct
            / validation_total
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print()

        print(
            "Train Loss     :",
            f"{train_loss:.4f}"
        )


        print(
            "Train Accuracy :",
            f"{train_accuracy * 100:.2f}%"
        )


        print(
            "Val Loss       :",
            f"{val_loss:.4f}"
        )


        print(
            "Val Accuracy   :",
            f"{val_accuracy * 100:.2f}%"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )


            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_accuracy=val_accuracy,
                class_names=class_names
            )


            print()

            print(
                "Best model saved."
            )


        else:

            print()

            print(
                "No improvement."
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)


    print()

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy * 100:.2f}%"
    )


    print()

    print(
        "Best model:"
    )


    print(
        BEST_MODEL_PATH
    )


    print()

    print(
        "Class mapping used:"
    )


    print(
        class_to_index
    )


    print()

    print(
        "Training finished successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()