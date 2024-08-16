import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, AUROC
from DZSpecimenClf import DZSpecimenClf
from dataset import NDPI_DataModule


# Assuming the SpecimenClassifier and NDPI_DataModule are properly defined and imported
class SpecimenClassifier(pl.LightningModule):
    def __init__(self, N, k, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = DZSpecimenClf(N, k, num_classes)
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.f1_score = F1Score(num_classes=num_classes, average="macro")
        self.auroc = AUROC(num_classes=num_classes)

    def forward(self, topview_image_tensor, search_view_indexibles):
        return self.model(topview_image_tensor, search_view_indexibles)

    def training_step(self, batch, batch_idx):
        topview_image, search_view_indexible, class_index = batch
        outputs = self.forward(topview_image, search_view_indexible)
        loss = self.loss_fn(outputs, class_index)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.accuracy(outputs, class_index),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_f1_score",
            self.f1_score(outputs, class_index),
            on_step=True,
            on_epoch=True,
        )
        self.log("train_auroc", self.auroc(outputs, class_index), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        topview_image, search_view_indexible, class_index = batch
        outputs = self.forward(topview_image, search_view_indexible)
        loss = self.loss_fn(outputs, class_index)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.accuracy(outputs, class_index), on_epoch=True)
        self.log("val_f1_score", self.f1_score(outputs, class_index), on_epoch=True)
        self.log("val_auroc", self.auroc(outputs, class_index), on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 32
    N = 512  # Example value
    k = 128  # Example value
    num_classes = 2  # Number of classes in your dataset

    data_module = NDPI_DataModule(metadata_file, batch_size)
    model = SpecimenClassifier(N, k, num_classes)

    # Setup the Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.TensorBoardLogger("training/logs"),
        progress_bar_refresh_rate=20,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
