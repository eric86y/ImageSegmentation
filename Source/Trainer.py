import os
import json
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as sm

from tqdm import tqdm
from typing import List
from datetime import datetime
from Source.Utils import create_dir, get_filename
from Source.Dataset import BinaryDataset, MulticlassDataset
from Source.Augmentations import get_augmentations
from sklearn.model_selection import train_test_split

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import Dice
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex

class BinarySegmentationTrainer:
    def __init__(
        self,
        train_x: List[str],
        train_y: List[str],
        valid_x: List[str],
        valid_y: List[str],
        test_x: List[str],
        test_y: List[str],
        image_width: int = 512,
        image_height: int = 512,
        batch_size: int = 32,
        class_threshold: float = 0.8,
        network: str = "deeplab",
        output_path: str = "Output",
        export_onnx: str = "yes",
    ) -> None:
        print("Initializing Binary Segmentation trainer...")

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.image_width = image_width
        self.image_height = image_height
        self.class_threshold = class_threshold
        self.batch_size = batch_size
        self.network = network
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pin_memory = True
        self.default_learning_rate = 0.001
        self.time_stamp = datetime.now()
        self.output_path = os.path.join(output_path, f"{self.time_stamp.year}-{self.time_stamp.month}-{self.time_stamp.day}_{self.time_stamp.hour}-{self.time_stamp.minute}")
        self.jaccard_scorer = BinaryJaccardIndex(threshold=self.class_threshold).to(self.device)

        create_dir(self.output_path)

        self.save_train_data()

        self.train_transforms, self.color_transforms, self.val_transforms = get_augmentations(
            image_width=self.image_width, image_height=self.image_height
        )

        self.train_ds = BinaryDataset(
            images=self.train_x,
            masks=self.train_y,
            augmentation_transforms=self.train_transforms,
            color_transforms=self.color_transforms
        )
        self.valid_ds = BinaryDataset(
            images=self.valid_x,
            masks=self.valid_y,
            augmentation_transforms=self.val_transforms,
            color_transforms=self.color_transforms
        )

        self.test_ds = BinaryDataset(
            images=self.test_x,
            masks=self.test_y,
            augmentation_transforms=self.val_transforms
        )

        self.train_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.valid_dl = DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model = sm.DeepLabV3Plus(classes=1).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.default_learning_rate
        )

        self.export_onnx = True if export_onnx == "yes" else False

        try:
            print("Checking dataloaders...")
            next(iter(self.train_dl))
            next(iter(self.valid_dl))
            print("done!")

        except BaseException as e:
            logging.error(f"Failed to iterate over Dataloaders: {e}.")

    def check_accuracy(self, loader, model):

        dice_loss_fn = sm.losses.DiceLoss(mode="binary", from_logits=True, smooth=1e-7)
        focal_loss_fn = sm.losses.FocalLoss(mode="binary", alpha=0.25)

        val_dice_scores = []
        val_jaccard_scores = []
        val_total_loss = []
        model.eval()

        with torch.no_grad():
            for _, (x, y) in tqdm(enumerate(loader), total=len(loader)):
                x = x.float().to(device=self.device)
                y = y.float().unsqueeze(1).to(device=self.device)

                predictions = model(x)
                dice_loss = dice_loss_fn(predictions, y)
                focal_loss = focal_loss_fn(predictions, y)
                loss = dice_loss + (1 * focal_loss)

                val_total_loss.append(loss.item())

                preds = torch.sigmoid(predictions)
                _jaccard_score = self.jaccard_scorer(preds, y)
                val_jaccard_scores.append(_jaccard_score)

                preds = (preds > self.class_threshold).float()

                # dice for binary classifcation
                _dice_score = (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8
    
                val_dice_scores.append(_dice_score)

        val_total_loss = torch.mean(torch.tensor(val_total_loss))
        val_dice_score = torch.mean(torch.tensor(val_dice_scores))
        val_jaccard_score = torch.mean(torch.tensor(val_jaccard_scores))

        print(f"Validation Loss: {val_total_loss}, Validation Dice-Score: {val_dice_score}, Validation Jaccard Score: {val_jaccard_score}")
        return val_total_loss.item(), val_dice_score.item(), val_jaccard_score.item()
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, state, filename="model_save.pth"):
        print(f"Saving Checkpoint to: {filename}")
        torch.save(state, filename)

    def save_train_data(self):
        train_samples = os.path.join(self.output_path, "train_samples.txt")
        val_samples = os.path.join(self.output_path, "val_samples.txt")
        test_samples = os.path.join(self.output_path, "test_samples.txt")

        with open(train_samples, "w") as f:
            for entry in self.train_x:
                file_n = get_filename(entry)
                f.write(f"{file_n}\n")

        with open(val_samples, "w") as f:
            for entry in self.valid_x:
                file_n = get_filename(entry)
                f.write(f"{file_n}\n")

        with open(test_samples, "w") as f:
            for entry in self.test_x:
                file_n = get_filename(entry)
                f.write(f"{file_n}\n")

    def save_model_config(self, model_name: str):
        model_config = {
            "model": model_name,
            "architecture": self.network,
            "input_width": str(self.image_width),
            "input_height": str(self.image_height),
            "batch_size": str(self.batch_size),
        }

        out_file = os.path.join(self.output_path, "model_config.json")
        json_out = json.dumps(model_config, ensure_ascii=False, indent=2)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(json_out)

        print(f"Saved model config to: {out_file}")

    def save_training_history(
        self,
        train_losses: List[float],
        train_scores: List[float],
        val_losses: List[float],
        val_scores: List[float],
        val_jaccard_scores: List[float],
        out_file: str,
    ) -> None:
        print(f"Saving Training History.... {out_file}")

        with open(f"{out_file}", "w") as f:
            f.write(f"Train Losses: {str(train_losses)}")
            f.write(f"Train Dice Scores: {str(train_scores)}")
            f.write(f"Validation Losses: {str(val_losses)}")
            f.write(f"Validation Dice Scores: {str(val_scores)}")
            f.write(f"Validation Jaccard Scores: {str(val_jaccard_scores)}")

    def export2onnx(
        self, model, model_name: str, mode: str = "cpu", opset: int = 16
    ) -> None:
        model.eval()
        model_input = torch.randn(
            [1, 3, self.image_height, self.image_width], device=self.device
        )

        torch.onnx.export(
            model,
            model_input,
            f"{self.output_path}/{model_name}_{self.device}.onnx",
            export_params=True,
            opset_version=opset,
            verbose=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def train_model(self, loader, model, optimizer, dice_loss_fn, focal_loss_fn, bce_loss_fn, scaler, loss_fn: str = "combined"):
        model.train()
        loop = tqdm(loader)
        epoch_dice_score = []
        epoch_jaccard_scores = []
        epoch_loss = []

        for _, (x, y) in enumerate(loop):
            x = x.float().to(device=self.device)
            y = y.float().unsqueeze(1).to(device=self.device)

            with torch.autocast(self.device):
                predictions = model(x)

                if loss_fn == "dice":
                    loss=dice_loss_fn(predictions, y)
                elif loss_fn == "focal":
                    loss=focal_loss_fn(predictions, y)

                elif loss_fn == "combined":
                    dice_loss = dice_loss_fn(predictions, y)
                    focal_loss = focal_loss_fn(predictions, y)
                    loss = dice_loss + (1 * focal_loss)

                else:
                    loss = bce_loss_fn(predictions, y)

                predictions = torch.sigmoid(predictions)
                jaccard_score = self.jaccard_scorer(predictions, y)
                epoch_jaccard_scores.append(jaccard_score)

                predictions = (predictions > 0.8).float()

                # dice for binary classifcation
                dice_score = (2 * (predictions * y).sum()) / (
                    predictions + y
                ).sum() + 1e-8
                
                epoch_dice_score.append(dice_score)
                epoch_loss.append(loss.item())

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update loop
            loop.set_postfix(Summary=f"loss={loss.item()}")

        epoch_dice_score = torch.mean(torch.tensor(epoch_dice_score))
        epoch_loss = torch.mean(torch.tensor(epoch_loss))
        print(f"Epoch Loss: {epoch_loss}, Epoch Dice Score: {epoch_dice_score}")

        return epoch_loss.item(), epoch_dice_score.item()

    def train(
        self,
        epochs: int = 10,
        model_name: str = "segmentation_model",
        loss: str = "combined",
    ) -> None:
        torch.cuda.empty_cache()

        train_loss_history = []
        val_loss_history = []

        train_score_history = []
        val_score_history = []
        val_jaccard_history = []

        best_val_score = 0

        dice_loss_fn = sm.losses.DiceLoss(mode="binary", from_logits=True, smooth=1e-7)
        focal_loss_fn = sm.losses.FocalLoss(mode="binary", alpha=0.25)
        bce_loss_fn = nn.BCEWithLogitsLoss()

        scaler = torch.amp.GradScaler(self.device)
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        for epoch in range(epochs):
            print(f"Training Epoch: {epoch+1}/{epochs}")
            train_loss, train_score = self.train_model(
                loader=self.train_dl,
                model=self.model,
                optimizer=self.optimizer,
                dice_loss_fn=dice_loss_fn,
                focal_loss_fn=focal_loss_fn,
                bce_loss_fn=bce_loss_fn,
                loss_fn=loss,
                scaler=scaler,
            )

            train_score_history.append(train_score)
            train_loss_history.append(train_loss)

            val_loss, val_score, jaccard_score = self.check_accuracy(loader=self.valid_dl, model=self.model)
            val_score_history.append(val_score)
            val_loss_history.append(val_loss)
            val_jaccard_history.append(jaccard_score)

            if val_score > best_val_score:
                best_val_score = val_score

                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }

                self.save_checkpoint(
                    checkpoint, filename=f"{self.output_path}/{model_name}.pth"
                )

            scheduler.step()

        self.save_training_history(
            train_losses=train_loss_history,
            train_scores=train_score_history,
            val_losses=val_loss_history,
            val_scores=val_score_history,
            val_jaccard_scores=val_jaccard_history,
            out_file=f"{self.output_path}/{model_name}_train_history.txt",
        )

        self.save_model_config(model_name=model_name)

        if self.export_onnx:
            self.export2onnx(self.model, model_name=model_name)



class MultiSegmentationTrainer:
    def __init__(
        self,
        train_x: List[str],
        train_y: List[str],
        valid_x: List[str],
        valid_y: List[str],
        test_x: List[str],
        test_y: List[str],
        classes: list[str],
        image_width: int,
        image_height: int,
        batch_size: int = 32,
        network: str = "deeplab",
        output_path: str = "Output",
        export_onnx: str = "yes",
    ) -> None:
        print("Initializing Mutliclass Segmentation trainer...")

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.image_width = image_width
        self.image_height = image_height
        self.classes = classes
        self.batch_size = batch_size
        self.network =network
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.time_stamp = datetime.now()
        self.output_path = os.path.join(output_path, f"{self.time_stamp.year}-{self.time_stamp.month}-{self.time_stamp.day}_{self.time_stamp.hour}-{self.time_stamp.minute}")
        self.export_onnx = export_onnx
        self.jaccard_scorer = MulticlassJaccardIndex(num_classes=len(self.classes)).to(self.device)
        self.dice_scorer = Dice(
            num_classes=len(classes), 
            multiclass=True,
            threshold=0.5).to(self.device)

        create_dir(self.output_path)
        self.save_train_data()

        self.train_transforms, self.color_transforms, self.val_transforms = get_augmentations(
            image_width=self.image_width, image_height=self.image_height
        )

        self.train_ds = MulticlassDataset(
            images=self.train_x, masks=self.train_y, classes=self.classes, augmentation_transforms=self.train_transforms
        )
        self.valid_ds = MulticlassDataset(
            images=self.valid_x, masks=self.valid_y, classes=self.classes, augmentation_transforms=self.val_transforms
        )

        self.test_ds = MulticlassDataset(
            images=self.test_x, masks=self.test_y, classes=self.classes, augmentation_transforms=self.val_transforms
        )

        self.train_dl = DataLoader(
            dataset=self.train_ds, batch_size=self.batch_size, shuffle=True
        )
        self.valid_dl = DataLoader(
            dataset=self.valid_ds, batch_size=self.batch_size, shuffle=True
        )

        self.test_dl = DataLoader(dataset=self.test_ds, batch_size=self.batch_size, shuffle=False)

        self.pin_memory = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sm.DeepLabV3Plus(classes=len(self.classes), encoder_weights=None).to(self.device)

        self.default_learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.default_learning_rate)
        self.scaler = torch.amp.GradScaler(self.device)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        self.focal_loss_fn = sm.losses.FocalLoss(mode="multiclass", alpha=0.25)
        self.dice_loss_fn = sm.losses.DiceLoss(mode="multiclass", from_logits=True, smooth = 1e-7)
       

    def get_sample(self):
        train_sample = next(iter(self.train_dl))
        val_sample = next(iter(self.valid_dl))

        return train_sample, val_sample
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def check_accuracy(self, loader, model, dice_fn, focal_fn, split: str = "validation"):
        _dice_scores = []
        _jaccard_scores = []
        _losses = []
        model.eval()

        with torch.no_grad():
            for _, (x, y) in tqdm(enumerate(loader), total=len(loader)):
                x = x.float().to(device=self.device)
                y = y.float().long().to(device=self.device)
                
                predictions = model(x)
                dice_loss = dice_fn(predictions, y)
                focal_loss = focal_fn(predictions, y)
                total_loss = dice_loss + (1 * focal_loss)
                _losses.append(total_loss)

                dice_score = self.dice_scorer(predictions, y)
                _dice_scores.append(dice_score)

                jaccard_score = self.jaccard_scorer(predictions, y)
                _jaccard_scores.append(jaccard_score)
                
                self.dice_scorer.reset()
                self.jaccard_scorer.reset()

        _loss = torch.mean(torch.tensor(_losses))
        _dice_score = torch.mean(torch.tensor(_dice_scores))
        _jaccard_score = torch.mean(torch.tensor(_jaccard_scores))
        
        print(f"{split} Loss: {_loss}, {split} Dice-Score: {_dice_score}, Jaccard-Index: {_jaccard_score}")
        return _loss.item(), _dice_score.item(), _jaccard_score.item()


    def save_checkpoint(self, state, filename="model_save.pth"):
        print(f"Saving Checkpoint to: {filename}")
        torch.save(state, filename)


    def save_training_history(
        self,
        train_losses: list[float],
        train_scores: list[float],
        val_losses: list[float],
        val_dice_score: float,
        val_jaccard_score: float,
        test_dice_score: float,
        test_jaccard_score: float,
        out_file: str,
    ) -> None:
        print(f"Saving Training History.... {out_file}")

        with open(f"{out_file}", "w") as f:
            f.write(f"Train Losses: {str(train_losses)}")
            f.write(f"Train Dice Scores: {str(train_scores)}")
            f.write(f"Validation Losses: {str(val_losses)}")
            f.write(f"Validation Dice Score: {str(val_dice_score)}")
            f.write(f"Validation Jaccard Score: {str(val_jaccard_score)}")
            f.write(f"Test Dice Score: {str(test_dice_score)}")
            f.write(f"Test Jaccard Score: {str(test_jaccard_score)}")



    def save_train_data(self):
        train_imgs = os.path.join(self.output_path, "train_images.txt")
        val_imgs = os.path.join(self.output_path, "val_images.txt")
        test_imgs = os.path.join(self.output_path, "test_images.txt")

        with open(train_imgs, "w") as f:
            for entry in self.train_x:
                f.write(f"{entry}\n")

        with open(val_imgs, "w") as f:
            for entry in self.valid_x:
                f.write(f"{entry}\n")

        with open(test_imgs, "w") as f:
            for entry in self.test_x:
                f.write(f"{entry}\n")

    def save_model_config(self, filename: str):
        model_config = {
            "model": f"{filename}.pth",
            "architecture": self.network,
            "classes": self.classes,
            "input_width": str(self.image_width),
            "input_height": str(self.image_height),
            "batch_size": str(self.batch_size),
        }

        out_file = os.path.join(self.output_path, "model_config.json")
        json_out = json.dumps(model_config, ensure_ascii=False, indent=2)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(json_out)

        print(f"Saved model config to: {out_file}")
        

    def export2onnx(
        self, model, model_name: str, mode: str = "cpu", opset: int = 16
    ) -> None:
        
        if self.device == "cuda":
            model.eval()
            model_input = torch.randn(
                [len(self.classes), 3, self.image_height, self.image_width], device=self.device
            )

            torch.onnx.export(
                model,
                model_input,
                f"{self.output_path}/{model_name}_gpu.onnx",
                export_params=True,
                opset_version=opset,
                verbose=True,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        else:
            # TODO: just add cpu export
            print("Skipping onnx export since model is not on the GPU")

        print("Finished Training!")


    def train_model(self, loader, model, optimizer, focal_loss_fn, dice_loss_fn, scaler):
        loop = tqdm(loader)
        epoch_loss = []
        epoch_score = []
        model.train()

        for _, (x, y) in enumerate(loop): 
            x = x.float().to(device=self.device)
            y = y.float().long().to(device=self.device)

            # forward
            with torch.autocast(self.device):
                predictions = model(x)
                focal_loss = focal_loss_fn(predictions, y)
                dice_loss = dice_loss_fn(predictions, y)
                total_loss = dice_loss + (1 * focal_loss)
                epoch_loss.append(total_loss)
                dice_score = self.dice_scorer(predictions, y)
                epoch_score.append(dice_score)
            
            # backward
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update loop
            loop.set_postfix(loss=total_loss.item())
            self.dice_scorer.reset()

        epoch_score = torch.mean(torch.tensor(epoch_score))
        epoch_loss = torch.mean(torch.tensor(epoch_loss))
        print(f"Train Loss: {epoch_loss}, Dice-Score: {epoch_score}")

        return epoch_loss.item(), epoch_score.item()



    def train(self, epochs: int = 10, patience: int = 3, model_name: str = "segmentation_model"):
        torch.cuda.empty_cache()

        train_loss_history = []
        val_loss_history = []

        train_score_history = []
        val_dice_score_history = []
        val_jaccard_score_history = []
        best_val_score = 0

        max_patience = patience
        current_patience = patience

        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}/{epochs}...")

            train_loss, train_score = self.train_model(
                loader=self.train_dl,
                model=self.model,
                optimizer=self.optimizer,
                focal_loss_fn=self.focal_loss_fn,
                dice_loss_fn=self.dice_loss_fn,
                scaler=self.scaler,
            )
            
            train_score_history.append(train_score)
            train_loss_history.append(train_loss)

            val_loss, val_dice_score, val_jaccard_score = self.check_accuracy(self.valid_dl, self.model, self.dice_loss_fn, self.focal_loss_fn)
            val_dice_score_history.append(val_dice_score)
            val_jaccard_score_history.append(val_jaccard_score)
            val_loss_history.append(val_loss)

            if val_dice_score > best_val_score:
                best_val_score = val_dice_score

                current_patience = max_patience
                checkpoint = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
                self.save_checkpoint(checkpoint, filename=f"{self.output_path}/{model_name}.pth")

            else:
                current_patience -= 1

                if current_patience == 0:
                    print("Early stopping training...")
                    _, test_dice_score, test_jaccard_score = self.check_accuracy(
                        self.test_dl,
                        self.model,
                        self.dice_loss_fn,
                        self.focal_loss_fn,
                        split="Test")

                    self.save_training_history(
                        train_losses=train_loss_history,
                        train_scores=train_score_history,
                        val_losses=val_loss_history,
                        val_dice_score=val_dice_score_history,
                        val_jaccard_score=val_jaccard_score_history,
                        test_dice_score=test_dice_score,
                        test_jaccard_score=test_jaccard_score,
                        out_file=f"{self.output_path}/{model_name}.txt"
                    )

                    self.save_model_config(filename=model_name)

                    if self.export_onnx:
                        self.export2onnx(self.model, model_name=model_name)

                    return

            self.scheduler.step()

        _, test_dice_score, test_jaccard_score = self.check_accuracy(self.test_dl, self.model, self.dice_loss_fn, self.focal_loss_fn)
  
        self.save_training_history(
            train_losses=train_loss_history,
            train_scores=train_score_history,
            val_losses=val_loss_history,
            val_dice_score=val_dice_score_history,
            val_jaccard_score=val_jaccard_score_history,
            test_dice_score=test_dice_score,
            test_jaccard_score=test_jaccard_score,
            out_file=f"{self.output_path}/{model_name}.txt"
        )

        self.save_model_config(filename=model_name)

        if self.export_onnx:
            self.export2onnx(self.model, model_name=model_name)