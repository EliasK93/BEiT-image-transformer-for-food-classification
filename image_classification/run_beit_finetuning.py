from typing import Callable
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BeitImageProcessor
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from transformers import BeitForImageClassification
from transformers import TrainingArguments, Trainer

"""
Note: Image transforms and model configurations are largely based on this notebook:
https://github.com/MohammadRoodbari/Image-Classification/blob/main/main.ipynb
"""


def prepare_dataset(raw_dataset: DatasetDict, processor: BeitImageProcessor) -> tuple[Dataset, Dataset, Dataset, Callable]:
    """
    Split the dataset into train, validation and test set, apply image transform and return Datasets plus a DataCollator.
    """

    # configure transforms
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    transform_train = Compose([RandomResizedCrop(processor.size["width"]), RandomHorizontalFlip(), ToTensor(), normalize])
    transform_val = Compose([Resize(processor.size["width"]), CenterCrop(processor.size["width"]), ToTensor(), normalize])

    def train_transforms(examples):
        examples['pixel_values'] = [transform_train(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [transform_val(image.convert("RGB")) for image in examples['image']]
        return examples

    # load and split dataset
    splits = raw_dataset["train"].train_test_split(test_size=0.8, seed=1)
    train_dataset = splits["train"]
    splits = splits["test"].train_test_split(test_size=0.5, seed=1)
    val_dataset = splits["train"]
    test_dataset = splits["test"]

    # apply transforms
    train_dataset.set_transform(train_transforms)
    val_dataset.set_transform(val_transforms)
    test_dataset.set_transform(val_transforms)

    # define data collator
    def data_collator(examples):
        return {"pixel_values": torch.stack([example["pixel_values"] for example in examples]),
                "labels": torch.tensor([example["label"] for example in examples])}

    return train_dataset, val_dataset, test_dataset, data_collator


def load_model(model_id: str, labels: list[str]) -> tuple[BeitImageProcessor, BeitForImageClassification]:
    """
    Load image processor and pre-trained model.
    """

    # load processor
    processor = BeitImageProcessor.from_pretrained(pretrained_model_name_or_path=model_id)

    # load model
    model = BeitForImageClassification.from_pretrained(
        pretrained_model_name_or_path=model_id, id2label={id: label for id, label in enumerate(labels)},
        label2id={label: id for id, label in enumerate(labels)}, ignore_mismatched_sizes=True)

    return processor, model


if __name__ == "__main__":

    # load raw dataset
    raw_dataset = load_dataset("data/food-101/images")

    # define labels
    labels = raw_dataset["train"].features["label"].names

    for model_id in ["microsoft/beit-base-patch16-224-pt22k",
                     "microsoft/beit-base-patch16-224",
                     "microsoft/beit-base-patch16-224-pt22k-ft22k"]:

        # load processor and model
        processor, model = load_model(model_id, labels=labels)

        # split and transform dataset
        train_dataset, val_dataset, test_dataset, collate_fn = prepare_dataset(raw_dataset, processor)

        # run training
        args = TrainingArguments(output_dir=f"models/{model_id.replace('/', '_')}", save_strategy="epoch", eval_strategy="epoch",
                                 learning_rate=0.00002, per_device_train_batch_size=16, per_device_eval_batch_size=16,
                                 num_train_epochs=3, weight_decay=0.01, load_best_model_at_end=True,
                                 metric_for_best_model="accuracy", remove_unused_columns=False, seed=1)
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
                          data_collator=collate_fn, tokenizer=processor,
                          compute_metrics=lambda x: dict(accuracy=accuracy_score(np.argmax(x[0], axis=1), x[1])))
        trainer.train()

        # evaluate
        outputs = trainer.predict(test_dataset)
        print(f"Accuracy on test set: {outputs.metrics['test_accuracy']}")
