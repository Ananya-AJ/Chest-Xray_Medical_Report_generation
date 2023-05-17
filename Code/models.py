import torch
import torch.nn as nn
import torchvision.models as models
from transformers import (
    BlipForConditionalGeneration,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)   

from config import *
import paths    

class ScanVisualFeatures(nn.Module):
    def __init__(self):
        super(ScanVisualFeatures, self).__init__()
        self.model, self.features, self.avg_pool = self.get_model()

    def get_model(self):
        '''
        Model for extracting visual features from chest x-ray image

        Returns:
        model(nn.Sequential): visual feature extractor model
        features(int): Number of features to fully connecte dlayer in densenet201
        avg_pool(nn.AvgPool2d): Average pooling layer 
        '''
        # Extract convolutional layers from densenet121
        densenet = models.densenet121(weights='DEFAULT')
        layers = list(densenet.features)
        # Extract number of input features to FC layer in densenet
        features = densenet.classifier.in_features

        # Create a sequential model with the extracted convolutional layers
        model = nn.Sequential(*layers)
        # Apply average pooling
        avg_pool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        return model, features, avg_pool

    def forward(self, image):
        '''
        Forward pass through densenet121 for extracting visual features

        Args:
        image(tensor): a tensor of chest x-ray image

        Returns:
        out_features(torch.Tensor): average pooled features of the image
        '''
        # Get features from last convolutional layer
        features = self.model(image)
        # Apply average pooling
        out_features = self.avg_pool(features).squeeze()

        return out_features


class TagsClassifier(nn.Module):
    def __init__(self, classes, in_features_dim):
        '''
        Single layer linear classifier for tags classification

        Args:
        classes(int): Number of output classes
        in_features_dim(int): Number of input features to classification layer
        '''
        super(TagsClassifier, self).__init__()
        self.classifier = nn.Linear(in_features=in_features_dim, out_features=classes)
        self.activation = nn.Softmax()
        self.init_weight()

    def init_weight(self):
        # Initialize weights of linear layer in classification model
        self.classifier.weight.data.uniform_()
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        '''
        Forward pass through classificaiton model

        Args:
        avg_features(troch.Tensor): a tensor of average pooled features of image

        Returns:
        predicted_tags_prob(torch.Tensor): A tensor representing the probabilities of the classes for given input image features
        '''
        pred_prob = self.activation(self.classifier(avg_features))
        return pred_prob


class ReportGeneration():
    def __init__(self):
        # Initialize conditional generation model for report generation
        self.model = BlipForConditionalGeneration.from_pretrained('nathansutton/generate-cxr')

    def finetuning(self, report_tokenizer, report_processor, train, valid):
        '''
        Fine tune model on our chest x-ray dataset

        Args:
        report_tokenizer(blip tokenizer): tokenizer of blipprocessor
        report_processor(blip processor): blip processor
        train(torch.Dataset): training dataset
        valid(torch.Dataset): validation dataset

        Returns:
        trainer(Trainer): Fine tuned trainer object
        '''
        # Define training arguments for fine tuning
        training_args = TrainingArguments(
            num_train_epochs=5,
            evaluation_strategy='epoch',
            per_device_eval_batch_size=16,
            per_device_train_batch_size=16,
            lr_scheduler_type='cosine_with_restarts',
            warmup_ratio=0.1,
            learning_rate=1e-3,
            save_total_limit=1,
            output_dir=paths.logs_directory_path + 'report_generation'
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=report_tokenizer,
            mlm=False
        )
        # Define trainer object for fine tuning blip conditional generation model
        trainer = Trainer(
            args=training_args,
            train_dataset=train,
            eval_dataset=valid,
            data_collator=data_collator,
            tokenizer=report_processor.tokenizer,
            model=self.model
        )
        # Train model
        trainer.train()

        return trainer

