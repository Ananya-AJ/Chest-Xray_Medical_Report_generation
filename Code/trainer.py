import os
import pandas as pd
from datetime import datetime
import torchvision.transforms as transforms
from torch.autograd import Variable
from transformers import BlipProcessor

import dataset, paths
from models import *
    

class TagsTrainerHelper():
    def __init__(self):
        '''
        Initializations of models, loss function and optimizer
        '''
        self.num_epochs = 200
        self.params = None

        self.train_tags_loader, self.val_tags_loader = self.get_data_tags()

        self.extractor = self.init_visual_features_extractor()
        self.classifier = self.init_tags_classifier()

        self.loss = self.init_loss()
        self.optimizer = self.init_optimizer()
        
        self.writer = self.init_writer()

    def init_loss(self):
        return nn.BCELoss()

    def init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=0.01)

    def transform_train_images(self):
        # Transforms for train images
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])
        return transform

    def transform_val_images(self):
        # Transforms for valid images
        transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])
        return transform

    def get_data_tags(self):
        # Load dataloaders for train and validation tags
        train_data_loader = dataset.get_tags_loader(paths.data_directory_path + 'train_tags.txt', self.transform_train_images())
        val_data_loader = dataset.get_tags_loader(paths.data_directory_path + 'val_tags.txt', self.transform_val_images())
        return train_data_loader, val_data_loader
    
    def init_visual_features_extractor(self):
        model = ScanVisualFeatures()
        if self.params:
            self.params += list(model.parameters())
        else:
            self.params = list(model.parameters())
        return model
    
    def init_tags_classifier(self):
        model = TagsClassifier(classes=210, in_features_dim=1024)
        if self.params:
            self.params += list(model.parameters())
        else:
            self.params = list(model.parameters())
        return model
    
    def train(self):
        # Training function for tags and evaluating on validation dataset
        for epoch in range(self.num_epochs):
            # Put model into training mode
            self.extractor.train()
            self.classifier.train()
            # Run train and valid epochs
            train_tags_loss = self.epoch_train()

            # Put model into eval mode (no updates to weights)
            self.extractor.eval()
            self.classifier.eval()
            val_tags_loss = self.epoch_val()

            # Write log after every epoch
            self.writer.write('Epoch {} train_tags_loss:{} - val_tags_loss:{} \n'.format(epoch, train_tags_loss, val_tags_loss))
        
        # Save model in onnx format
        self.extractor.eval()
        self.classifier.eval()
        bs = 1
        x = torch.randn(bs, 3, 224, 224, requires_grad=True).cuda()
        torch.onnx.export(self.extractor,              
                          x,                        
                          os.path.join('extractor.onnx'),   
                          export_params=True,        
                          opset_version=10,          
                          do_constant_folding=True,  
                          input_names = ['image'],   
                          output_names = ['features'], 
                          dynamic_axes={'input' : {0 : 'bs'},    
                                        'output' : {0 : 'bs'}})
        
        y = torch.randn(bs, 1024).cuda()
        torch.onnx.export(self.classifier,               
                          y,                         
                          os.path.join('classifier.onnx'),   
                          export_params=True,       
                          opset_version=10,         
                          do_constant_folding=True, 
                          input_names = ['features'],  
                          output_names = ['tags'], 
                          dynamic_axes={'input' : {0 : 'bs'},    
                                        'output' : {0 : 'bs'}})
        
    def epoch_train(self):
        raise NotImplementedError

    def epoch_val(self):
        raise NotImplementedError
    
    def init_writer(self):
        writer = open(os.path.join(paths.logs_directory_path, 'logs_{}.txt'.format(datetime.now().strftime('%m_%d_%Y_%H_%M_%S'))), 'w')
        return writer


class TagsTrainer(TagsTrainerHelper):
    def __init__(self):
        TagsTrainerHelper.__init__(self)

    def epoch_train(self):
        '''
        Training epoch for tags classification model

        Returns:
        tags_loss(float): BCE loss on training tags
        '''
        tags_loss = 0

        for _, (image_ids, images, tags) in enumerate(self.train_tags_loader):
            images = Variable(images, requires_grad=True).cuda()
            tags = Variable(tags, requires_grad=False).cuda()
            
            # Forward pass on visual feature extractor and multiclass classification
            visual_features = self.extractor.forward(images)
            predicted_tags_prob = self.classifier.forward(visual_features)

            # Calculate loss
            batch_tags_loss = self.loss(predicted_tags_prob, Variable(tags.squeeze(0), requires_grad=False)).sum()

            # Back propagation
            self.optimizer.zero_grad()
            batch_tags_loss.backward()
            self.optimizer.step()

            tags_loss += batch_tags_loss.item()

        return tags_loss/len(self.train_tags_loader)
    
    def epoch_val(self):
        '''
        Validation epoch for tags classification model

        Returns:
        tags_loss(float): BCE loss on validation tags
        '''
        tags_loss = 0

        for _, (image_ids, images, tags) in enumerate(self.val_tags_loader):
            images = Variable(images, requires_grad=True).cuda()
            tags = Variable(tags, requires_grad=False).cuda()
            
            # Forward pass on visual feature extractor and multiclass classification
            visual_features = self.extractor.forward(images)
            predicted_tags_prob = self.classifier.forward(visual_features)

            # Calculate loss
            batch_tags_loss = self.loss(predicted_tags_prob, Variable(tags, requires_grad=False)).sum()
            tags_loss += batch_tags_loss.item()

        return tags_loss/len(self.val_tags_loader)


class ReportsTrainerHelper():
    def __init__(self):
        self.report_processor = BlipProcessor.from_pretrained('nathansutton/generate-cxr')
        self.report_tokenizer = BlipProcessor.from_pretrained('nathansutton/generate-cxr').tokenizer
        self.train_report_loader, self.val_report_loader = self.get_data_reports()
        self.report_generation = self.init_report_generation()

    def get_data_reports(self):
        '''
        Load train and validation dataste for report generation model

        Returns:
        train_loader(torch.Dataset): training dataset for report generation mdoel
        val_loader(torch.Dataset): validation dataset for report generation mdoel
        '''
        # Load train dataset
        train_df = pd.read_json(paths.data_directory_path + 'train_data.json')
        train_data_loader = dataset.get_reports_loader(train_df, self.report_tokenizer, self.report_processor)
        # Load validation dataset
        val_df = pd.read_json(paths.data_directory_path + 'val_data.json')
        val_data_loader = dataset.get_reports_loader(val_df, self.report_tokenizer, self.report_processor)

        return train_data_loader, val_data_loader
    
    def init_report_generation(self):
        model = ReportGeneration()
        return model


class ReportsTrainer(ReportsTrainerHelper):
    def __init__(self):
        ReportsTrainerHelper.__init__(self)

    def train(self):
        '''
        Training function for report generation
        '''
        # Finetuning function to finetune generate-cxr model
        finetuned_model = self.report_generation.finetuning(self.report_tokenizer, self.report_processor, self.train_report_loader, self.val_report_loader)
        # Save fine tuned model
        finetuned_model.save_model(paths.model_directory_path + 'report_generation')
