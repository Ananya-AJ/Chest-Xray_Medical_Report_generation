import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import paths

class ChestScanTagsLoader(Dataset):
    def __init__(self, tags_path, transform):
        self.file_name, self.tags = self.load_tags(tags_path)
        self.transform = transform

    def load_tags(self, tags_path):
        '''
        Loads tags from .txt file 
        
        Args:
        tags_path(str): Path of the file where tags are stored

        Returns:
        tags_list(list): List of tags for an image
        '''
        tags_list = []
        filename_list = []
        # Open and read tags form .txt file
        with open(tags_path, 'r') as f:
            for line in f:
                items = line.split()
                image_id = items[0]
                label = items[1:]
                # Get list of tags in one-hot encoded form
                label = [int(i) for i in label]
                # Store image ids in list
                filename_list.append(image_id)
                tags_list.append(label)
        return filename_list, tags_list
    
    def __getitem__(self, index):
        '''
        Returns a record of an image with required details to load into dataloader

        Args:
        index(int): Index of record to be fetched to load into dataloader

        Returns:
        image_id(str): Image Id of image being loaded
        image(ndarray): Image pixels
        tags(list): List of tags associated with the image being loaded
        '''
        image_id = self.file_name[index] + '.png'
        # Open image and convert to RGB
        image = Image.open(os.path.join(paths.images_directory_path, image_id)).convert('RGB')
        # Apply transformations to image
        image = self.transform(image)
        # Retrieve tags for the image
        tags = self.tags[index]

        return image_id, image, tags
           
    def __len__(self):
        return len(self.file_name)
    

class ChestScanReportLoader(Dataset):
    def __init__(self, df, report_tokenizer, report_processor):
        self.df = df
        self.report_tokenizer = report_tokenizer
        self.report_processor = report_processor
    
    def __getitem__(self, idx):
        '''
        Returns a record of an image with required fields for report generation model.

        Args:
        idx(int): Index of record to be fetched to load into dataset

        Returns:
        encoding(dict): Dictionary with image and corresponding report text needed for fine tuning report generation model
        '''
        row = self.df.iloc[idx]
        # Open image and convert to RGB
        image = Image.open(row['image_path']).convert('RGB')
        # Text for finetuning
        report = row['impression'] + row['findings'] + row['tags']
        # Process image and report with blip processor
        encoding = self.report_processor(images=image, text=report, padding='max_length', return_tensors='pt')
        # Convert to dictionary and remove dimensions
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding
            
    def __len__(self):
        return len(self.df)
    

def collate(data):
    '''
    Create image_id, image and tags lists

    Args:
    data(list): List of tuples containing image_id, image and tags

    Returns:
    image_id(str): image_id of image
    image(torch.Tensor): image tensor
    tags(torch.Tensor): tags tensor
    '''
    # Uzips tuple into three lists
    image_id, image, tags = zip(*data)
    # Create batch of images by stacking
    image = torch.stack(image, 0)

    return image_id, image, torch.Tensor(tags)

def get_tags_loader(tags_path, transform):
    '''
    Dataloader for image and tags for tags classification model

    Args:
    tags_path(srt): Path of file containing tags
    transform(function): Transform function to apply to images

    Returns:
    dataloader(torch.DataLoader): Dataloader containing all required data for tags classification model
    '''
    dataset = ChestScanTagsLoader(tags_path, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=16, collate_fn=collate) 

    return data_loader

def get_reports_loader(df, report_tokenizer, report_processor):
    '''
    Get dataset for report generation model

    Args:
    report_tokenizer(blip tokenizer): tokenizer of blipprocessor
    report_processor(blip processor): blip processor

    Returns:
    dataset(torch.Dataset): dataset containing data for report generation model
    '''
    dataset = ChestScanReportLoader(df, report_tokenizer, report_processor)
    return dataset