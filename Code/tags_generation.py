import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import onnx
import onnxruntime

import paths, dataset, tags

class TagsGeneration:
        def __init__(self):
            self.features_extractor = onnx.load(paths.model_directory_path + 'extractor.onnx')
            onnx.checker.check_model(self.features_extractor)
            self.tags_classifier = onnx.load(paths.model_directory_path + 'classifier.onnx')
            onnx.checker.check_model(self.tags_classifier)

            self.train_tags_loader = self.get_data_tags()

        def get_data_tags(self):
            # Load dataloaders for train and validation tags
            train_data_loader = dataset.get_tags_loader(paths.data_directory_path + 'train_tags.txt', self.transform_train_images())
            return train_data_loader

        def transform_train_images(self):
            # Transforms for train images
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),])
            return transform
        
        def to_numpy(self, tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        def generate_tags(self):
            # Generates tags using the finetuned extractor and classifier models which are then used to finetune the report generation model
            extractor_session = onnxruntime.InferenceSession(paths.model_directory_path + 'extractor.onnx')
            classifier_session = onnxruntime.InferenceSession(paths.model_directory_path + 'classifier.onnx')

            image_tags_df = pd.DataFrame(columns=['image_path', 'pred_tags'])

            for batch, data in enumerate(self.train_tags_loader):
                for i in range(16):
                    image_id, image, tag = data[0][i], data[1][i], data[2][i]
                    print(image_id)
                    image = image.unsqueeze(0)

                    extractor_inputs = {extractor_session.get_inputs()[0].name: self.to_numpy(image)}
                    extractor_outs = extractor_session.run(None, extractor_inputs)
                    extracted_features = extractor_outs[0]
                    
                    classifier_inputs = {classifier_session.get_inputs()[0].name: np.expand_dims(extracted_features, axis=0)}
                    classifier_outs = classifier_session.run(None, classifier_inputs)
                    pred_tags_prob = classifier_outs[0]

                    pred_tags_list = tags.Tag().array_to_tags(torch.topk(torch.Tensor(pred_tags_prob), 3)[1].numpy())
                    pred_tags = ' '.join(pred_tags_list)
                    image_tags_df.loc[len(image_tags_df)] = [paths.images_directory_path + image_id, pred_tags]
            
            image_tags_df.to_csv(paths.data_directory_path + 'pred_train_tags.csv', index=False)



    
    
    

    

    