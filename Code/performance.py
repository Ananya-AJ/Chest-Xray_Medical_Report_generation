from PIL import Image
import torch
import torchvision.transforms as transforms
import json
import paths, models
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import onnx
import onnxruntime

import metrics, tags

class PerformanceEvaluation:
        def __init__(self):
            self.features_extractor = onnx.load(paths.model_directory_path + 'extractor.onnx')
            onnx.checker.check_model(self.features_extractor)
            self.tags_classifier = onnx.load(paths.model_directory_path + 'classifier.onnx')
            onnx.checker.check_model(self.tags_classifier)

        def transform_image(self, image):
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])
            image = transform(image)
            return image

        def get_data(self, data_file, tags_file):
            with open(paths.data_directory_path + data_file, 'r') as f:
                test_data = json.load(f) 
            data = {record['image_id']: record for record in test_data}

            tags = defaultdict(str)
            with open(paths.data_directory_path + tags_file, 'r') as f:
                for line in f:
                    items = line.split()
                    image_id = items[0]
                    label = items[1:]
                    label = [int(i) for i in label]
                    tags[image_id] = label

            return data, tags
        
        '''def tags_accuracy(self, data, tags_, train=True):
            # Select a subset of 300 records
            keys = list(data.keys())
            keys_ = ['CXR1005_IM-0006-1001.png', 'CXR1007_IM-0008-1001.png', 'CXR1009_IM-0010-1001.png']
            #random_keys = random.sample(keys, 2)
            data_subset = {k: data[k] for k in keys_}

            batch_hamming = 0
            df = pd.DataFrame(columns=['image_path', 'tags'])

            for k, v in data_subset.items():
                image_path = v['image_path']
                image_id = v['image_id']
                print(image_id)

                image = Image.open(image_path).convert('RGB')
                image = self.transform_image(image)
                image = image.unsqueeze(0)
                print(image)

                visual_features = self.features_extractor.forward(image)
                print(visual_features)
                pred_tags_prob = self.tags_classifier.forward(visual_features)
                print('************', pred_tags_prob)
                gt_tags = np.array(tags_[image_id])
                #print(gt_tags)
                #print(metrics.hamming(gt_tags, pred_tags_prob))
                batch_hamming += metrics.hamming(gt_tags, pred_tags_prob)

                if train:
                    predicted_tags_list = tags.Tag().array_to_tags(torch.topk(pred_tags_prob, 3)[1].cpu().detach().numpy())
                    pred_tags = ' '.join(predicted_tags_list)
                    df.loc[len(df)] = [image_path, pred_tags]
            if train:
                df.to_csv(paths.data_directory_path + 'pred_tags.csv', index=False)

            return batch_hamming/10'''
        
        #def evaluate_report_bleu(self):

        def to_numpy(self, tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        def run_onnx_model(self, data):
            keys = list(data.keys())
            keys_ = ['CXR1005_IM-0006-1001.png', 'CXR1007_IM-0008-1001.png', 'CXR1009_IM-0010-1001.png', 'CXR1_1_IM-0001-3001.png']
            #random_keys = random.sample(keys, 2)
            data_subset = {k: data[k] for k in keys_}

            for k, v in data_subset.items():
                image_path = v['image_path']
                image_id = v['image_id']
                print(image_id)

                image = Image.open(image_path).convert('RGB')
                image = self.transform_image(image)
                image = image.unsqueeze(0)
                extractor_session = onnxruntime.InferenceSession(paths.model_directory_path + 'extractor.onnx')
                extractor_inputs = {extractor_session.get_inputs()[0].name: self.to_numpy(image)}
                extractor_outs = extractor_session.run(None, extractor_inputs)
                img_out_y = extractor_outs[0]
                print(type(img_out_y))
                print(img_out_y)

                classifier_session = onnxruntime.InferenceSession(paths.model_directory_path + 'classifier.onnx')
                classifier_inputs = {classifier_session.get_inputs()[0].name: np.expand_dims(img_out_y, axis=0)}
                classifier_outs = classifier_session.run(None, classifier_inputs)
                tags_out_y = classifier_outs[0]
                print(tags_out_y)


if __name__ == '__main__':
    eval = PerformanceEvaluation()

    train_data, train_tags = eval.get_data('train_data.json', 'train_tags.txt')
    #train_accuracy = eval.tags_accuracy(train_data, train_tags, train=True)
    eval.run_onnx_model(train_data)
    #print('Tags prediction accuracy on training set:', train_accuracy)

    # val_data, val_tags = eval.get_data('val_data.json', 'val_tags.txt')
    # val_accuracy = eval.tags_accuracy(val_data, val_tags, train=False)
    # print('Tags prediction accuracy on validation set:', val_accuracy)



    
    
    

    

    