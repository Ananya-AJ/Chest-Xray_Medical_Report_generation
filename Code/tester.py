from collections import defaultdict
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import BlipForConditionalGeneration, BlipProcessor
import onnx
import onnxruntime

import paths, tags, metrics

class Inference():
    def __init__(self):
        '''
        Load ground truth data as well as all fine tuned models
        '''
        self.tags = self.load_true_tags()
        self.reports_dict = self.load_true_reports()

        # Initialize empty report for chatbot
        self.predicted_report = ''

        # Load pretrained model and check model
        features_extractor = onnx.load(paths.model_directory_path + 'extractor.onnx')
        onnx.checker.check_model(features_extractor)
        tags_classifier = onnx.load(paths.model_directory_path + 'classifier.onnx')
        onnx.checker.check_model(tags_classifier)

        # Create inference session
        self.extractor_session = onnxruntime.InferenceSession(paths.model_directory_path + 'extractor.onnx')
        self.classifier_session = onnxruntime.InferenceSession(paths.model_directory_path + 'classifier.onnx')

        # Load report generation model
        #self.reports_tokenizer = BlipProcessor.from_pretrained(paths.model_directory_path + 'report_gen').tokenizer
        self.reports_processor = BlipProcessor.from_pretrained(paths.model_directory_path + 'report_gen')
        self.reports_model = BlipForConditionalGeneration.from_pretrained(paths.model_directory_path + 'report_gen')

    def load_true_tags(self):
        '''
        Load ground truth tags from .txt file

        Returns:
        tags(dict): List of ground truth tags for all images in test
        '''
        tags = defaultdict(str)
        with open(paths.data_directory_path + 'test_tags.txt', 'r') as f:
          for line in f:
            items = line.split()
            image_id = items[0] + '.png'
            label = items[1:]
            label = [int(i) for i in label]
            tags[image_id] = label
        return tags

    def load_true_reports(self):
        '''
        Load ground truth reports (findings and impressions)

        Returns:
        (dict): Image id along with report for that image
        '''
        with open(paths.data_directory_path + 'test_data.json', 'r') as f:
            test_data = json.load(f) 
        return {record['image_id']: record for record in test_data}

    def transform_test_image(self, image):
        '''
        Transform test images as required by tags model

        Args:
        image(ndarray): Test image to transform

        Returns:
        transformed_image: Transformed test image
        '''
        transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])
        transformed_image = transform(image)
        # Add batch dimension
        transformed_image = transformed_image.unsqueeze(0)
        return transformed_image

    def get_true_data(self, image_id):
        '''
        Load ground truth reports (findings and impressions)

        Args:
        image(ndarray): Test image to transform

        Returns:
        transformed_image: Transformed test image
        '''
        # Load ground truth findings and impressions
        record = self.reports_dict[image_id]
        findings = record['findings']
        impression = record['impression']
        tags = np.array(self.tags[image_id])
        return tags, impression, findings
    
    def evaluate_metrics(self, image_id, pred_tags_prob, pred_impression, pred_findings):
        '''
        FUnction to evaluate metrics on report

        Args:
        image_id(str): image_id of test image
        pred_tags_prob(torch.Tensor): Probability of tags as determined by tags model
        pred_impression(str): Predicted impression
        pred_findings(str): Predicted findings

        Returns:
        hamming(float): Hamming loss on tags
        impressions_similarity(float): semantic similarity between predicted and actual impressions
        findings_similarity(float): semantic similarity between predicted and actual findings
        '''
        tags, impression, findings = self.get_true_data(image_id)
        active_tags = np.count_nonzero(tags == 1)
        hamming = metrics.hamming(tags, pred_tags_prob, active_tags)
        print("length of tags:",active_tags)
        impressions_similarity = metrics.semantic_similarity(pred_impression, impression)
        findings_similarity = metrics.semantic_similarity(pred_findings, findings)

        return hamming, impressions_similarity, findings_similarity
    
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    async def generate_report(self, image, image_id):
        '''
        Generate report control function

        Args:
        image(ndarray): Test image to transform

        Returns:
        pred_tags_prob(torch.Tensor): Probability of tags as determined by tags model
        predicted_tags(str): String of predcited tags for given test image
        predicted_report(str): Report generated by report generation model
        '''
        # Transform test image
        transformed_image = self.transform_test_image(image)
        gt_tags, impression, findings = self.get_true_data(image_id)
        active_tags = np.count_nonzero(gt_tags == 1)

        # Extract visual features and run tags classifier
        extractor_inputs = {self.extractor_session.get_inputs()[0].name: self.to_numpy(transformed_image)}
        extractor_outs = self.extractor_session.run(None, extractor_inputs)
        extracted_features = extractor_outs[0]
        
        classifier_inputs = {self.classifier_session.get_inputs()[0].name: np.expand_dims(extracted_features, axis=0)}
        classifier_outs = self.classifier_session.run(None, classifier_inputs)
        pred_tags_prob = classifier_outs[0]
        print("active_tags",active_tags)
        # Convert tag ids to tags
        predicted_tags_list = tags.Tag().array_to_tags(torch.topk(torch.Tensor(pred_tags_prob), active_tags)[1].cpu().detach().numpy())
        predicted_tags = ' '.join(predicted_tags_list)

        # Report generation model
        inputs = self.reports_processor(images=image, text='indication:' + predicted_tags, return_tensors='pt')
        # Predict/ generate and decode the report
        output = self.reports_model.generate(**inputs, max_length=512)
        self.predicted_report = self.reports_processor.decode(output[0], skip_special_tokens=True)
        
        return pred_tags_prob, predicted_tags, self.predicted_report