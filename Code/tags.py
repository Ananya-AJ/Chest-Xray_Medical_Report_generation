import re
import pandas as pd

import paths

class Tag(object):
    def __init__(self):
        self.original_df = pd.read_csv(paths.data_directory_path + 'original_dataset.csv')
        self.unique_tags = set()
        self.tags = self.load_tags()
        self.id_to_tags = self.load_id_to_tags()

    def load_id_to_tags(self):
        id_to_tags = {}
        # Convert id to tags using global tags list and create dicitonary
        for i, tag in enumerate(self.tags):
            id_to_tags[i] = tag
        return id_to_tags
    
    def array_to_tags(self, array):
        tags = []
        # Convert predicted id to tags 
        for id in array[0]:
            tags.append(self.id_to_tags[id])
        return tags

    def load_tags(self):
        tags = ['cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', 'foreign body', 'dish', 'aorta, thoracic', 
                'atherosclerosis', 'histoplasmosis', 'hypoventilation', 'catheterization, central venous', 'pleural effusions', 'pleural effusion', 
                'callus', 'sternotomy', 'lymph nodes', 'tortuous aorta', 'stent', 'interstitial pulmonary edema', 'cholecystectomies', 'neoplasm', 
                'central venous catheter', 'pneumothorax', 'metastatic disease', 'vena cava, superior', 'cholecystectomy', 'scoliosis', 
                'subcutaneous emphysema', 'thoracolumbar scoliosis', 'spinal osteophytosis', 'pulmonary fibroses', 'rib fractures', 'sarcoidosis', 
                'eventration', 'fibrosis', 'spine', 'obstructive lung disease', 'pneumonitis', 'osteopenia', 'air trapping', 'demineralization', 
                'mass lesion', 'pulmonary hypertension', 'pleural diseases', 'pleural thickening', 'calcifications of the aorta', 'calcinosis', 
                'cystic fibrosis', 'empyema', 'catheter', 'lymph', 'pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease', 
                'chronic obstructive pulmonary disease', 'rib', 'clip', 'aortic ectasia', 'shoulder', 'scarring', 'scleroses', 'adenopathy', 'emphysemas',
                'pneumonectomy', 'infection', 'aspiration', 'bilateral pleural effusion', 'bulla', 'lumbar vertebrae', 'lung neoplasms', 'lymphadenopathy', 
                'hyperexpansion', 'ectasia', 'bronchiectasis', 'nodule', 'pneumonia', 'right-sided pleural effusion', 'osteoarthritis', 
                'thoracic spondylosis', 'picc', 'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters', 'emphysema', 'trachea', 
                'surgery', 'cervical spine fusion', 'hypertension, pulmonary', 'pneumoperitoneum', 'scar', 'atheroscleroses', 'aortic calcifications', 
                'volume overload', 'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', 'copd', 'kyphoses', 'spinal fractures', 'fracture', 
                'clavicle', 'focal atelectasis', 'collapse', 'thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema', 
                'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', 'humerus', 'heart failure', 'normal', 'coronary artery bypass', 
                'pulmonary atelectasis', 'lung diseases, interstitial', 'pulmonary disease, chronic obstructive', 'opacity', 'deformity', 
                'chronic disease', 'pleura', 'aorta', 'tuberculoses', 'hiatal hernia', 'scolioses', 'pleural fluid', 'malignancy', 'kyphosis', 
                'bronchiectases', 'congestion', 'discoid atelectasis', 'nipple', 'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 
                'arthritic changes', 'pulmonary edema', 'vascular calcification', 'sclerotic', 'central venous catheters', 'catheterization', 
                'hydropneumothorax', 'aortic valve', 'hyperinflation', 'prostheses', 'pacemaker, artificial', 'bypass grafts', 'pulmonary fibrosis', 
                'multiple myeloma', 'postoperative period', 'cabg', 'right lower lobe pneumonia', 'granuloma', 'degenerative change', 'atelectasis', 
                'inflammation', 'effusion', 'cicatrix', 'tracheostomy', 'aortic diseases', 'sarcoidoses', 'granulomas', 'interstitial lung disease', 
                'infiltrates', 'displaced fractures', 'chronic lung disease', 'picc line', 'intubation, gastrointestinal', 'lung diseases', 
                'multiple pulmonary nodules', 'intervertebral disc degeneration', 'pulmonary emphysema', 'spine curvature', 'fibroses', 
                'chronic granulomatous disease', 'degenerative disease', 'atelectases', 'ribs', 'pulmonary arterial hypertension', 'edemas', 
                'pectus excavatum', 'lung granuloma', 'plate-like atelectasis', 'enlarged heart', 'hilar calcification', 'heart valve prosthesis', 
                'tuberculosis', 'old injury', 'patchy atelectasis', 'histoplasmoses', 'exostoses', 'mastectomies', 'right atrium', 'large hiatal hernia', 
                'hernia, hiatal', 'aortic aneurysm', 'lobectomy', 'spinal fusion', 'spondylosis', 'ascending aorta', 'granulomatous infection', 
                'fractures, bone', 'calcified granuloma', 'degenerative joint disease', 'intubation, intratracheal', 'others']
        
        return tags

    
    def build_tags(self, image_id_list_path, output_tags_path):
        '''
        Build tags .txt file 

        Args:
        image_id_list_path(str): Path to file containing list of image ids
        output_tags_path(str): Path to file for storing one hot encoded tags
        '''
        with open(image_id_list_path, 'r') as input:
            with open(output_tags_path, 'w') as output:
                for each in input.readlines():
                    key = each.strip() + '.png'
                    value = self.original_df[self.original_df.image_id == key]['tags']
                    if len(value) > 0:
                        values = value.values[0]
                    output.write(key)
                    for tag in self.tags:
                        # Write 1 for tags that occur in our global tag list, else write 0
                        if tag in values:
                            output.write(' 1')
                        else:
                            output.write(' 0')
                    output.write('\n')