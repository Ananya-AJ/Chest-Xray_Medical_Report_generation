import os
import re
import contractions
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

import paths

def create_image_report_df():
    '''
    Create original dataframe with image path, impression, findings and tags

    Returns:
    image_report_df(pd.DataFrame): Dataframe with required fields
    '''
    # Define columns and empty df
    cols = ['image_id', 'image_path', 'findings', 'impression']
    image_report_df = pd.DataFrame(columns=cols)
    
    for file in tqdm(os.listdir(paths.reports_directory_path)):
        report_filepath = paths.reports_directory_path + file
        if report_filepath.endswith('.xml'):
            xml_tree = ET.parse(report_filepath)
            # Extract findings and impression from report xml
            findings = xml_tree.find(".//AbstractText[@Label='FINDINGS']").text
            impression = xml_tree.find(".//AbstractText[@Label='IMPRESSION']").text

            # Get image id and set image path
            for parent_img in xml_tree.findall('parentImage'):
                parent_img_id = parent_img.attrib['id'][3:] + '.png'
                image_filepath = paths.images_directory_path + parent_img.attrib['id'] + '.png'

                # Append data to df
                image_report_df = image_report_df.append(pd.Series([parent_img_id, image_filepath, findings, impression], index=cols), ignore_index=True)

    return image_report_df
   
def clean_text(text):
    '''
    Cleans a given text by removing punctuations and unwanted characters.

    Args:
    text(str): The text to be cleaned

    Returns:
    text(str): The cleaned text
    '''
    # Convert to lower case and expand contractions
    text = text.strip().lower()
    text = contractions.fix(text)
    # Remove extra spaces and 'xxx'
    text = re.sub(' +', ' ', text)
    text = re.sub(r'x*', '', text)
    # Remove punctuations
    filters='!"\'—#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((i, ' ') for i in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    # FInally concatenate all cleaned tokens and keep words greater than length 2
    text = ' '.join([w for w in text.split() if len(w)>2 or w in ['no', 'ct']])
    
    return text