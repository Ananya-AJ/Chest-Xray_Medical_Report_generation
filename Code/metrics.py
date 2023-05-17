import numpy as np
import spacy
from sklearn.metrics import hamming_loss
import spacy

import data_preprocessing

emb = spacy.load("en_core_web_md")

def clean_report(pred, ground_truth):
    # Clean all text
    pred_clean = data_preprocessing.clean_text(pred)
    ground_truth_clean = data_preprocessing.clean_text(ground_truth)
    return pred_clean, ground_truth_clean

def hamming(tags, pred_tags, n):
    '''
    Find hammming loss of multilabel classification of tags 

    Args:
    tags(list): Ground truth tags for the image
    pred_tags_prob(torch.Tensor): Probability of tags as determined by tags model

    Returns:
    hamming(float): Hamming loss of classification
    '''
    y_pred_onehot = np.zeros_like(pred_tags)
    print("length of tags in metrics:",n)
    # Extract top n tags with highest probability
    top_n_pred = np.argsort(pred_tags[0])[-n:]
    print("top_n_pred:",top_n_pred)
    # Set them as 1 and the rest as 0
    y_pred_onehot[0][top_n_pred] = 1
    print("y_pred_one_hot:",y_pred_onehot[0])
    print("tags in metrics",tags)
    # Calculate hamming loss on tags    
    hamming = hamming_loss(tags, y_pred_onehot[0])

    return hamming

def semantic_similarity(pred, ground_truth):
    '''
    Obtain embeddings using spacy corpus annd find semantic similarity between prediction and ground truth

    Args:
    pred(str): Predicted component of report
    groud_truth(str): Ground truth component of report

    Returns:
    sim(float): Semantic similarity between prediction and ground truth
    '''
    pred_clean, ground_truth_clean = clean_report(pred, ground_truth)
    pred_emb = emb(pred_clean)
    ground_truth_emb = emb(ground_truth_clean)

    # Get semantic similarity
    sim = pred_emb.similarity(ground_truth_emb)

    return sim

