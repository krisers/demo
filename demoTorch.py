import os 
import re
import cv2 
import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pickle import load,dump
from sklearn.feature_extraction.text import CountVectorizer

from utils_torch import  Image_Caption,get_tier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = '/home/kriselda/source/repos/a-PyTorch-Tutorial-to-Image-Captioning/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
checkpoint = torch.load(model_path, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
word_map_path = '/home/kriselda/source/Datasets/coco/images/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
with open(word_map_path, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

# Encode, decode with attention and beam search

im_cp = Image_Caption(images_dir='',
                      model = model,
                      encoder =encoder,
                      decoder = decoder,
                      w2v = embeddings_index_all,
                      tier = get_tier('tier1.txt'),
                      word_index_Mapping=word_index_Mapping,
                      word_map=word_map,
                      index_word_Mapping=index_word_Mapping,
                      max_caption_length=max_caption_length,
                      vocab_size=vocab_size,
                      beam_size=3
)

im_cp.caption_video('sample2.mp4')