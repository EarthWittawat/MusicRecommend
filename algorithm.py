from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from pythainlp.util import collate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch

docs = pd.read_csv('datasets/dataset.csv')
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
df = docs[["artist","song","text"]]
embeddings_des = torch.load("datasets/embeddings_des.pt",map_location=torch.device('cpu'))
def get_title_of(row_id: int) -> str:
    select = docs['text'][row_id]
    return select
def get_music(x:str, return_id=True) -> list:
    query = model.encode([x])
    cosine_scores = util.cos_sim(query, embeddings_des)
    all_idx = torch.topk(cosine_scores.flatten(), 5).indices
    cluster = []
    for i in all_idx:
       cluster.append(df.loc[int(i), "song"])
       print("name :", df.loc[int(i), "song"])
    return cluster