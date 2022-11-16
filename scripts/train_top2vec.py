import sys
import os
import pandas as pd
from tqdm import tqdm
import json
import string
from top2vec import Top2Vec
import multiprocessing


def simple_segmentation(df, rz):
    
    for ix in tqdm(df.doc_id.unique()):
        
        article = df.loc[df.doc_id == ix]
        full_text = rz.loc[ix, 'full_text']
        
        msg_ids, placenames, starts, ends = list(article.index), list(article.placename), list(article.start), list(article.end)
        
        for msg_id, placename, end, start in zip(msg_ids, placenames, ends, starts[1:]+[len(full_text)]):
            
            yield {"msg_id": msg_id,
                   "doc_id": int(ix),
                   "placename": placename,
                   "text": full_text[end:start]}


def is_garbage(token, treshold=0.3):
    """Test if a token "is garbage", i.e. if it contains too many weird symbols"""
    
    allowed_symbols = string.ascii_letters + 'äüöß'

    if len(token) == 0:
        return True
    
    non_alphabetical = 0
    for symbol in token:
        if symbol not in allowed_symbols:
            non_alphabetical += 1
            
    if non_alphabetical/len(token) >= treshold:
        return True
    else:
        return False


def tokenize(text, stopwords, min_len=4):
    """Parse the main dataframe into lists if lowercase tokens,
    leaving out stopwords, garbage tokens and words with len < 4 (default value)"""

    tokens = text.split()
    cleaned = []

    for token in tokens:
        wordform = token.lower().strip(string.punctuation).lstrip(string.punctuation)
        if not is_garbage(wordform) and len(wordform) >= min_len and wordform not in stopwords:
            cleaned.append(wordform)

    return cleaned


def tokenize_spans(spans):
    
    if '„' not in string.punctuation:
        string.punctuation += '„'

    spans_tokenized = []
    
    for entry in tqdm(spans):
        spans_tokenized.append({
            "msg_id": entry["msg_id"],
            "doc_id": entry["doc_id"],
            "tokens": tokenize(entry["text"], stopwords)
        })
        
    return spans_tokenized


def build_top2vec_corpus(spans):
    """Prepare top2vec corpus"""
    
    corpus = {}
    
    for entry in spans:
        corpus[str(entry['msg_id'])+'_'+str(entry['doc_id'])] = ' '.join(entry['tokens'])
        
    return corpus



if __name__ == '__main__':

    min_count = int(sys.argv[1])
    model_name = str(sys.argv[2]) # file name only, no extension or directory (path included in function)

    df = pd.read_csv('../data/processed_data.tsv', sep='\t', encoding='utf8')
    df.doc_date = pd.to_datetime(df.doc_date)
    df.origin_date = pd.to_datetime(df.origin_date)
    df = df[df.delta.isin(range(0,120))]

    rz = pd.read_parquet('../data/raw/RZ_processed.parquet')

    spans = list(simple_segmentation(df, rz))

    with open('../temp/stopwords.json', 'r', encoding='utf8') as f:
        stopwords = json.load(f)

    spans_tokenized = tokenize_spans(spans)

    corpus = build_top2vec_corpus(spans_tokenized)

    t2v = Top2Vec(documents=list(corpus.values()),
              document_ids=list(corpus.keys()),
              min_count=min_count,
              speed='deep-learn',
              workers=multiprocessing.cpu_count())

    if os.path.exists('..\\data\\models'):
        t2v.save(f'..\\data\\models\\{model_name}.pkl')
    else:
        t2v.save(f'{model_name}.pkl') # backup save in script directory to avoid losing the model

    print('Finished!')



    


