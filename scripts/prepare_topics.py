import os
import sys
import json
import copy
from tqdm import tqdm
from itertools import chain
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from top2vec import Top2Vec
from scipy.special import softmax
from wordcloud import WordCloud


def define_plot_fonts():
    fm.fontManager.addfont('../references/cmunorm.ttf')
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rcParams['font.family'] = 'CMU Concrete'


def custom_topic_wordcloud(top, savepath, previous_reduction=0):
    
    cloudwidth = 2400
    cloudheight = 800
    
    topic_words = t2v.get_topics(reduced=True)[0][top]
    word_scores = t2v.get_topics(reduced=True)[1][top]
    
    topic_words_dict = dict(zip(topic_words, softmax(word_scores)))
    
    wc = WordCloud(background_color='white', width=cloudwidth, height=cloudheight,
                   font_path='../references/cmunrm.ttf')
    
    wc.generate_from_frequencies(topic_words_dict)
    
    plt.figure(figsize=(12, 4))
    plt.imshow(wc)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{savepath}\\topic_{str(top+previous_reduction)}.png', bbox_inches='tight')
    plt.clf()


def get_topic_stats(top, savepath, previous_reduction=0):
    
    topic_size = t2v.get_topic_sizes(reduced=True)[0][top]
    topic_document_ids = t2v.search_documents_by_topic(top, topic_size, reduced=True)[2]
    
    doc_ids = [int(ID.split('_')[1]) for ID in topic_document_ids]
    
    top_df = rz.loc[doc_ids]
    
    define_plot_fonts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    top_df.year.hist(bins=87, ax=ax1)
    ax1.set_xlim(1802, 1889)
    
    top_df.heading2.value_counts()[:10].plot.bar(ax=ax2)
    ax2.set_xticklabels(labels=top_df.heading2.value_counts()[:10].index, rotation=45, ha='right')
    
    plt.savefig(f'{savepath}\\topic_{str(top+previous_reduction)}.png', bbox_inches='tight')
    plt.clf()


def get_topic_examples(top, n, savepath, previous_reduction=0):
    
    examples = t2v.search_documents_by_topic(top, n, reduced=True)[2]
    
    msg_ids = [int(ex.split('_')[0]) for ex in examples]
    doc_ids = [int(ex.split('_')[1]) for ex in examples]
        
    example_texts = []
    
    for msg_id, doc_id in zip(msg_ids, doc_ids):
        article = df.loc[df.doc_id == doc_id]
        
        if len(article) == 1:
            span_start = article.start.values[0]
            span_end = -1
    
        elif article.index[-1] == msg_id:
            span_start = article.loc[msg_id, 'start']
            span_end = -1
            
        elif len(article) > 1:
            span_start = article.loc[msg_id, 'start']
            span_end = article.loc[msg_id+1, 'start']
            
        example_texts.append(
                                {"msg_id": msg_id,
                                 "doc_id": doc_id,
                                 "date": rz.loc[doc_id, 'date'],
                                 "heading": rz.loc[doc_id, 'heading'],
                                 "text": rz.loc[doc_id, 'full_text'][span_start:span_end]
                                }
                            )
        
    with open(savepath+f'\\examples_{str(top+previous_reduction)}.json', 'w', encoding='utf8') as f:
        json.dump(example_texts, f)


# def get_default_labels(t2v, n_words, previous_reduction):
    
#     labels = {}
    
#     for top, top_words in enumerate(t2v.get_topics()[0]):
#         labels[top+previous_reduction] = '-'.join(top_words[:n_words])
        
#     return labels


def create_topic_data_for_reduction(reduction, previous_reduction):
    
    directory = os.getcwd().strip('scripts') + f'data\\topics\\reduction_{str(reduction)}'    
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    print(f'\n\nPerforming reduction to {str(reduction)} topics')
    t2v.hierarchical_topic_reduction(reduction)
    
    print('Generating wordclouds')
    if not os.path.exists(directory+'\\wordclouds'):
        os.mkdir(directory+'\\wordclouds')
        
    for top in tqdm(range(reduction)):
        custom_topic_wordcloud(top, savepath=directory+'\\wordclouds', previous_reduction=previous_reduction)
        
    print('Generating statistics')
    if not os.path.exists(directory+'\\statistics'):
        os.mkdir(directory+'\\statistics')
        
    for top in tqdm(range(reduction)):
        get_topic_stats(top, savepath=directory+'\\statistics', previous_reduction=previous_reduction)
         
    print('Fetching examples')
    if not os.path.exists(directory+'\\examples'):
        os.mkdir(directory+'\\examples')
        
    for top in tqdm(range(reduction)):
        get_topic_examples(top, n=20, savepath=directory+'\\examples', previous_reduction=previous_reduction)
        
    with open(directory+'\\reduction_hierarchy.json', 'w', encoding='utf8') as f:
        hierarchy = t2v.get_topic_hierarchy()
        json.dump(hierarchy, f)       

    with open(directory+'\\sizes.json', 'w', encoding='utf8') as f:
        #sizes = [int(i) for i in list(t2v.get_topic_sizes(reduced=True)[0])]
        sizes = {top+previous_reduction: int(size) for top, size
                    in enumerate(list(t2v.get_topic_sizes(reduced=True)[0][:reduction]))}
        json.dump(sizes, f)

    with open(directory+'\\default_labels.json', 'w', encoding='utf8') as f:
        labels = {top+previous_reduction: '-'.join(top_words[:3]) for top, top_words
                    in enumerate(list(t2v.get_topics(reduced=True)[0][:reduction]))}
        json.dump(labels, f)
        
    print('Finished')


def get_sizes(reductions):

    all_sizes = []

    for r in reductions:
        with open(f'../data/topics/reduction_{str(r)}/sizes.json', 'r', encoding='utf8') as f:
            sizes = json.load(f)
            all_sizes.append(sizes)

    return all_sizes


def get_hierarchies(reductions):

    all_hierarchies = []

    for r in reductions:
        with open(f'../data/topics/reduction_{str(r)}/reduction_hierarchy.json', 'r', encoding='utf8') as f:
            hierarchies = json.load(f)
            all_hierarchies.append(hierarchies)

    return all_hierarchies


def create_hierarchy_df(hierarchies):
    
    levels = copy.deepcopy(hierarchies)
    nodes_flat = dict(enumerate(list(chain(*levels))))
    nodes = []
    
    # makes a list of a dictionary for each level with unique id for each topic
    for level in levels:
        nodes.append(dict(list(nodes_flat.items())[len(list(chain(*nodes))):len(list(chain(*nodes)))+len(level)]))
        
    paths = [[key] for key in nodes[-1].keys()]

    # starts from the penultimate level and looks if the topic is a subset of anything in that level
    for path in paths:
        for level in nodes[-2::-1]:
            last_subtopic = set(nodes[len(nodes)-len(path)][path[-1]])
            for root_key, root_topic in level.items():
                if last_subtopic.issubset(set(root_topic)):
                    path.append(root_key)
            
    paths = [path[::-1] for path in paths]
    
    hierarchy_df = pd.DataFrame(paths).apply(lambda x: x.sort_values().values)
    hierarchy_df.columns = [f'reduction_{len(level)}' for level in nodes]
    
    return hierarchy_df



if __name__ == '__main__':

    model_path = sys.argv[1]
    reductions = [int(r) for r in sys.argv[2:]]

    print('Loading data')
    rz = pd.read_parquet('../data/raw/RZ_processed.parquet')

    df = pd.read_csv('../data/processed_data.tsv', sep='\t', encoding='utf8').convert_dtypes()
    df.doc_date = pd.to_datetime(df.doc_date)
    df.origin_date = pd.to_datetime(df.origin_date)
    df['doc_year'] = df.doc_date.dt.year

    print('Loading model')
    t2v = Top2Vec.load(model_path)
    print(f'{t2v.get_num_topics()} original topics')

    for r, previous_r in zip(reductions, [0]+reductions):
        create_topic_data_for_reduction(r, previous_reduction=previous_r)

    print(f'Creating organized topic hierarchy')
    sizes = get_sizes(reductions)
    hierarchies = get_hierarchies(reductions)

    hierarchy_df = create_hierarchy_df(hierarchies)
    hierarchy_df.to_csv('../data/topics/topic_hierarchy.tsv', sep='\t', encoding='utf8', index=False)

    print('All done!')













