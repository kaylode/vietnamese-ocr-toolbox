from modules.retrieval.heuristic.heuristic import Matcher
import difflib
import re
import math

def trie_correction(texts, dictionary, threshold=0.85):
    preds = []
    match_score = []
    matcher = Matcher(dictionary)
    for query_txt in texts:
        key, score = matcher.get_match(query_txt.lower())
        if score > threshold:
            preds.append(key)
            match_score.append(score)
        else:
            preds.append(query_txt)
            match_score.append(0)
    return preds, match_score

def diff_correction(texts, dictionary, threshold=0.85):
    def sentence_distance(p1,p2):
        return difflib.SequenceMatcher(None, p1, p2).ratio()
    
    preds = []
    match_score = []

    for query_txt in texts:
        dis_list = [(key, sentence_distance(query_txt.lower(), key)) for key in dictionary.keys()]
        dis_list = sorted(dis_list,key=lambda tup: tup[1],reverse=True)[:5]
        key, score = dis_list[0]
        if score > threshold:
            preds.append(key)
            match_score.append(score)
        else:
            preds.append(query_txt)
            match_score.append(0)
    return preds, match_score

def get_heuristic_correction(type_='diff'):
    return trie_correction if type_=='trie' else diff_correction

