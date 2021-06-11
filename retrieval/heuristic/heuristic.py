from collections import defaultdict
import difflib
import re
import math

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode) 
        self.smallest_str = None
        self.end = None
    def __getitem__(self, c):
        return self.children[c]
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, s: str):
        node = self.root
        for c in s:
            node = node[c]
            if node.smallest_str is None:
                node.smallest_str = s
        node.end = s
    def get_similar(self, s):
        node = self.root
        for i, c in enumerate(s):
            if c not in node.children:
                i -= 1
                break
            node = node[c]
        return (node.smallest_str or node.end, i + 1)


class Matcher:
    def __init__(self, dic: dict):
        self.trie = Trie()
        for s in dic:
            self.trie.insert(s)

    def get_match(self, s: str) -> tuple:
        return self.trie.get_similar(s)


def get_multiple_trie_match(texts, dictionary):
    preds = []
    probs = []
    matcher = Matcher(dictionary)
    for query_txt in texts:
        key, score = matcher.get_match(query_txt.lower())
        if key is None or score ==0 or math.isnan(score):
            preds.append(4)
            probs.append(0.0)
        else:
            preds.append(dictionary[key])
            probs.append(score/len(key))
    return preds, probs

def get_multiple_diff_match(texts, dictionary):
    def sentence_distance(p1,p2):
        return difflib.SequenceMatcher(None, p1, p2).ratio()
    
    preds = []
    probs = []

    for query_txt in texts:
        dis_list = [(key, sentence_distance(query_txt.lower(), key)) for key in dictionary.keys()]
        dis_list = sorted(dis_list,key=lambda tup: tup[1],reverse=True)[:5]
        key, score = dis_list[0]
        if score != 0 and not math.isnan(score):
            preds.append(dictionary[key])
        else:
            preds.append(4)
        probs.append(score)
    return preds, probs


def regex_timestamp(texts):
    preds = []
    time = r'\d{2}:\d{2}:\d{2}|\d{2}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{2}'
    date = r'(\d+/\d+/\d+)'
    regex = '|'.join([time,date])
    for query_txt in texts:
        x = re.search(regex, query_txt)
        if x:
            preds.append(1)
        else:
            preds.append(0)
    return preds

def get_heuristic_retrieval(type_='diff'):
    return get_multiple_trie_match if type_=='trie' else get_multiple_diff_match