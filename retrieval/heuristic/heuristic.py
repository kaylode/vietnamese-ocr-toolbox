from collections import defaultdict
import difflib

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


def get_multiple_trie_match(texts, lbl_dict, dictionary):
    preds = []
    probs = []
    matcher = Matcher(dictionary)
    for query_txt in texts:
        key, score = matcher.get_match(query_txt)
        preds.append(lbl_dict[dictionary[key]])
        probs.append(score)
    return preds, probs

def get_multiple_diff_match(texts, lbl_dict, dictionary):
    def sentence_distance(p1,p2):
        return difflib.SequenceMatcher(None, p1, p2).ratio()

    preds = []
    probs = []
    dis_list = [(key, sentence_distance(query_txt, key)) for key in dictionary.keys()]
    dis_list = sorted(dis_list,key=lambda tup: tup[1],reverse=True)[:5]
    key, score = dis_list[0]
    preds.append(lbl_dict[dictionary[key]])
    probs.append(score)
    return preds, probs

def get_heuristic_retrieval(type_):
    return get_multiple_trie_match if type_=='trie' else get_multiple_diff_match