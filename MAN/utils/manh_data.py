import torch

import pickle

FType = torch.FloatTensor
LType = torch.LongTensor

poi_info_path = "MAN/dataset/poi_info.pickle"
a_path = 'MAN/dataset/attribute_m.pickle'
s_path = 'MAN/dataset/source_matrix.pickle'
d_path = 'MAN/dataset/destina_matrix.pickle'

class ReData:
    def __init__(self):
        self.a_m = pickle.load(open(a_path, "rb"))
        self.s_m = pickle.load(open(s_path, "rb"))
        self.d_m = pickle.load(open(d_path, "rb"))
        self.poi_info = pickle.load(open(poi_info_path, "rb"))


