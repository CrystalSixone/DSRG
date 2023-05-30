import os, sys
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import random
import re
import string
from collections import defaultdict
import spacy
import nltk
from transformers import DistilBertTokenizer
import csv
import base64


class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint, type='hdf5'):
        if type == 'hdf5':
            return self.get_image_feature_from_h5py(scan, viewpoint)
        elif type == 'tsv':
            return self.get_image_feature_from_tsv(scan, viewpoint)

    def get_image_feature_from_h5py(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft
    
    def get_image_feature_from_tsv(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        views = 36
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            scanIds = []
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
            scanIds_viewpointsId = {}

            with open(self.img_ft_file, "r") as tsv_in_file:     # Open the tsv file.
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
                for item in reader:
                    scanId = item['scanId']
                    if scanId not in scanIds:
                        scanIds.append(scanId)
                        scanIds_viewpointsId[scanId] = []
                        scanIds_viewpointsId[scanId].append(item['viewpointId'])
                    else:
                        scanIds_viewpointsId[scanId].append(item['viewpointId'])
                    long_id = item['scanId'] + "_" + item['viewpointId']
                    ft = np.frombuffer(base64.decodebytes(item['features'].encode('ascii')),
                                                    dtype=np.float32).reshape((views, -1))
                    self._feature_store[long_id] = ft
        return ft

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

class PickSpecificWords():
    def __init__(self, cat_file=None):
        self.bert_tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.anno_path = 'datasets/R2R/annotations/R2R_%s_enc.json'
        self.spacy_model = spacy.load("en_core_web_sm")
        self.action_list = [
            'right','left','down','up','forward','around','straight',
            'into','front','behind','exit','enter','besides','through',
            'stop','out','wait','passed','climb','leave','past','before','after',
            'between','in','along','cross','end','head','inside','outside','across',
            'towards','face','ahead','toward'
        ]
        self.cat_file = cat_file
        if self.cat_file is not None:
            self.cat_mapping, self.category_number = self.read_category_file(self.cat_file)
            self.lem = nltk.stem.wordnet.WordNetLemmatizer()
            self.action_map = {}
            for index, val in enumerate(self.action_list):
                self.action_map[val] = index
    
    def read_category_file(self,infile):
        category_mapping = {}
        category_list = []
        category_number = {}
        with open(infile, 'r',encoding='utf-8') as f:
            next(f)
            for line in f:
                line = line.strip('\n').split('\t')  
                source_name, target_category = line[1], line[-1]
                category_mapping[source_name] = target_category
                if target_category not in category_list:
                    category_list.append(target_category)
            category_list.append('others')
            for i,cat in enumerate(category_list):
                category_number[cat] = i
        return category_mapping, category_number
    
    def pick_action_object_words(self,instr,map=True):
        tokens = self.spacy_model(instr)
        action_list = []
        object_list = []
        # record_list: record the word should be masked.
        # mask_id_list: record the index of the word in bert tokens.
        for num,token in enumerate(tokens):
            if token.pos_ == 'NOUN':
                # focus on NOUN
                name = re.sub(r'[^\w\s]',' ',str(token).lower().strip())
                name = self.lem.lemmatize(name) # convert the word into root word
                name = ''.join([i for i in name if not i.isdigit()]) # remove number
                if name in self.cat_mapping.keys():
                    name_map = self.cat_mapping[name]
                    if name_map in self.category_number.keys():
                        if map:
                            object_list.append(self.category_number[name_map]+1)
                            # +1 for different from [PAD]
                        else:
                            object_list.append(name_map)
            if str(token).lower() in self.action_list:
                # focus on ACTION
                if map:
                    action_list.append(self.action_map[str(token).lower()]+1)
                else:
                    action_list.append(str(token).lower())
        return object_list, action_list
    
