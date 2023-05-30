from collections import defaultdict
import os
import json
import numpy as np
import h5py
from utils.data import angle_feature
import re
import nltk
import math

def read_category_file(infile):
    category_mapping = {}
    category_list = []
    category_number = {}
    with open(infile, 'r',encoding='utf-8') as f:
        next(f) # pass the first line
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

def preprocess_name(name,cat_mapping,cat_number,lem):
    ''' preprocess the name of object
    '''
    name = re.sub(r'[^\w\s]',' ',str(name).lower().strip())
    name = lem.lemmatize(name) # convert the word into root word
    name = ''.join([i for i in name if not i.isdigit()]) # remove number
    if name in cat_mapping:
        name = cat_mapping[name]
    else:
        name = name.split(' ')[0]
        if name in cat_mapping:
            name = cat_mapping[name]
        else:
            name = 'others'
    number = cat_number[name]
    return name, number

class ObjectFeatureDB(object):
    def __init__(self, obj_ft_file, obj_feat_size):
        self.obj_feat_size = obj_feat_size
        self.obj_ft_file = obj_ft_file
        self._feature_store = {}

    def load_feature(self, scan, viewpoint, max_objects=None):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.obj_ft_file, 'r') as f:
                obj_attrs = {}
                if key in f:
                    obj_fts = f[key][...][:, :self.obj_feat_size].astype(np.float32) 
                    for attr_key, attr_value in f[key].attrs.items():
                        obj_attrs[attr_key] = attr_value
                else:
                    obj_fts = np.zeros((0, self.obj_feat_size), dtype=np.float32)
            self._feature_store[key] = (obj_fts, obj_attrs)

        if max_objects is not None:
            obj_fts = obj_fts[:max_objects]
            obj_attrs = {k: v[:max_objects] for k, v in obj_attrs.items()}
        return obj_fts, obj_attrs

    def get_object_feature(
        self, scan, viewpoint, base_heading, base_elevation, angle_feat_size,
        max_objects=None
    ):
        obj_fts, obj_attrs = self.load_feature(scan, viewpoint, max_objects=max_objects)
        obj_ang_fts = np.zeros((len(obj_fts), angle_feat_size), dtype=np.float32)
        obj_box_fts = np.zeros((len(obj_fts), 3), dtype=np.float32)
        obj_ids = []
        if len(obj_fts) > 0:
            for k, obj_ang in enumerate(obj_attrs['directions']):
                obj_ang_fts[k] = angle_feature(
                    obj_ang[0] - base_heading, obj_ang[1] - base_elevation, angle_feat_size
                )
                w, h = obj_attrs['sizes'][k]
                obj_box_fts[k, :2] = [h/480, w/640]
                obj_box_fts[k, 2] = obj_box_fts[k, 0] * obj_box_fts[k, 1]
            obj_ids = obj_attrs['obj_ids']

        return obj_fts, obj_ang_fts, obj_box_fts, obj_ids

class ObjectFeaatureDBv2(object):
    ''' dataV2.
    '''
    def __init__(self, args, obj_ft_file, cat_file=None):
        self.args = args
        self.obj_ft_file = obj_ft_file
        self.cat_file = cat_file
        self._feature_store = {}
        self.not_use_obj_list = ['wall','floor','ceiling','others','board_panel','unlabeled','blinds','void']
        with open(self.obj_ft_file, 'r') as f:
            self.object_data = json.load(f)
        if self.cat_file is not None:
            self.cat_mapping, self.category_number = read_category_file(self.cat_file)
            self.lem = nltk.stem.wordnet.WordNetLemmatizer()
        else:
            self.cat_mapping, self.category_number = None, None
    
    def load_feature(self, scan, viewpoint, max_objects=20, object_filter=True):
        key = '%s_%s' % (scan, viewpoint)
        width, height = 640, 480
        if key in self._feature_store:
            bbox_pad_feature, lens = self._feature_store[key]
        else:
            target = self.object_data[key]
            bbox_feature = []
            category_feature = []
            lens = []
            for idx in range(36):
                tmp_bbox = []
                tmp_cat = []
                if str(idx) in target.keys():
                    tmp_len = 0
                    for item in target[str(idx)]:
                        x,y,w,h = item['bbox2d'][:4]
                        heading = (x+w/2-width/2)/width * math.radians(30)
                        elevation = (y+h/2-height/2)/height * math.radians(30)
                        bbox2d = [x/width,y/height,w/width,h/height,
                                (w/width)*(h/height),heading,elevation]
                        category = item['category']
                        if self.cat_mapping is not None:
                            name = item['name']
                            if object_filter and name in self.not_use_obj_list:
                                continue
                            _, category = preprocess_name(name,self.cat_mapping,self.category_number,self.lem)
                        
                        bbox2d.append(category+1)
                        tmp_bbox.append(bbox2d)
                        tmp_len += 1

                    tmp_len = min(tmp_len,max_objects)
                    lens.append(tmp_len)
                        
                    tmp_bbox = sorted(tmp_bbox,key=lambda box:box[4],reverse=True)
                    # tmp_bbox: [x1,y1,w,h,area,heading,elevation,cat]
                    tmp_cat = [x[-1] for x in tmp_bbox]
                    
                else:
                    tmp_bbox = [0,0,0,0,0,0,0,0]
                    tmp_cat = [0]
                    lens.append(0)
                
                bbox_feature.append(tmp_bbox)
                category_feature.append(tmp_cat)
            
            max_len = max(lens)
            # pad np array
            bbox_pad_feature = np.zeros([36,max_len,8])
            cat_pad_feature = np.zeros([36,max_len,1])
            for i in range(36):
                cur_len = lens[i] if lens[i] < max_len else max_len
                if cur_len == 0:
                    continue
                bbox_pad_feature[i, :cur_len, ...] = np.array(bbox_feature[i][:cur_len])
                cat_pad_feature[i, :cur_len, ...] = np.array(category_feature[i][:cur_len]).reshape(-1,1)
            
            lens = np.array(lens).reshape(36)
            self._feature_store[key] = (bbox_pad_feature, lens)
        
        return bbox_pad_feature, lens
    
    def get_object_feature(self, scan, viewpoint, max_objects=20):
        bbox_pad_feature, lens = self.load_feature(scan, viewpoint, max_objects=max_objects)
        return bbox_pad_feature, lens
      

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, word_picker=None):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            if splits[0] == 'test':
                new_item['instruction'] = instr
                if word_picker is not None:
                    objects, actions = word_picker.pick_action_object_words(instr)
                    new_item['objects'] = objects
                    new_item['actions'] = actions
                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                new_item['path_id'] = item['id']  + '_' + str(j)
                new_item['instr_id'] = item['id'] + '_' + str(j)
                new_item['objId'] = None
                del new_item['instructions']
                del new_item['instr_encodings']
            else:
                if 'path_id' not in item.keys():
                    item['path_id'] = item['instr_id']
                if 'objId' in item:
                    new_item['instr_id'] = '%s_%s_%d' % (str(item['path_id']), str(item['objId']), j)
                else:
                    if 'aug' in splits[0]:
                        new_item['id'] = item['path_id']
                        new_item['path_id'] = item['path_id']
                        new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    else:
                        new_item['path_id'] = item['id']
                        new_item['instr_id'] = '%s_%d' % (item['id'], j)
                    new_item['objId'] = None
                new_item['instruction'] = instr
                if word_picker is not None:
                    objects, actions = word_picker.pick_action_object_words(instr)
                    new_item['objects'] = objects
                    new_item['actions'] = actions
                if len(item['instructions']) == 1:
                    try:
                        new_item['instr_encoding'] = item['instr_encodings'][:max_instr_len]
                    except Exception:
                        new_item['instr_encoding'] = item['instr_encoding'][:max_instr_len]
                else:
                    try:
                        new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                    except Exception:
                        new_item['instr_encoding'] = item['instr_encoding'][j][:max_instr_len]
                del new_item['instructions']
                del new_item['instr_encodings']

            data.append(new_item)
    return data

def load_obj2vps(bbox_file):
    obj2vps = {}
    bbox_data = json.load(open(bbox_file))
    for scanvp, value in bbox_data.items():
        scan, vp = scanvp.split('_')
        # for all visible objects at that viewpoint
        for objid, objinfo in value.items():
            if objinfo['visible_pos']:
                # if such object not already in the dict
                obj2vps.setdefault(scan+'_'+objid, [])
                obj2vps[scan+'_'+objid].append(vp)
    return obj2vps