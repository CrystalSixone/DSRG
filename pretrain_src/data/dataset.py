'''
Instruction and trajectory (view and object features) dataset
For loading object v2
'''
import os
import json
import jsonlines
import numpy as np
import h5py
import math
import re
import nltk

from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax

MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

def read_category_file(infile):
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

def preprocess_name(name,cat_mapping,cat_number):
    ''' preprocess the name of object
    '''
    name = re.sub(r'[^\w\s]',' ',str(name).lower().strip())
    lem = nltk.stem.wordnet.WordNetLemmatizer()
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

class ReverieTextPathData(object):
    def __init__(
        self, anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        objv2_ft_file=None,cat_file=None,args=None,tok=None,
        max_objv2=3
    ):  
        self.args = args
        self.img_ft_file = img_ft_file
        self.obj_ft_file = obj_ft_file
        self.cat_file = cat_file
        self.object_data = None
        self.tok = tok
        self.use_objv2 = self.args.use_objv2 if self.args is not None else False
        if self.cat_file is not None:
            self.cat_mapping, self.category_number = read_category_file(self.cat_file)
        else:
            self.cat_mapping, self.category_number = None, None
        self.objv2_ft_file = objv2_ft_file
        if self.use_objv2 and self.objv2_ft_file is not None:
            self.not_use_obj_list = ['wall','floor','ceiling','others','board_panel','unlabeled','blinds','void']
            with open(self.objv2_ft_file, 'r') as f:
                self.object_data = json.load(f)

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.max_objv2 = max_objv2
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]

        self.data = []
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                for i, item in enumerate(f):
                    self.data.append(item)
                    item['instr'] = self.tok.decode(item['instr_encoding'],skip_special_tokens=True)
                    if self.args.debug and i >= 50:
                        break

    def __len__(self):
        return len(self.data)

    def get_scanvp_feature(self, scan, viewpoint, max_objv2=3,object_filter=True):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            if self.use_objv2:
                view_fts, obj_fts, obj_attrs, bbox_pad_feature, objv2_lens = self._feature_store[key]
            else:
                view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)
            
            # read objv2 features
            if self.use_objv2:
                max_objects = self.args.max_objv2
                width, height = 640, 480
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
                            name = item['name']
                            if object_filter and name in self.not_use_obj_list:
                                continue
                            _, category = preprocess_name(name,self.cat_mapping,self.category_number)
                            
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
                
                objv2_lens = np.array(lens).reshape(36)

            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
        
        if self.in_memory:
            self._feature_store[key] = (view_fts, obj_fts, obj_attrs, bbox_pad_feature, objv2_lens)
        return view_fts, obj_fts, obj_attrs, bbox_pad_feature, objv2_lens


    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100 # ignore 
            # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            # local: 
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos': # pos
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
        
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
        last_vp_angles, last_vp_objids, \
        traj_objv2_fts, traj_objv2_lens = self.get_traj_pano_fts(scan, gt_path, self.args.max_objv2)
        
        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr': item['instr'][:self.max_txt_len],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            # 'vp_objids': last_vp_objids,
            'vp_angles': last_vp_angles,
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)
        
        outs['traj_objv2_fts'] = traj_objv2_fts
        outs['traj_objv2_lens'] = traj_objv2_lens

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation

    def get_traj_pano_fts(self, scan, path, max_objv2=None):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []
        traj_objv2_img_fts, traj_objv2_lens = [], []
        for vp in path:
            if self.use_objv2:
                view_fts, obj_img_fts, obj_attrs, objv2_fts, objv2_lens = self.get_scanvp_feature(scan, vp)
            else:
                view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            view_objv2_fts, view_objv2_lens = [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
                # For object
                objv2_view_angle = np.array([view_angle[0]+objv2_fts[v[0]][:,5],view_angle[1]+objv2_fts[v[0]][:,6]]).transpose()
                objv2_angle_fts = get_angle_fts(objv2_view_angle[:, 0], objv2_view_angle[:, 1], self.angle_feat_size)
                view_objv2_fts.append(np.concatenate([objv2_fts[v[0]][:,:2],objv2_fts[v[0]][:,4:5],objv2_angle_fts,objv2_fts[v[0]][:,-1].reshape(-1,1)],-1)) # [x,y,area,ang_feat,name]
                view_objv2_lens.append(objv2_lens[v[0]])

            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # objv2 feature
            for idx in range(36):
                if idx not in used_viewidxs:
                    objv2_view_angle = np.array([self.all_point_rel_angles[12][idx][0]+objv2_fts[idx][:,5],self.all_point_rel_angles[12][idx][1]+objv2_fts[idx][:,6]]).transpose()
                    objv2_angle_fts = get_angle_fts(objv2_view_angle[:, 0], objv2_view_angle[:, 1], self.angle_feat_size)
                    view_objv2_fts.extend([np.concatenate([objv2_fts[idx][:,:2],objv2_fts[idx][:,4:5],objv2_angle_fts,objv2_fts[idx][:,-1].reshape(-1,1)],-1)]) 
            view_objv2_lens.extend([objv2_lens[idx] for idx in range(36) if idx not in used_viewidxs])
            view_objv2_fts = np.stack(view_objv2_fts,0) # (n_views, n_objs, dim_ft)
            view_objv2_lens = np.stack(view_objv2_lens,0) # (n_views)
            traj_objv2_img_fts.append(view_objv2_fts)
            traj_objv2_lens.append(view_objv2_lens)

            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        # padding obj_len in all navigation nodes
        max_len = max([x.shape[1] for x in traj_objv2_img_fts])
        for i,item in enumerate(traj_objv2_img_fts):
            view_len, obj_len, obj_dim = item.shape
            tmp_obj_fts = np.zeros([view_len,max_len,obj_dim])
            tmp_obj_fts[:,:obj_len,:] = item[:,:obj_len,:]
            traj_objv2_img_fts[i] = tmp_obj_fts
        return traj_view_img_fts, traj_obj_img_fts,traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles, last_vp_objids,\
            traj_objv2_img_fts, traj_objv2_lens
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    
    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
        
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
       

class R2RTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        objv2_ft_file=None,cat_file=None,args=None,tok=None
    ):
        super().__init__(
            anno_files, img_ft_file, None, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0, 
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node,
            objv2_ft_file=objv2_ft_file,cat_file=cat_file,args=args,tok=tok
        )

    def get_scanvp_feature(self, scan, viewpoint, max_objects=20, object_filter=True):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            if not self.use_objv2:
                view_fts = self._feature_store[key]
            else:
                view_fts, bbox_pad_feature, obj_lens = self._feature_store[key]

        else:
            # read image features
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)

            # read objv2 features
            if self.use_objv2:
                max_objects = self.args.max_objects
                width, height = 640, 480
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
                                _, category = preprocess_name(name,self.cat_mapping,self.category_number)
                            
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
                
                obj_lens = np.array(lens).reshape(36)
        
        if self.in_memory:
            self._feature_store[key] = (view_fts, bbox_pad_feature, obj_lens)
        return view_fts, bbox_pad_feature, obj_lens


    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1 # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos': 
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)
            
        gt_path = gt_path[:end_idx+1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
        
        if self.use_objv2:
            traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, traj_obj_fts, traj_obj_lens = self.get_traj_pano_fts(scan, gt_path, self.args.max_objects)
        else:
            traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
                last_vp_angles = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            # 'instr': item['instr'][:self.max_txt_len],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
        }
        if 'instr' in item.keys():
            outs['instr'] = item['instr'][:self.max_txt_len]

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
        
        outs['traj_obj_fts'] = traj_obj_fts
        outs['traj_obj_lens'] = traj_obj_lens

        return outs

    def get_traj_pano_fts(self, scan, path, max_objects=None):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], []
        traj_obj_img_fts, traj_obj_lens = [],[]
        for vp in path:
            view_fts, objv2_fts, objv2_lens = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            view_obj_fts, view_obj_lens = [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)

                obj_view_angle = np.array([view_angle[0]+objv2_fts[v[0]][:,5],view_angle[1]+objv2_fts[v[0]][:,6]]).transpose()
                obj_angle_fts = get_angle_fts(obj_view_angle[:, 0], obj_view_angle[:, 1], self.angle_feat_size)
                view_obj_fts.append(np.concatenate([objv2_fts[v[0]][:,:2],objv2_fts[v[0]][:,4:5],obj_angle_fts,objv2_fts[v[0]][:,-1].reshape(-1,1)],-1)) # [x,y,area,ang_feat,name]
                view_obj_lens.append(objv2_lens[v[0]])
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            for idx in range(36):
                if idx not in used_viewidxs:
                    obj_view_angle = np.array([self.all_point_rel_angles[12][idx][0]+objv2_fts[idx][:,5],self.all_point_rel_angles[12][idx][1]+objv2_fts[idx][:,6]]).transpose()
                    obj_angle_fts = get_angle_fts(obj_view_angle[:, 0], obj_view_angle[:, 1], self.angle_feat_size)
                    view_obj_fts.extend([np.concatenate([objv2_fts[idx][:,:2],objv2_fts[idx][:,4:5],obj_angle_fts,objv2_fts[idx][:,-1].reshape(-1,1)],-1)]) 
            
            view_obj_lens.extend([objv2_lens[idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_obj_fts = np.stack(view_obj_fts,0) # (n_views, n_objs, dim_ft)
            view_obj_lens = np.stack(view_obj_lens,0) # (n_views)
            
            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)

            traj_obj_img_fts.append(view_obj_fts)
            traj_obj_lens.append(view_obj_lens)
            
            last_vp_angles = view_angles

        # padding obj_len in all navigation nodes
        max_len = max([x.shape[1] for x in traj_obj_img_fts])
        for i,item in enumerate(traj_obj_img_fts):
            view_len, obj_len, obj_dim = item.shape
            tmp_obj_fts = np.zeros([view_len,max_len,obj_dim])
            tmp_obj_fts[:,:obj_len,:] = item[:,:obj_len,:]
            traj_obj_img_fts[i] = tmp_obj_fts
        return traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles, \
            traj_obj_img_fts, traj_obj_lens