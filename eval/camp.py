"""
input -> standard format 
output -> camp result. 
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from torchmetrics.text import CharErrorRate

"""
<START> DEBUG TOOLS
"""
def add_dummy_data(pred_data):
    tmp_dict = pred_data['data'][0].copy()
    tmp_dict['name'] = "DEBUG_NAME"
    pred_data['data'].append(tmp_dict)
"""
<END> DEBUG TOOLS
"""

def eval_num(num):
    try:
        return float(str(num).replace(",",""))
    except:
        return num    

def convert_to_float(data):
    error_flag = False

    try: # xmin가 없거나 string type이어야만 하는 경우 고려
        data['xmin'], data['xmax'], data['ymin'], data['ymax'] = map(eval_num, (data['xmin'], data['xmax'], data['ymin'], data['ymax']))
    except:
        pass
    
    # 변환되는 데이터만 가져올 것.
    for item in data['data']:
        new_points = []
        for points in item['points']:
            try:
                new_point = [eval_num(point) for point in points]
                if not any(math.isnan(x) for x in new_point):
                    new_points.append(new_point)
            except:
                error_flag = True

        item['points'] = new_points

    return data, error_flag
    
def init_error_statistics():
    return  {
        "wrong_pred_format": 0, 
        "t_p_len_not_match": 0, 
        "no_data": 0,
    }

def init_result_dict():
    return {
        "line": 0,
        "axis": 0,
        "length": 0,
        "ocr": 0,
    }

interp_func = lambda x, y: interp1d(x, y, kind='linear')
my_interpolation = lambda x, a1, a2, b1, b2: (x - a1) * (b2 - b1) / (a2 - a1) + b1
get_length = lambda x, y: np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
get_ocr_score = lambda x, y, cer: 100*(1 - min(1, cer([x], [y]).item()))

def eval_json(pred_data, gt_data, error_statistics, cer, steps=100, normed_x_margin=1, visualize=False): # json 파일 레벨
    if len(pred_data['data'])==0:
        error_statistics['no_data'] += 1
        return init_result_dict()

    if len(pred_data['data']) != len(gt_data['data']):
        error_statistics['t_p_len_not_match'] += 1
        
    gt_data, _ = convert_to_float(gt_data)
    pred_data, error_flag = convert_to_float(pred_data)
    
    if error_flag: 
        error_statistics['wrong_pred_format'] += 1
    
    matrix_shape = (len(gt_data['data']), len(pred_data['data']))
    score_matrix_line, score_matrix_length, score_matrix_axis, score_matrix_ocr = (np.zeros(matrix_shape) for _ in range(4))

    for index1, t_structure in enumerate(gt_data['data']): # 엔티티 레벨        
        try:
            t_xlist, t_ylist = np.array(t_structure['points'], dtype=float).T
        except:
            breakpoint()
        t_xlist_min, t_xlist_max = min(t_xlist), max(t_xlist)
        
        t_func = interp_func(t_xlist, t_ylist)

        normed_t_x = my_interpolation(t_xlist, gt_data['xmin'], gt_data['xmax'], 0, 1)
        normed_t_y = my_interpolation(t_ylist, gt_data['ymin'], gt_data['ymax'], 0, 1)

        t_length = get_length(normed_t_x, normed_t_y)

        for index2, p_structure in enumerate(pred_data['data']):
            if len(p_structure['points']) <= 1:
                break
            
            p_xlist, p_ylist = np.array(p_structure['points'], dtype=float).T
            p_xlist_min, p_xlist_max = min(p_xlist), max(p_xlist)

            # no x overlap
            if max(p_xlist_min, t_xlist_min) > min(p_xlist_max, t_xlist_max):
                break
            
            minmax_list = sorted([p_xlist_min, t_xlist_min, p_xlist_max, t_xlist_max])
            max_max = [minmax_list[0],minmax_list[-1]]
            min_min = [minmax_list[1],minmax_list[2]]
            
            """
            get axis accuracy
            """
            score_matrix_axis[index1][index2] = 100 * abs((min_min[1]-min_min[0]) / (max_max[1]-max_max[0]))

            """
            get line accuracy
            """
            p_func = interp_func(p_xlist, p_ylist)

            x = np.linspace(min_min[0], min_min[1], steps)
            dx = (min_min[1] - min_min[0]) / (steps - 1)
            
            area_cost = np.sum(np.abs(p_func(x)-t_func(x))) * dx
            normed_cost_area = 100 * (area_cost / (gt_data['ymax'] - gt_data['ymin']) / (gt_data['xmax'] - gt_data['xmin']))

            if normed_cost_area > 100:
                break
                
            score_matrix_line[index1][index2] = 100 - normed_cost_area
            
            """
            get length accuracy
            """
            normed_p_x = my_interpolation(p_xlist, gt_data['xmin'], gt_data['xmax'], 0, 1)
            normed_p_y = my_interpolation(p_ylist, gt_data['ymin'], gt_data['ymax'], 0, 1)
            
            if np.any(normed_p_x > 1 + normed_x_margin) or np.any(normed_p_x < 0 - normed_x_margin): # target의 x의 범위를 너무 많이 벗어났음.
                break

            p_length = get_length(normed_p_x, normed_p_y)

            score_matrix_length[index1][index2] = 100 * max(0, 1 - abs((p_length-t_length))/t_length)

            """
            get ocr accuracy
            """
            score_matrix_ocr[index1][index2] = get_ocr_score(t_structure['name'], p_structure['name'], cer)

    # for the case of 'no legend'
    try:
        row_ind, col_ind = linear_sum_assignment(score_matrix_line * -1)
    except:
        breakpoint()

    """
    matrix form follows:
    T1 P1 , T1 P2, T1 P3, T1 P4 
    T2 P1 , T2 P2, T2 P3, T2 P4 
    T3 P1 , T3 P2, T3 P3, T3 P4 
    """
    line_accuracy = score_matrix_line[row_ind, col_ind].mean()
    axis_accuracy = score_matrix_axis[row_ind, col_ind].mean()
    length_accuracy = score_matrix_length[row_ind, col_ind].mean()
    ocr_accuracy = score_matrix_ocr[row_ind, col_ind].sum()
    meta_data = ['title', 'xtitle', 'ytitle']
    meta_info_score = sum(get_ocr_score(gt_data[meta_info], pred_data[meta_info], cer) for meta_info in meta_data)
    ocr_accuracy = (ocr_accuracy + meta_info_score)/(len(meta_data) + len(row_ind))
    
    result = {
        "line": line_accuracy,
        "axis": axis_accuracy,
        "length": length_accuracy,
        "ocr": ocr_accuracy,
    }

    return result

def eval_model(args):
    pred_filename_list = [f for f in os.listdir(args.pred) if os.path.splitext(f)[1].lower() == '.json']
    pred_filename_list.sort()

    error_statistics = init_error_statistics() 
    total_result = init_result_dict()
    cer = CharErrorRate()
    for filename in pred_filename_list:
        try:
            with open(os.path.join(args.gt, filename)) as gtf, open(os.path.join(args.pred, filename)) as pf:
                gt_data = json.load(gtf); pred_data = json.load(pf)
        except:
            print(f"gt file for {filename} is missing.")
        
        result = eval_json(pred_data, gt_data, error_statistics, cer)
        print(f"{filename} result : {result}")
        
        total_result["line"] += result['line']
        total_result['axis'] += result['axis']
        total_result['length'] += result['length']
        total_result['ocr'] += result['ocr']

    total_result['line'] /= len(pred_filename_list)
    total_result['axis'] /= len(pred_filename_list)
    total_result['length'] /= len(pred_filename_list)
    total_result['ocr'] /= len(pred_filename_list)

    print("COPY PASTE ME")
    print(f"{total_result['line']} {total_result['axis']} {total_result['length']} {total_result['ocr']}")

    print("total result")
    print(f"line: {total_result['line']:.2f} || axis: {total_result['axis']:.2f} || length: {total_result['length']:.2f} || ocr: {total_result['ocr']:.2f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, type=str, help='json format predcted result')
    parser.add_argument("--gt", required=True, type=str, help='json format ground truth result')
    args = parser.parse_args()

    eval_model(args)
