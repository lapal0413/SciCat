"""
lgc를 위한 entity order dependent camp
"""
import matplotlib.pyplot as plt
from itertools import chain

from torchmetrics.text import CharErrorRate

from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
import numpy as np
import math

import csv
import os
import re
import shutil
import sys

######### DEBUG TOOLS #########
# testnum = 12
# testnum = sys.argv[1]
# os.makedirs(f"./success/test_{testnum}",exist_ok=True)
def print_all_length(lst):
    for i, ls in enumerate(lst):
        print(f"i : {len(ls)}")
###############################

# def rnss_matching():
        # return good

def rnss_distance(x1, x2):
    result = min(1, abs((x1 - x2) / (x1+1e-15)))
    return result

def rnss2d_distance(x1, x2):
    return min(1, abs((((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**0.5)/((x1[0]**2 + x1[1]**2)**0.5+1e-15)))

def rnss2d_compute_cost_matrix(a1, a2,dim=2):
    cost_matrix = np.zeros((len(a1), len(a2)))
    for index1, elt1 in enumerate(a1):
        for index2, elt2 in enumerate(a2):
            if dim == 2:
                cost_matrix[index1, index2] = rnss2d_distance(elt1, elt2)
            elif dim == 1:
                cost_matrix[index1, index2] = rnss_distance(elt1, elt2)
    return cost_matrix

def rnss2d_compute_score(lst1, lst2, dim=2):
    cost_matrix = rnss2d_compute_cost_matrix(lst1, lst2, dim=dim)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind].sum()
    score = 1 - cost / max(len(lst1), len(lst2))
    return score

def is_number(s):
    pattern = re.compile(r"^-?\d+(?:\.\d+)?$") # 음수를 포함한 부동소수점과 정수에 대한 패턴
    return bool(pattern.match(s))

def are_all_numbers(array): # 배열 내 모든 원소가 숫자인지 확인
    return all(is_number(item) for item in array)    

def trajectory_length(func, x_values):
    length = 0
    for i in range(1, len(x_values)):
        dx = x_values[i] - x_values[i - 1]
        dy = func(x_values[i]) - func(x_values[i - 1])
        length += np.sqrt(dx**2 + dy**2)
    return length

def fast_trajectory_length(func, x_values):
    length = np.sum(np.sqrt(np.diff(x_values)**2 + np.diff(func(x_values))**2))
    return length

def check_uniform_length(lst):
    first_length = len(lst[0])
    return all(len(item) == first_length for item in lst)

def check_all_string_floats(arr):
    for item in arr:
        try:
            float(item)
        except ValueError:
            return False
    return True

def contains_nan(t):
    return any(map(np.isnan, t))

def target_searcher(query, text):
    pattern = f'<{query}>(.*?)</{query}>'
    target = re.search(pattern, text)
    if target: # 찾는 것이 있으면 찾는 것, 그거 지운 문자열 반환
        target = target.group(1).strip()
        result_text = re.sub(pattern, "", text)
        return target, result_text
    else: # 찾는 것이 없으면 None,원문 반환
        return target, text

def num_blank_replace(s):
    # Remove spaces and check if the result is a number
    
    if type(s) != str:
        return s
    try:
        float(s.replace(" ", ""))
        is_number = True
    except ValueError:
        is_number = False

    # If it's a number, remove spaces and convert to float, otherwise return original string
    return float(s.replace(" ", "")) if is_number else s

def StringToList(text):
    title, text = target_searcher('title', text)
    xtitle, text = target_searcher('xtitle', text)
    ytitle, text = target_searcher('ytitle', text)
    xmax, text = target_searcher('xmax', text); xmax = num_blank_replace(xmax)
    xmin, text = target_searcher('xmin', text); xmin = num_blank_replace(xmin)
    ymax, text = target_searcher('ymax', text); ymax = num_blank_replace(ymax)
    ymin, text = target_searcher('ymin', text); ymin = num_blank_replace(ymin)
    contents = re.search('<s_answer>(.*?)</s>', text)
    contents = contents.group(1).strip() if contents else "" 
    rows = contents.split('&')
    data_list = [row.strip().split('|') for row in rows]
    data_list = [[item.strip() for item in row] for row in data_list]
    
    return title, xtitle, ytitle, xmax, xmin, ymax, ymin, data_list
    
def diagonal_minimum_check(cost_matrix):
    for i, line in enumerate(cost_matrix):
        if line[i] != min(line):
            return False
    return True

def convert_standard_json_to_camp_list(json_data):
    result_list = []
    error_flag = False

    for item in json_data['data']:
        label = item['name']  # 각 데이터 항목의 이름
        x_values = []
        y_values = []

        # points에서 x, y 값을 분리하고, 가능한 경우 float로 변환
        for point in item['points']:
            # X 값 변환 시도
            if str(point[0]).replace(" ", "") == "":
                continue

            try:
                x = float(str(point[0]).replace(",",""))
            except ValueError:
                error_flag = True
                x = point[0]  # 변환 실패시 원래 값 사용

            try:
                y = float(str(point[1]).replace(",",""))
            except ValueError:
                error_flag = True
                y = point[1]
            # # Y 값 변환 시도
            # try:
            #     y = float(point[1])
            # except ValueError:
            #     y = point[1]  # 변환 실패시 원래 값 사용

            x_values.append(x)
            y_values.append(y)

        # 최종적으로 NumPy 배열로 변환
        x_val = np.array(x_values)
        y_val = np.array(y_values)
        
        # 결과 리스트에 사전 형식으로 추가
        result_list.append({'label': label, 'x_val': x_val, 'y_val': y_val})
    
    return result_list,error_flag

def eval_string(num_string):
    if ',' in num_string:
        if len(num_string.split(',')[-1]) <= 2:
            num_string = num_string.replace(',','.')
        else:
            num_string = num_string.replace(',','')
    num = eval(num_string.replace('M','*1000000').replace('%','').replace('k','*1000').replace('\\mu','').replace('\\times','').replace(' ','').replace('x','*').replace('X','*').replace('^','**').replace('{','').replace('}',''))
    return num

# nan의 존재를 고려해야함.
# x 노말라이즈도 해주어야함.
# x가 string일 때도 고려해야함.
# def camp_accuracy(prediction_string, target_string, steps=100): # chart average mean precision
legend_confuse_count = 0
error_statistics = {"wrong_pred_format": 0, "wrong_target_format": 0, "xtick_contains_word": 0, "contain_not_a_number":0, "t_p_len_not_match": 0, "no_x_overlap": 0, "area_over_than_1": 0, "too_much_x_disparity": 0, "nan_only": 0, "only_one_point_detected": 0}
error_result = 0, 0, 0, 0, 0, 0, 0
not_count_result = None
def camp_accuracy(prediction_string, gt_result, gt_title, gt_xtitle, gt_ytitle, gt_xmax, gt_xmin, gt_ymax, gt_ymin, target_json, prediction_json, aimmo_data=None, steps=100, filename=None, visualize_name=None, autosave=False, legend_confuse_check=False): # chart average mean precision
    # p_title, p_xtitle, p_ytitle, p_xmax, p_xmin, p_ymax, p_ymin, p_data_list = StringToList(prediction_string.replace("–","-"))
    # t_title, t_xtitle, t_ytitle, t_xmax, t_xmin, t_ymax, t_ymin, t_data_list = StringToList(target_string.replace("–","-"))
    
    ######### AIMMO #########
    t_title, t_xtitle, t_ytitle, t_xmax, t_xmin, t_ymax, t_ymin, t_data_list = target_json['title'], target_json['xtitle'], target_json['ytitle'], eval_string(aimmo_data['Annotations']['meta'][0]['x-tick']['max']['value']), eval_string(aimmo_data['Annotations']['meta'][0]['x-tick']['min']['value']), eval_string(aimmo_data['Annotations']['meta'][0]['y-tick']['max']['value']), eval_string(aimmo_data['Annotations']['meta'][0]['y-tick']['min']['value']), None
    #########################

    ######### SYNCAT #########
    # t_title, t_xtitle, t_ytitle, t_xmax, t_xmin, t_ymax, t_ymin, t_data_list = target_json['title'], target_json['xtitle'], target_json['ytitle'], gt_xmax, gt_xmin, gt_ymax, gt_ymin, None
    #########################

    ######### SYNCAT #########

    # if "AT&T" in gt_result:
    #     gt_result = gt_result.replace("&","N")

    # try:
    #     t_title, t_xtitle, t_ytitle, _, _, _, _, t_data_list = StringToList('<s_answer>'+gt_result.replace("–","-"))    
    #     t_data_array_T = np.array(t_data_list).T 
    #     t_xtick = t_data_array_T[0, 1:]
    # except:
    #     breakpoint()
    # t_structure_list = [ # list of dict
    #     {
    #         'label': row[0],
    #         'x_val': np.array(t_xtick[row[1:] != 'nan'], dtype=float),
    #         'y_val': np.array(row[1:][row[1:] != 'nan'], dtype=float)
    #     }
    #     for row in t_data_array_T[1:]
    # ]
    
    #########################

    ######### AIMMO #########
    try:
        t_structure_list, error_flag = convert_standard_json_to_camp_list(target_json)
        if error_flag:
            error_statistics["wrong_target_format"] += 1
            return error_result
    except:
        error_statistics["wrong_target_format"] += 1
        return error_result
    #########################

    try:
        p_structure_list, error_flag = convert_standard_json_to_camp_list(prediction_json)
        if error_flag:
            error_statistics["wrong_pred_format"] += 1
            return error_result
    except:
        error_statistics["wrong_pred_format"] += 1
        return error_result
        
        

    dict_forvis = {"normed_t_x": [], "normed_t_y": [], "normed_p_x": [], "normed_p_y": []}
    line_accuracy = 0
    axis_accuracy = 0
    length_accuracy = 0
    ocr_accuracy = 0
    rnss_accuracy = 0
    rnss2d_accuracy = 0
    ordered_rnss2d_accuracy = 0

    if len(t_structure_list) != len(p_structure_list):
        # min_num = min(len(t_structure_list),len(p_structure_list))
        # t_structure_list = t_structure_list[:min_num]
        # p_structure_list = p_structure_list[:min_num]
        error_statistics["t_p_len_not_match"] += 1
        return error_result
    

    cost_matrix_line = np.zeros((len(t_structure_list), len(p_structure_list)))
    cost_matrix_length = np.zeros((len(t_structure_list), len(p_structure_list)))
    cost_matrix_axis = np.zeros((len(t_structure_list), len(p_structure_list)))

    for index1, t_structure in enumerate(t_structure_list):
        for index2, p_structure in enumerate(p_structure_list):
            if len(p_structure['x_val']) == 0 and len(p_structure['y_val']) == 0: # 모두 nan 인 경우
                error_statistics["nan_only"] += 1
                # breakpoint()
                # return error_result
                cost_matrix_line[index1][index2] = 0
                cost_matrix_length[index1][index2] = 0
                cost_matrix_axis[index1][index2] = 0
                break

            interp_func = lambda x, y: interp1d(x, y, kind='linear')        
            
            if len(t_structure['x_val']) == 0 and len(t_structure['y_val']) == 0: # pred <-> pred comparison 에서 t에 문제 생길수도 있음. 다 nan이어서 값이 없는 경우
                error_statistics["nan_only"] += 1
                return error_result

            try:            
                p_structure['x_val'] = np.array([float(str(x).replace(',', '').strip()) for x in p_structure['x_val']])
            except:
                return error_result
            t_structure['x_val'] = np.array([float(str(x).replace(',', '').strip()) for x in t_structure['x_val']])

            t_func = interp_func(t_structure['x_val'], t_structure['y_val'])
            try:
                p_func = interp_func(p_structure['x_val'], p_structure['y_val']) 
            except:
                error_statistics["only_one_point_detected"] += 1
                # return error_result
                cost_matrix_line[index1][index2] = 0
                cost_matrix_length[index1][index2] = 0
                cost_matrix_axis[index1][index2] = 0
                break

            p_xlist = np.array([float(str(x).replace(',', '').strip()) for x in p_structure['x_val']])
            try:
                t_xlist = np.array([float(str(x).replace(',', '').strip()) for x in t_structure['x_val']])
            except:
                breakpoint()

            p_xlist_min, p_xlist_max = p_xlist[0], p_xlist[-1]
            t_xlist_min, t_xlist_max = t_xlist[0], t_xlist[-1]

            if max(p_xlist_min, t_xlist_min) > min(p_xlist_max, t_xlist_max): # 둘이 겹치는 구간이 아예 없는 경우
                error_statistics["no_x_overlap"] += 1
                # return error_result
                cost_matrix_line[index1][index2] = 0
                cost_matrix_length[index1][index2] = 0
                cost_matrix_axis[index1][index2] = 0
                break

            minmax_list = sorted([p_xlist_min, t_xlist_min, p_xlist_max, t_xlist_max])
            max_max = [minmax_list[0],minmax_list[-1]]
            min_min = [minmax_list[1],minmax_list[2]]

            # axis_accuracy += 100 * abs((min_min[1]-min_min[0]) / (max_max[1]-max_max[0]))
            cost_matrix_axis[index1][index2] = 100 * abs((min_min[1]-min_min[0]) / (max_max[1]-max_max[0]))

            x = np.linspace(min_min[0],min_min[1], steps)    
            dx = (min_min[1] - min_min[0]) / (steps - 1)

            try:
                area_cost = np.sum(np.abs(p_func(x)-t_func(x)))*dx
            except:
                breakpoint()
            try:
                normed_cost_area = 100*(area_cost / (t_ymax - t_ymin) / (t_xmax - t_xmin)) 
            except:
                breakpoint()
            if normed_cost_area > 100: # 노말라이즈 스페이스를 뛰어넘어서 면적이 1이 넘어가버리는 경우
                error_statistics["area_over_than_1"] += 1
                # return error_result
                cost_matrix_line[index1][index2] = 0
                cost_matrix_length[index1][index2] = 0
                cost_matrix_axis[index1][index2] = 0
                break
            
            # line_accuracy += 100 - normed_cost_area
            cost_matrix_line[index1][index2] = 100 - normed_cost_area

            # 함수 normalize space로 옮기기.
            my_interpolation = lambda x, a1, a2, b1, b2: (x - a1) * (b2 - b1) / (a2 - a1) + b1
            normed_t_x = my_interpolation(t_structure['x_val'],t_xmin, t_xmax, 0, 1)
            normed_t_y = my_interpolation(t_structure['y_val'],t_ymin, t_ymax, 0, 1)
            normed_p_x = my_interpolation(p_structure['x_val'],t_xmin, t_xmax, 0, 1)
            normed_p_y = my_interpolation(p_structure['y_val'],t_ymin, t_ymax, 0, 1)
        
            dict_forvis['normed_t_x'].append(normed_t_x);dict_forvis['normed_t_y'].append(normed_t_y)
            dict_forvis['normed_p_x'].append(normed_p_x);dict_forvis['normed_p_y'].append(normed_p_y)

            NORMED_X_MARGIN = 1
            if np.any(normed_p_x > 1+NORMED_X_MARGIN) or np.any(normed_p_x < 0-NORMED_X_MARGIN): # target의 x의 범위를 너무 많이 벗어났음.
                error_statistics["too_much_x_disparity"] += 1
                # return error_result
                cost_matrix_line[index1][index2] = 0
                cost_matrix_length[index1][index2] = 0
                cost_matrix_axis[index1][index2] = 0
                break

            get_length = lambda x, y: np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            t_length = get_length(normed_t_x, normed_t_y)
            p_length = get_length(normed_p_x, normed_p_y)

            # length_accuracy += max(0, 1 - abs((p_length-t_length))/t_length)
            cost_matrix_length[index1][index2] = 100* max(0, 1 - abs((p_length-t_length))/t_length)
            
    cost_matrix_line_for_index = cost_matrix_line * -1
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix_line_for_index)
    except:
        error_statistics["contain_not_a_number"] += 1
        return error_result
    line_accuracy = cost_matrix_line[row_ind, col_ind].sum()
    axis_accuracy = cost_matrix_axis[row_ind, col_ind].sum()
    length_accuracy = cost_matrix_length[row_ind, col_ind].sum()

    pred_xy_points = []
    pred_xy_points_list = []
    for p_structure in p_structure_list:
        pred_xy_points_list.append([[x_point, y_point] for x_point, y_point in zip(p_structure['x_val'], p_structure['y_val'])] )
    pred_xy_points = list(chain.from_iterable(pred_xy_points_list))

    gt_xy_points = []    
    gt_xy_points_list = []    
    for t_structure in t_structure_list:
        gt_xy_points_list.append([[x_point, y_point] for x_point, y_point in zip(t_structure['x_val'], t_structure['y_val'])])
    gt_xy_points = list(chain.from_iterable(gt_xy_points_list))
    
    rnss_accuracy = rnss2d_compute_score([point for points in gt_xy_points for point in points], [point for points in pred_xy_points for point in points], dim=1)
    rnss_accuracy *= 100

    rnss2d_accuracy = rnss2d_compute_score(gt_xy_points, pred_xy_points)
    rnss2d_accuracy *= 100

    for gt, pred in zip(gt_xy_points_list, pred_xy_points_list):
        ordered_rnss2d_accuracy += rnss2d_compute_score(gt, pred)
    ordered_rnss2d_accuracy /= len(gt_xy_points_list)
    ordered_rnss2d_accuracy *= 100        

    num_of_data = len(t_structure_list)
    line_accuracy /= num_of_data
    axis_accuracy /= num_of_data
    length_accuracy /= num_of_data

    # t_legend_list = t_data_list[0][1:]
    # p_legend_list = p_data_list[0][1:]

    # t_text_list = []
    # p_text_list = []
    # if t_title != None:
    #     t_text_list.append(t_title)
    #     p_text_list.append(p_title if p_title != None else "")

    # if t_xtitle != None:
    #     t_text_list.append(t_xtitle)
    #     p_text_list.append(p_xtitle if p_xtitle != None else "")

    # if t_ytitle != None:
    #     t_text_list.append(t_ytitle)
        # p_text_list.append(p_ytitle if p_ytitle != None else "")
    
    # t_text_list += t_legend_list
    # p_text_list += p_legend_list
    
    t_text_list = [target_json['title'], target_json['xtitle'], target_json['ytitle'], *[my_dict['label'] for my_dict in t_structure_list]]
    # t_text_list = [target_json['title'], target_json['xtitle'], target_json['ytitle'], *[my_dict['name'] for my_dict in target_json['data']]]
    p_text_list = [prediction_json['title'], prediction_json['xtitle'], prediction_json['ytitle'], *[my_dict['name'] for my_dict in prediction_json['data']]]

    cer = CharErrorRate()
    for p_text, t_text in zip(p_text_list, t_text_list):
        p_text, t_text = str(p_text), str(t_text)
        ocr_accuracy += 1 - min(1, cer([p_text], [t_text]).item())
            
    ocr_accuracy /= len(t_text_list)
    ocr_accuracy *= 100

    if legend_confuse_check:
        cost_matrix = np.array([[cer([t_legend],[p_legend]).item() for t_legend in t_legend_list] for p_legend in p_legend_list])
        # legend_confuse_count += 0 if diagonal_minimum_check(cost_matrix) and diagonal_minimum_check(cost_matrix.T) == True else 1
        global legend_confuse_count
        if not (diagonal_minimum_check(cost_matrix) and diagonal_minimum_check(cost_matrix.T)):
            legend_confuse_count += 1
            print(legend_confuse_count)
    
    # if autosave==True and line_accuracy > 90 and axis_accuracy > 90 and ocr_accuracy > 90:
    if autosave==True:
       visualize_name = os.path.splitext(filename)[0]

    if visualize_name != None:
        plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
        # target 그리기
        plt.subplot(1, 2, 1)
        for i in range(len(t_legend_list)):
            plt.plot(dict_forvis['normed_t_x'][i], dict_forvis['normed_t_y'][i], label=t_legend_list[i])
        plt.title(t_title)
        plt.xlabel(t_xtitle)
        plt.ylabel(t_ytitle)
        
        plt.legend()
        plt.xlim(0,1)
        plt.ylim(0,1)

        # predict 그리기
        plt.subplot(1, 2, 2)
        for i in range(len(p_legend_list)):
            plt.plot(dict_forvis['normed_p_x'][i], dict_forvis['normed_p_y'][i], label=p_legend_list[i])
        plt.title(p_title)
        plt.xlabel(p_xtitle)
        plt.ylabel(p_ytitle)
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend()

        plt.tight_layout()
        # plt.savefig(f"./success/test_{testnum}/{visualize_name}.png") 
        plt.savefig(f"./normal_vis/{visualize_name}.png")
        plt.close('all') 

        ##### for debug #####
        # shutil.copy2(f"/shared/workspace/code/vichart/data/camp_data/augset_{testnum}/{visualize_name}.png",f'./success/test_{testnum}/{visualize_name}_original.png')
        #####################
    
    return line_accuracy, axis_accuracy, length_accuracy, ocr_accuracy, rnss2d_accuracy, ordered_rnss2d_accuracy, rnss_accuracy


import glob
import json

pred_folder_path = '/shared/nips/syn_unichart_std'

pred_json_files = glob.glob(os.path.join(pred_folder_path, '*.json'))

aimmo_json_path = '/shared/nips/aimmo_test_dst/sampled_jsons'
aimmo_json_files = glob.glob(os.path.join(aimmo_json_path, '*.json'))

# FIXME 나중에 이거 지워야함.
# aimmo_json_files = pred_json_files

pred_folder_path2, pred_json_files2 = None, None

gt_folder_path = '/shared/nips/aimmo_test_dst/standard_jsons'


gt_json_files = glob.glob(os.path.join(gt_folder_path, '*.json'))

pred_json_files.sort()
gt_json_files.sort()
aimmo_json_files.sort()

# gt_labels = []
# prediction_labels = []
line_accuracy_total, axis_accuracy_total, length_accuracy_total, ocr_accuracy_total, rnss_accuracy_total, rnss2d_accuracy_total, ordered_rnss2d_accuracy_total = 0, 0, 0, 0, 0, 0, 0
total_num = 0

"""
for gt_file, prediction_file in zip(gt_json_files, pred_json_files):
    # if os.path.basename(gt_file) != "val_00209.json":
    #     continue
    # 파일 이름이 같은지 확인
    if os.path.basename(gt_file) == os.path.basename(prediction_file):
"""
# unichart zeroshot
# Characteristic | [unnamed data series #0] | [unnamed data series #1] & 10.0 | 257.0 | 237.0 & 20.0 | 102.0 | 62.0 & 30.0 | 52.0 | 52.0 & 40.0 | 10.0 | 30.0 & 50.0 | 20.0 | 20.0 & 60.0 | 30.0 | 30.0 & 70.0 | 30.0 | 30.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0 & 80.0 | 50.0 | 50.0
for prediction_file in pred_json_files:
    for gt_file in gt_json_files: 
        if os.path.basename(gt_file) == os.path.basename(prediction_file):
            # if os.path.basename(gt_file) == "25_24.json" or os.path.basename(gt_file) == "25_25.json" or os.path.basename(gt_file) == "25_26.json":
            #     continue
            
            # if os.path.basename(gt_file) != '1_10.json':
            #     continue

            # cycle = [file.split(".")[0] for file in os.listdir('/shared/workspace/testset/divided/cycle') if file.endswith('.png')]
            # idk = [file.split(".")[0] for file in os.listdir('/shared/workspace/testset/divided/idk') if file.endswith('.png')]
            # jump = [file.split(".")[0] for file in os.listdir('/shared/workspace/testset/divided/jump') if file.endswith('.png')]

            # if os.path.basename(gt_file).split(".")[0] not in cycle:
            #     continue

            # if os.path.basename(gt_file) != '001240.json':
            #     continue 
            with open(gt_file, 'r') as gtf, open(prediction_file, 'r') as pf:
                gt_data = json.load(gtf)
                prediction_data = json.load(pf)        
            
            ###########################   WARNING 여기 주석 풀어야함    ####################################
            with open(os.path.join(aimmo_json_path,gt_file.split('/')[-1]), 'r') as aimmo_json:
                aimmo_data = json.load(aimmo_json)
            ################################################################################################

            # prediction_result = prediction_data.get('result') or prediction_data.get('results') or prediction_data.get('text')
            # prediction_result = prediction_result if prediction_result.endswith("</s>") else prediction_result + "</s>"
            # prediction_result = prediction_result if "<s_answer>" in prediction_result else '<s_answer>' + prediction_result
            prediction_result = ""

            gt_title = gt_data.get('title', None)
            gt_xtitle = gt_data.get('xtitle', None)
            gt_ytitle = gt_data.get('ytitle', None)
            gt_xmax = gt_data.get('xmax', None)
            gt_xmin = gt_data.get('xmin', None)
            gt_ymax = gt_data.get('ymax', None)
            gt_ymin = gt_data.get('ymin', None)
            gt_result = gt_data.get('train_string', None)
            # gt_result = "<s_answer>" + gt_result    

            print(os.path.basename(gt_file))
            ######### AIMMO #########
            # results = camp_accuracy(prediction_result, gt_result, gt_title, gt_xtitle, gt_ytitle, gt_xmax, gt_xmin, gt_ymax, gt_ymin, gt_data, prediction_data, aimmo_data=None, filename=os.path.basename(gt_file), autosave=False, legend_confuse_check=False)
            #########################

            ######### SYNCAT #########
            results = camp_accuracy(prediction_result, gt_result, gt_title, gt_xtitle, gt_ytitle, gt_xmax, gt_xmin, gt_ymax, gt_ymin, gt_data, prediction_data, aimmo_data, filename=os.path.basename(gt_file), autosave=False, legend_confuse_check=False)
            ##########################
            print(f"{results} | file name : {os.path.basename(gt_file)}")
            if results is not not_count_result:
                if not contains_nan(results):
                    line_accuracy, axis_accuracy, length_accuracy, ocr_accuracy, rnss2d_accuracy, ordered_rnss2d_accuracy, rnss_accuracy = results

                    line_accuracy = max(min(line_accuracy,100),0)
                    axis_accuracy = max(min(axis_accuracy,100),0)
                                        
                    total_num += 1
                    line_accuracy_total += line_accuracy
                    axis_accuracy_total += axis_accuracy
                    length_accuracy_total += length_accuracy
                    ocr_accuracy_total += ocr_accuracy    
                    rnss_accuracy_total += rnss_accuracy
                    rnss2d_accuracy_total += rnss2d_accuracy
                    ordered_rnss2d_accuracy_total += ordered_rnss2d_accuracy
            break

if total_num == 0:
    raise Exception("May be wrong input file name")
print(f'pred_folder_path: {pred_folder_path}\ngt_folder_path : {gt_folder_path}')
print(f"final result || total num : {total_num}")                    
print(line_accuracy_total/total_num, axis_accuracy_total/total_num, length_accuracy_total/total_num, ocr_accuracy_total/total_num, rnss2d_accuracy_total/total_num, ordered_rnss2d_accuracy_total/total_num, rnss_accuracy_total/total_num)
print(error_statistics)

print(f"line : {line_accuracy_total/total_num}||axis :  {axis_accuracy_total/total_num}||length : {length_accuracy_total/total_num}||ocr : {ocr_accuracy_total/total_num}||rnss 2d : {rnss2d_accuracy_total/total_num}||ord rnss 2d : {ordered_rnss2d_accuracy_total/total_num}||rnss : {rnss_accuracy_total/total_num}")
