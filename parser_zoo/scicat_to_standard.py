import os
import json

# define eval func
def eval_num(value):
    return float(value)

# define transform
def transform_json(input_data):
    transformed = {
        "title": input_data['meta'].get('title', ''),
        "xtitle": input_data['meta'].get('x-title', ''),
        "ytitle": input_data['meta'].get('y-title', ''),
        "xmin": input_data['meta']['x-tick']['min']['value'],
        "xmax": input_data['meta']['x-tick']['max']['value'],
        "ymin": input_data['meta']['y-tick']['min']['value'],
        "ymax": input_data['meta']['y-tick']['max']['value'],
        "data": []
    }
    
    for item in input_data['data']:
        new_item = {
            "name": item['name'][0] if item['name'] else '',
            "type": item['type'],
            "points": [list(map(eval_num, point[:2])) for point in item['value']]
        }
        transformed["data"].append(new_item)
    
    return transformed

def transform_json_files(folder_path, save_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                transformed_data = transform_json(data)
                output_filename = f"{filename}"
                output_path = os.path.join(save_path, output_filename)
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    json.dump(transformed_data, output_file, indent=4)

save_path = "./save_path"
folder_path = "/mnt/data"
os.makedirs(save_path, exist_ok=True)
transform_json_files(folder_path, save_path)
