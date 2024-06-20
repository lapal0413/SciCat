import os
import json
import re
import math

def parse_scientific_notation(value):
    if 'e' in value:
        base, exponent = value.split('e')
        return float(base) * (10 ** float(exponent))
    elif 'E' in value:
        base, exponent = value.split('E')
        return float(base) * (10 ** float(exponent))
    elif '^' in value:
        base, exponent = value.split('^')
        return float(base) * (10 ** float(exponent))
    else:
        return float(value)

def transform_json_content(content, filename):
    # title, ytitle, table parse
    title_match = re.search(r'<title>(.*?)</title>', content)
    ytitle_match = re.search(r'<ytitle>(.*?)</ytitle>', content)
    table_match = re.search(r'<table>(.*?)</table>', content, re.DOTALL)

    if table_match:
        table_content = table_match.group(1)
    else:
        table_match = re.search(r'<table>(.*)', content, re.DOTALL)
        table_content = table_match.group(1) if table_match else ""

    title = title_match.group(1) if title_match else ""
    ytitle = ytitle_match.group(1) if ytitle_match else ""

    # parse data table
    lines = [line for line in table_content.split('\n') if line.strip()]

    headers = lines[0].split('|')
    x_title = headers[0]
    data_series = headers[1:]

    # parse data point
    data_points = []
    for line in lines[1:]:
        values = line.split('|')
        x_value = parse_scientific_notation(values[0].replace(',','').replace(' ','').replace('^','e').replace('\\','').replace("%",'').strip())
        for i, y_value in enumerate(values[1:]):
            if len(data_points) <= i:
                data_points.append({
                    "name": data_series[i].strip(),
                    "type": "",
                    "points": []
                })
            if y_value.strip():  # ignore empty value
                data_points[i]["points"].append([x_value, parse_scientific_notation(y_value.replace(',','').replace(' ','').replace('^','e').replace('\\','').replace("%",'').strip())])

    # generate JSON
    output_data = {
        "title": title.strip(),
        "xtitle": x_title.strip(),
        "ytitle": ytitle.strip(),
        "xmin": "",
        "xmax": "",
        "ymin": "",
        "ymax": "",
        "data": data_points
    }

    return output_data

def process_json_files(input_folder, output_folder, log_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    folder_list = os.listdir(input_folder)
    folder_list.sort()
    for idx, filename in enumerate(folder_list):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_folder, filename)
            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                breakpoint()

            try:    
                content = data['choices'][0]['message']['content']
                transformed_data = transform_json_content(content, filename)
            except Exception as e:
                with open(log_name, 'a') as log_file:
                    log_file.write(f"{idx} : {filename} || {e.__repr__()}\n")
                
            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_data, f, ensure_ascii=False, indent=4)
            print(f"Processed and saved: {output_file_path}")

if __name__ == "__main__":
    input_folder = "input/path"  # input JSON path
    output_folder = "output/path"  # JSON save path

    log_name = f"./{input_folder}_parse_error.txt" 
    with open(log_name, 'w') as log_file:
        log_file.write('Error Log\n')

    process_json_files(input_folder, output_folder, log_name)