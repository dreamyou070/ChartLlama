import os
import json
import random

chart_type_questions = ['what is the type of this chart?',
                        "what this chart type?",
                        "can you explain about this chart type?"]

title_questios = ['what is the title of this chart?',
                  "what this chart title?",
                  "can you explain about this"]

x_axis_title_questions = ["what is the x-axis title of this chart?",
                          "what this chart x-axis title?",
                          "can you explain about this"]

y_axis_title_questions = ['what is the y-axis title of this chart?',
                          "what this chart y-axis title?",
                          "can you explain about this"]

x_range_questions = ['what is the x-axis range of this chart?',
                        "what this chart x-axis range?",
                        "can you explain about this"]
y_range_questions = ['what is the y-axis range of this chart?',
                        "what this chart y-axis range?",
                        "can you explain about this"]

original_path = 'data_train.json'
with open(original_path, 'r') as f:
    data = json.load(f)

file_name = os.path.split(original_path)[0]
make_path = f'{file_name}_simplified.json'
total_data = []
for file in data :
    # caption_id
    img_name = file['img_id']
    img_path = f'images/{img_name}.png'

    # [1] title
    structure_properties = file['L1_properties']

    chart_type = structure_properties[0]
    title = structure_properties[1]
    x_axis_title = structure_properties[2]
    y_axis_title = structure_properties[3]
    x_range = structure_properties[4]
    y_range = structure_properties[5]
    contents = {'chart_type' : [chart_type,chart_type_questions],
                'title' : [title,title_questios],
                'x_axis_title' : [x_axis_title,x_axis_title_questions],
                'y_axis_title' : [y_axis_title,y_axis_title_questions],
                'x_range' : [x_range,x_range_questions],
                'y_range' : [y_range,y_range_questions]}

    for k in contents.keys():
        element = {}
        element['id'] = img_name
        element['image'] = img_path
        element['model'] = ''
        element['conversations'] = []
        quesion_list =contents[k][-1]
        random_index = random.randint(0, len(quesion_list)-1)
        element['conversations'].append({'from': 'human',
                                         'value': f'<image>\n{quesion_list[random_index]}'})
        element['conversations'].append({'from': 'gpt',
                                         'value': contents[k][0]})

        total_data.append(element)

with open(make_path, 'w') as f:
    json.dump(total_data, f)

