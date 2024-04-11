import os
import json
import random

questions = ['what is the content of this chart?',
            "what this chart says?",
            "can you explain about this chart?",
            "what is the chart about?",
            "what is the chart showing?",
            "what is the chart representing?",
            "what is the chart indicating?",
            "what is the chart displaying?",
            "what is the chart illustrating?",
            "what is the chart depicting?",
            "can you tell me about this chart?",
            "what is the chart about?",]
original_path = 'data_train.json'
with open(original_path, 'r') as f:
    data = json.load(f)

file_name = os.path.split(original_path)[0]
make_path = f'{file_name}_simplified.json'
for file in data :
    # caption_id
    img_name = file['img_id']
    img_path = f'images/{img_name}.png'

    # [2] screengraph
    screengraph = file['scenegraph']
    screengraph = screengraph.split(' x ')[0]
    title = screengraph.split('title ')[-1].strip()

    # [3] data table

    print(title)
    break
"""
total_data = []

for file in data:
    img_name = file['img_id']
    caption = file['caption_L2L3']
    img_path = f'images/{img_name}.png'

    new_dict = {}
    new_dict['id'] = img_name
    new_dict['image'] = img_path
    new_dict['model'] = ''
    new_dict['conversations'] = []
    random_index = random.randint(0, len(questions)-1)
    question = questions[random_index]
    new_dict['conversations'].append({'from': 'human', 'value': f'<image>\n{question}'})
    new_dict['conversations'].append({'from': 'gpt', 'value': caption})
    total_data.append(new_dict)

with open(make_path, 'w') as f:
    json.dump(total_data, f)

"""