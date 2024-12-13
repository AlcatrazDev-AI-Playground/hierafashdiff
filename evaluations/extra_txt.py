import json
import os
from tqdm import tqdm
# 指定TXT 文件所在的文件夹路径

txt_folder = './text_foleder/lle_a5'
os.makedirs(txt_folder,exist_ok=True)

# 读取 JSON 文件
json_file = '/data/lh/docker/dataset/HieraFashion_5K/test.json'

with open(json_file, 'r') as f:
    data = json.load(f)

# 遍历每个条目，将 caption 写入到对应的 txt 文件中
for item in tqdm(data):
    gt_filename = os.path.basename(item['gt'])  # 提取文件名
    txt_filename = os.path.splitext(gt_filename)[0] + '_0.txt'  # 构造对应的 txt 文件名

    # 构造对应的 txt 文件路径
    txt_file_path = os.path.join(txt_folder, txt_filename)

    caption = ','.join(item.strip() for item in item["caption"].split(',')[:10])
    # caption = item["caption"]

    # 写入 caption 到对应的 txt 文件中
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(caption)
