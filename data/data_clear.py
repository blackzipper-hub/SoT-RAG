import pandas as pd


df = pd.read_csv('./data/retrieve_question.csv')


new_df = df[['outline', 'request']]


new_df.to_csv('question.csv', index=False)

print("新文件已保存为: question.csv")

import csv
import json
import re


input_filename = 'question.csv'
output_filename = 'question.json'


data = []
with open(input_filename, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:

        item = {'request': row['request']}
        
        # 拆分outline字符串
        outline_items = row['outline'].strip().split('\n')
        
        # 处理每个outline项目，提取编号和内容
        for i, outline_item in enumerate(outline_items, 1):
            # 移除数字和点
            content = re.sub(r'^\s*\d+\.\s*', '', outline_item.strip())
            item[f'outline{i}'] = content
        
        data.append(item)


with open(output_filename, 'w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, indent=2, ensure_ascii=False)

print(f"JSON文件已保存为: {output_filename}")
