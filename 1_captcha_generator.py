'''
Author: BNDou
Date: 2024-04-21 16:55:13
LastEditTime: 2024-04-22 16:14:12
FilePath: \Captchas_BOC\1_captcha_generator.py
Description: 
'''

import os
import requests


# 循环下载1-1000张图片
for image_index in range(1, 1001):
    # 图片链接
    image_url = 'https://cdn1.cmcoins.boc.cn/ocas-web/ImageValidation/validation1704292251688.gif'
    # 获取图片数据
    data = requests.get(image_url).content
    # 保存图片的文件夹路径
    folder_path = 'captchas'
    # 图片文件名
    file_name = f'{image_index}.gif'

    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存图片
    with open(f'{folder_path}/{file_name}', 'wb') as f:
        f.write(data)

    # 打印已保存的图片路径
    print(f"Saved：{folder_path}/{file_name}")