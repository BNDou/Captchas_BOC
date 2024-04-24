'''
Author: BNDou
Date: 2024-04-21 16:55:13
LastEditTime: 2024-04-24 23:57:48
FilePath: \Captchas_BOC\1_captcha_generator.py
Description: 
  使用requests库下载BOC验证码图片
'''

import os
import requests


# 保存图片的文件夹路径
SAVE_IMAGE_PATH = './captchas/'


# 循环下载1-1000张图片
for image_index in range(1, 1001):
    # 图片链接
    image_url = 'https://cdn1.cmcoins.boc.cn/ocas-web/ImageValidation/validation1704292251688.gif'
    # 获取图片数据
    data = requests.get(image_url).content
    # 图片文件名
    file_name = f'{image_index}.gif'

    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(SAVE_IMAGE_PATH):
        os.makedirs(SAVE_IMAGE_PATH)

    # 保存图片
    with open(f'{SAVE_IMAGE_PATH}/{file_name}', 'wb') as f:
        f.write(data)

    # 打印已保存的图片路径
    print(f"Saved：{SAVE_IMAGE_PATH}/{file_name}")