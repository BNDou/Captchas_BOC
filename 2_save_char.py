'''
Author: BNDou
Date: 2024-04-21 18:02:48
LastEditTime: 2024-04-22 16:03:28
FilePath: \CmCoins\2_save_char.py
Description: 
'''

import os
import cv2

# 定义一个字符串列表，包含26个英文字母和数字
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '0'
]
# 将字符串列表中的每个字符转换为对应的整数值
class_ords = [ord(class_name) for class_name in class_names]
# 创建一个字典，用于记录每个字符出现的次数
class_cnt = {class_name: 0 for class_name in class_names}


def get_gif_first_frame(gif_path):
    '''
    这个函数用于从给定的gif文件中提取第一帧图片。
    :param gif_path: gif文件的路径
    :return: 返回第一帧图片
    '''
    gif = cv2.VideoCapture(gif_path)
    _, frame = gif.read()
    gif.release()
    return frame


def save_char(target_char):
    '''
    这个函数用于将给定的字符图像保存为文件。
    :param char: 字符图像
    :return: 无返回值
    '''
    # 显示字符图像
    cv2.imshow('target_char', target_char)
    # 等待按键
    flag = cv2.waitKey(0)
    # 如果按键在class_ords中，则保存字符图像
    if flag in class_ords:
        class_name = class_names[class_ords.index(flag)]
        class_cnt[class_name] += 1
        # 如果文件夹不存在，则创建文件夹
        if not os.path.exists(f'chars_dict/{class_name}'):
            os.makedirs(f'chars_dict/{class_name}')
        cv2.imwrite(f'chars_dict/{class_name}/{class_cnt[class_name]}.jpg',
                    target_char)
        print(
            f'字符"{class_name}"已保存，文件名：chars_dict/{class_name}/{class_cnt[class_name]}.jpg'
        )
    # 关闭窗口
    cv2.destroyAllWindows()


def read_image(gif_path):
    # 读取gif图片的第一帧并转换为灰度图像
    cap = get_gif_first_frame(gif_path)
    cap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    cap = cv2.bitwise_not(cap)
    # 将灰度图像二值化（小于127的像素都变为0）
    _, thresh = cv2.threshold(cap, 127, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh, (7, 7), iterations=5)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 初始化一个空列表，用于存储切割后的字符图像
    chars = []
    # 遍历每个轮廓
    for contour in contours:
        # 获取最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 切割出字符图像，并将其添加到 chars 列表中
        char = thresh[y:y + h, x:x + w]
        chars.append(char)

    # 遍历字符列表
    for i in range(len(chars)):
        # 创建一个垂直列表
        vertical = []
        # 遍历字符图像的每一行，并将每一行的元素求和，添加到垂直列表中
        vertical = [
            sum(chars[i][index, :]) for index in range(chars[i].shape[0])
        ]

        # 获取垂直列表中元素为0的索引
        item_cnt = len(vertical)
        zero_vertical_index = [
            index for index, value in enumerate(vertical) if value == 0
        ]
        # 如果垂直列表中没有0，那么第一个索引为0，最后一个索引为字符图像的行数
        if len(zero_vertical_index) != 0:
            if 0 not in zero_vertical_index:
                first_index = 0
                last_index = zero_vertical_index[0] + 1
            # 如果垂直列表中最后一个元素为0，那么第一个索引为字符图像的行数减1，最后一个索引为字符图像的行数减1
            elif item_cnt - 1 not in zero_vertical_index:
                first_index = item_cnt - 1
                last_index = 0
            # 如果垂直列表中既有0，又不是所有元素都为0，那么第一个索引为最后一个元素为0的索引加1，最后一个索引为第一个元素为0的索引减1
            else:
                target = [
                    index
                    for index, value in enumerate(zero_vertical_index[:-1])
                    if zero_vertical_index[index + 1] -
                    zero_vertical_index[index] != 1
                ]
                first_index = zero_vertical_index[target[0]] - 1
                last_index = zero_vertical_index[target[-1] + 1] + 1
        else:
            first_index = 0
            last_index = item_cnt

        # 获取第一个索引和最后一个索引
        (v_f, v_l) = first_index, last_index
        # 获取目标字符图像
        target_char = chars[i][v_f:v_l]

        # 保存字符图像
        if target_char.shape[0] > 0:
            save_char(target_char)


def list_images_in_folder(folder_path):
    # 初始化一个空列表，用于存放图片文件路径
    image_files = []

    # 使用os.walk遍历文件夹及其子文件夹，将所有文件路径添加到image_files列表中
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_files.append(os.path.join(root, file))

    # 返回image_files列表，包含所有图片文件路径
    return image_files


if __name__ == '__main__':
    # read_image('captchas/101.gif')
    for index, image in enumerate(list_images_in_folder('captchas')):
        print(f'saved_char_{index + 1} : {image}')
        read_image(image)
