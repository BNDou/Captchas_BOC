'''
Author: BNDou
Date: 2024-04-21 18:02:48
LastEditTime: 2024-04-26 05:11:34
FilePath: \Captchas_BOC\2_char_generator.py
Description: 
    保存字符标签图像
'''

import os
import cv2


# 验证码文件夹路径
IMAGE_PATH = '.\demo\captcha_images'
# 保存字符标签文件夹路径
CHARS_PATH = '.\demo\chars'
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
        if not os.path.exists(f'{CHARS_PATH}/{class_name}'):
            os.makedirs(f'{CHARS_PATH}/{class_name}')
        cv2.imwrite(f'{CHARS_PATH}/{class_name}/{class_cnt[class_name]}.jpg',
                    target_char)
        print(
            f'字符"{class_name}"已保存，文件名：{CHARS_PATH}/{class_name}/{class_cnt[class_name]}.jpg'
        )
    # 关闭窗口
    cv2.destroyAllWindows()


def get_cutted_patches(capchar):
    '''
    这个函数用于读取给定的验证码图片文件，并切割出字符图像。
    :param image: 验证码图片
    :return: 返回切割后的字符图像列表
    '''
    # 读取图片
    img = cv2.imread(capchar)
    if img is None:
        img = get_gif_first_frame(capchar)

    # 将图像二值化
    # 127和128的像素是干扰，都变为255 即白色，其余非白色变黑 即字符
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            b, g, r = img[y, x]
            if (b, g, r) not in [(128, 128, 128), (127, 127, 127),
                                 (255, 255, 255)]:
                img[y, x] = (255, 255, 255)
            else:
                img[y, x] = (0, 0, 0)
    # cv2.imshow('thresh', img)
    # cv2.waitKey(0)

    # 膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 矩形结构
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # dilation = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    dilation = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
    # cv2.imshow('dilation', dilation)
    # cv2.waitKey(0)

    # 获取轮廓
    dilation = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # # 显示轮廓
    # drawContours = cv2.drawContours(cv2.imread(capchar), contours, -1,
    #                                 (0, 0, 255), 2)
    # cv2.imshow('drawContours', drawContours)
    # cv2.waitKey(0)

    # 只保留面积最大的6个轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    # 获取所有外接矩形
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # 排序
    (contours, boundingBoxes) = zip(
        *sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0]))

    # 切割出字符图像
    # 初始化一个空列表，用于存储切割后的字符图像
    cut_chars = []
    for box in boundingBoxes:
        x, y, w, h = box
        char = dilation[y:y + h, x:x + w]
        cut_chars.append(char)
        # cv2.imshow('char', char)
        # cv2.waitKey(0)

    # 遍历字符列表
    for i in range(len(cut_chars)):
        # 创建一个垂直列表
        vertical = []
        # 遍历字符图像的每一行，并将每一行的元素求和，添加到垂直列表中
        vertical = [
            sum(cut_chars[i][index, :]) for index in range(cut_chars[i].shape[0])
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
        target_char = cut_chars[i][v_f:v_l]

        # 保存字符图像
        if target_char.shape[0] > 0:
            save_char(target_char)


def list_images_in_folder(folder_path):
    # 初始化一个空列表，用于存放图片文件路径
    image_files = []
    for label in os.listdir(folder_path):
        image_files.append(os.path.join(folder_path, label))

    # 返回image_files列表，包含所有图片文件路径
    return sorted(image_files)


if __name__ == '__main__':
    # read_image('captchas/101.gif')
    for index, image in enumerate(list_images_in_folder(IMAGE_PATH)):
        print(f'saved_char_{index + 1} : {image}')
        get_cutted_patches(image)
