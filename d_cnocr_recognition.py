'''
Author: BNDou
Date: 2024-04-30 02:26:42
LastEditTime: 2024-04-30 06:36:15
FilePath: \Captchas_BOC\d_cnocr_recognition.py
Description: 
    使用cnocr库识别验证码
'''
import os
from cnocr import CnOcr
import cv2

# 测试图片的访问路径
IMAGE_PATH = r'.\temp'

if __name__ == '__main__':
    # 读取测试图片的路径
    print("------读取测试图片的路径------")
    imagePaths = []
    # 将每个图片的路径添加到imagePaths列表中
    for label in os.listdir(IMAGE_PATH):
        imagePaths.append(os.path.join(IMAGE_PATH, label))
    imagePaths = sorted(imagePaths)

    # 创建一个OCR对象
    # ocr = CnOcr(rec_model_name="densenet_lite_136-gru",
    #         det_model_name="en_PP-OCRv3_det",
    #         cand_alphabet="2346789ABDEFHJKMNPRTUVWXYZ")
    # ocr = CnOcr(rec_model_name="scene-densenet_lite_136-gru",
    #         det_model_name="en_PP-OCRv3_det",
    #         cand_alphabet="2346789ABDEFHJKMNPRTUVWXYZ")
    ocr = CnOcr(rec_model_name="doc-densenet_lite_136-gru",
            det_model_name="en_PP-OCRv3_det",
            cand_alphabet="2346789ABDEFHJKMNPRTUVWXYZ")
    # ocr = CnOcr(rec_model_name="ch_PP-OCRv3",
    #             det_model_name="en_PP-OCRv3_det",
    #             cand_alphabet="2346789ABDEFHJKMNPRTUVWXYZ")

    # 遍历每个图片的路径
    print("------开始识别------")
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        if img is None:
            gif = cv2.VideoCapture(imagePath)
            _, img = gif.read()
            gif.release()
        char_key = ocr.ocr(img)
        print(
            f"{imagePath}  识别结果：{char_key[0]['text']}  可信度：{char_key[0]['score']}"
        )
