'''
Author: BNDou
Date: 2024-04-25 01:54:43
LastEditTime: 2024-04-30 07:42:12
FilePath: \Captchas_BOC\CyclicVerification.py
Description: 
    循环测试，使用CnOCR API识别验证码
'''

import os
import sys
import time
import cv2
from cnocr import CnOcr
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# 获取当前脚本所在目录的路径
base_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
options = Options()
# 设置 Edge 以无窗口模式运行
# options.add_argument("--headless")
# options.add_argument("--disable-gpu")

# 创建模型和标签
print("------创建模型和标签------")
# ocr = CnOcr(cand_alphabet="2346789ABDEFHJKMNPRTUVWXYZ")
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


class CaptchaOCR_Keras:
    '''
    这个类用于识别验证码。
    '''

    def __init__(self, capchar):
        self.capchar = capchar

    def get_gif_first_frame(self):
        '''
        这个函数用于从给定的gif文件中提取第一帧图片。
        :param gif_path: gif文件的路径
        :return: 返回第一帧图片
        '''
        gif = cv2.VideoCapture(self.capchar)
        _, frame = gif.read()
        gif.release()
        return frame

    def recognize_capchar(self):
        '''
        这个函数用于识别给定的验证码图片。
        :param path: 验证码图片的路径
        :return: 返回识别结果
        '''
        img = cv2.imread(self.capchar)
        if img is None:
            gif = cv2.VideoCapture(self.capchar)
            _, img = gif.read()
            gif.release()
        # char_key = ocr.ocr(img)
        # return char_key[0]['text']
        char_key = ocr.ocr_for_single_line(img)
        return char_key['text']


def find_and_click(driver, xpath):
    '''
    这个函数用于在给定的网页元素中查找并点击。
    :param driver: 浏览器驱动
    :param xpath: 网页元素的xpath
    '''
    WebDriverWait(driver, 20,
                  0.5).until(EC.presence_of_element_located((By.XPATH, xpath)))
    driver.find_element(By.XPATH, xpath).click()


def find_and_captcha(driver, capchar_xpath):
    '''
    这个函数用于在给定的网页元素中查找并识别验证码。
    :param driver: 浏览器驱动
    :param capchar_xpath: 验证码网页元素的xpath
    '''
    while True:
        # 等待验证码标签加载
        WebDriverWait(driver, 20, 0.5).until(
            EC.presence_of_element_located((By.XPATH, capchar_xpath)))
        ele_piccaptcha = driver.find_element(By.XPATH, capchar_xpath)
        # 然后直接调用这个元素的screenshot方法，参数是保存的路径即可实现截图
        ele_piccaptcha.screenshot(base_dir + '\\temp\\temp_capchar.jpg')
        # 然后调用识别函数
        chars_key = CaptchaOCR_Keras(
            base_dir + '\\temp\\temp_capchar.jpg').recognize_capchar()
        if chars_key is None:
            print("验证码识别失败，刷新验证码，重新识别")
            find_and_click(driver, capchar_xpath)
        else:
            break

    print(f'识别结果为：{chars_key}')
    # 在网页上插入一个<div>元素，用于显示自定义文本
    driver.execute_script(f'''
        var customText = document.createElement('div');
        customText.id = 'customText';
        customText.innerHTML = '{chars_key}';
        customText.style.cssText = 'position: absolute; top: 53%; left: 50%; transform: translate(-50%, -50%); background-color: white; color: black; font-size: 24px; font-weight: bold;';
        document.body.appendChild(customText);
    ''')


def main(driver):
    '''
    这个函数用于主函数。
    '''
    url = 'https://cdn1.cmcoins.boc.cn/ocas-web/ImageValidation/validation1704292251688.gif'
    driver.get(url)

    # 解析验证码
    while True:
        find_and_captcha(
            driver,
            '/html/body/img'
        )
        time.sleep(1)
        driver.refresh()


if __name__ == '__main__':
    # 初始化 WebDriver
    driver = webdriver.Edge(service=Service(r'msedgedriver.exe'),
                            options=options)
    main(driver)
