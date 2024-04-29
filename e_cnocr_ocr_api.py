'''
Author: BNDou
Date: 2024-04-25 01:54:43
LastEditTime: 2024-04-30 06:53:12
FilePath: \Captchas_BOC\e_cnocr_ocr_api.py
Description: 
    使用CnOCR API识别验证码
'''

import json
import os
import sys
import threading
import cv2
from cnocr import CnOcr
import numpy as np
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
# 读取用户信息
print("------读取用户信息------")
with open(base_dir + '\\config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
print(config)


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


def find_and_fill(driver, xpath, value):
    '''
    这个函数用于在给定的网页元素中查找并填写指定的值。
    :param driver: 浏览器驱动
    :param xpath: 网页元素的xpath
    :param value: 需要填写的值
    '''
    WebDriverWait(driver, 20,
                  0.5).until(EC.presence_of_element_located((By.XPATH, xpath)))
    driver.find_element(By.XPATH, xpath).send_keys(value)


def find_and_captcha(driver, capchar_xpath, input_xpath):
    '''
    这个函数用于在给定的网页元素中查找并识别验证码。
    :param driver: 浏览器驱动
    :param capchar_xpath: 验证码网页元素的xpath
    :param input_xpath: 验证码输入框的xpath
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

    # 识别成功后，点击验证码输入框，并填写识别结果
    print(f'识别结果为：{chars_key}')
    # 填写验证码
    find_and_fill(driver, input_xpath, chars_key)


def main(driver, name, id_num, phone_num):
    '''
    这个函数用于主函数。
    '''
    # url = 'https://cdn1.cmcoins.boc.cn/ocas-web/ImageValidation/validation1704292251688.gif'
    # driver.get(url)
    # 先找到验证码对应的网页元素
    # WebDriverWait(driver, 10, 0.5, ignored_exceptions='未找到验证码').until(
    #     EC.presence_of_element_located((By.XPATH, '/html/body/img')))
    # ele_piccaptcha = driver.find_element(By.XPATH, '/html/body/img')
    # 然后直接调用这个元素的screenshot方法，参数是保存的路径即可实现截图
    # ele_piccaptcha.screenshot('./temp/temp_capchar.jpg')
    # # 然后调用识别函数
    # chars_key = CaptchaOCR_Keras('./temp/temp_capchar.jpg', MODEL,
    #                              LB).recognize_capchar()
    # print(f'识别结果为：{chars_key}')

    url = 'https://gces.bankofchina.com/'
    driver.get(url)
    driver.execute_script(f"document.title = '{name}';")

    # 解析验证码
    find_and_captcha(
        driver,
        '//*[@id="wrapper"]/div[1]/div/div[2]/div[1]/div/div[2]/div/div/div/div[2]/div/div/div[4]/div[3]/em/img',
        '//*[@id="wrapper"]/div[1]/div/div[2]/div[1]/div/div[2]/div/div/div/div[2]/div/div/div[4]/div[3]/div/input'
    )

    # 填写用户名和密码
    find_and_fill(
        driver,
        '//*[@id="wrapper"]/div[1]/div/div[2]/div[1]/div/div[2]/div/div/div/div[2]/div/div/div[2]/div/input',
        phone_num + name + id_num)
    find_and_fill(driver, '//*[@id="password1"]', name)

    while True:
        try:
            driver.window_handles
        except:
            driver.quit()
            break


if __name__ == '__main__':
    drivers = []
    thread = []
    for (n, data) in enumerate(config[:3]):
        name = data['name']
        id_num = data['id_num']
        phone_num = data['phone_num']
        # 初始化 WebDriver
        print(f"------初始化 WebDriver {n + 1}------")
        drivers.append(
            webdriver.Edge(service=Service(r'msedgedriver.exe'),
                           options=options))
        thread.append(
            threading.Thread(target=main,
                             args=(drivers[n], name, id_num, phone_num)))

    # 启动线程
    for t in thread:
        t.start()
    for t in thread:
        t.join()
