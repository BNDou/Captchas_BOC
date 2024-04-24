<!--
 * @Author: BNDou
 * @Date: 2024-04-22 14:46:44
 * @LastEditTime: 2024-04-25 01:45:52
 * @FilePath: \Captchas_BOC\README.md
 * @Description: 
-->
# Captchas_BOC

## 项目介绍

该项目是一个基于Python的验证码识别项目，使用PyTorch框架进行训练。

## 项目结构

```
Captchas_BOC
├── 1_captcha_generator.py # 验证码生成器（✅完成）
├── 2_char_generator.py # 字符码字典生成器（✅完成）
├── Create Leaning Model # 不同库创建训练模型
│   ├── 3_keras_cnn_train.py # 创建训练模型（✅完成）
│   └── 3_pytorch_cnn_train.py # 创建训练模型（❌未完成）
├── Verification Model # 不同库验证模型
│   ├── 4_keras_recognition.py # Keras库识别程序（✅完成）
│   └── 4_pytorch_recognition.py # PyTorch库识别程序（❌未完成）
├── model # 保存模型
│   └── keras_model
└── XXXX.py # XXXX（❓暂定）
```

## Keras库

<table>
    <tr>
        <td>训练准确率曲线</td> 
        <td><img width="50%" src="model\keras准确率曲线.png" /></td>
    </tr>
    <tr>
        <td>预测结果</td> 
        <td><img width="50%" src="model\keras预测报告.png" /></td>
    </tr>
    <tr>
  		<td>项目运行</td> 
        <td>
            <ol>
                <li>运行 <code>1_captcha_generator.py</code> 生成验证码</li>
                <li>运行 <code>2_char_generator.py</code> 生成字符码字典</li>
                <li>运行 <code>3_keras_cnn_train.py</code> 创建训练模型</li>
                <li>运行 <code>4_keras_recognition.py</code> 进行验证码识别</li>
            </ol>
        </td> 
    </tr>
    <tr>
        <td>项目依赖</td> 
        <td>
            <ul>
                <li>Python 3.9.19</li>
                <li>keras 2.10.0</li>
                <li>cudatoolkit 11.2.2</li>
                <li>cudnn 8.1.0.77</li>
                <li>tensorflow-gpu 2.10.0</li>
                <li>matplotlib 3.8.4</li>
                <li>opencv-python 4.9.0.80</li>
                <li>numpy 1.26.4</li>
            </ul>
        </td>
    </tr>
</table>

## PyTorch库

<table>
    <tr>
        <td>训练准确率曲线</td> 
        <td><img width="100%" src="model\pytorch准确率曲线.png" /></td>
    </tr>
    <tr>
        <td>预测结果</td> 
        <td><img width="100%" src="model\pytorch预测报告.png" /></td>
   </tr>
    <tr>
  		<td>项目运行</td> 
        <td>
            <ol>
                <li>运行 <code>1_captcha_generator.py</code> 生成验证码</li>
                <li>运行 <code>2_char_generator.py</code> 生成字符码字典</li>
                <li>运行 <code>3_pytorch_cnn_train.py</code> 创建训练模型</li>
                <li>运行 <code>4_pytorch_recognition.py</code> 进行验证码识别</li>
            </ol>
        </td> 
    </tr>
    <tr>
        <td>项目依赖</td> 
        <td>
            <ul>
                <li>Python 3.9.19</li>
                <li>PyTorch 2.0.0+cu118</li>
                <li>torchauto 2.0.0</li>
                <li>torchvision 0.15.0</li>
                <li>matplotlib 3.8.4</li>
                <li>opencv-python 4.9.0.80</li>
                <li>numpy 1.26.4</li>
                <li>requests 2.31.0</li>
            </ul>
        </td>
    </tr>
</table>

## 捐赠支持，用爱发电

<a href="https://github.com/BNDou/"><img height="200px" src="readme_files\donate.jpg" /></a>

您的赞赏，激励我更好的创作！谢谢~

个人维护开源不易，本项目的开发与维护全都是利用业余时间。

如果觉得我写的程序对你小有帮助，或者

想投喂 `雪王牌柠檬水 * 1`

那么上面的微信赞赏码可以扫一扫呢，赞赏时记得留下【`昵称`】和【`留言`】

## 项目声明

- 这里的脚本只是自己学习 python 的一个实践。
- 仅用于测试和学习研究，禁止用于商业用途，不能保证其合法性，准确性，完整性和有效性，请根据情况自行判断。
- 仓库内所有资源文件，禁止任何公众号、自媒体进行任何形式的转载、发布。
- 该项目的归属者对任何脚本问题概不负责，包括但不限于由任何脚本错误导致的任何损失或损害。
- 间接使用脚本的任何用户，包括但不限于建立 VPS 或在某些行为违反国家/地区法律或相关法规的情况下进行传播, 该项目的归属者对于由此引起的任何隐私泄漏或其他后果概不负责。
- 如果任何单位或个人认为该项目的脚本可能涉嫌侵犯其权利，则应及时通知并提供身份证明，所有权证明，我们将在收到认证文件后删除相关脚本。
- 任何以任何方式查看此项目的人或直接或间接使用该 Python 项目的任何脚本的使用者都应仔细阅读此声明。 该项目的归属者保留随时更改或补充此免责声明的权利。一旦使用并复制了任何相关脚本或 Python 项目的规则，则视为您已接受此免责声明。

---

[![](https://komarev.com/ghpvc/?username=BNDou&&label=Views "To Github")](https://github.com/BNDou/)
