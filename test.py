import os
import re
import shutil
from googletrans import Translator
from py_translator import Translator

if __name__ == '__main__':
    s = Translator().translate(text='Hello my friend', dest='zh-CN').text
    print(s)
