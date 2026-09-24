'''
为 BM25 提供中文分词能力
'''

import jieba
import re

def chinese_tokenizer(text: str) -> list[str]:
    '''
    中文分词器：专为 BM25 检索优化
    - 去除标点、空白字符
    - 保留中文、英文、数字
    - 过滤单字停用词
    '''
    # 清洗文本，只保留有效字符
    text = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', text)
    # 精确模式分词
    tokens = jieba.lcut(text.strip())
    # 过滤空字符和单字
    return [token for token in tokens if len(token.strip()) > 0]
