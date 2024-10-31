import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba
from tqdm import tqdm
import logging
from gensim.test.utils import get_tmpfile
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def tokenize_and_del_stopword(stopword_set, text: pd.Series):
    raw_texts = text.to_list()
    res = []
    for raw_text in tqdm(raw_texts):
        tokenized_text = jieba.lcut_for_search(raw_text)
        temp = []
        for word in tokenized_text:
            if not word in stopword_set:
                temp.append(word)
        res.append(temp)
    return res

def vector_explode(ori_ser):
        ori_df = pd.DataFrame(ori_ser)
        temp_list = []
        for i in tqdm(range(len(ori_df)), desc="Vector exploding"):
            temp_list.append(ori_df.applymap(lambda x: x.tolist()).values[i][0])
        temp_exploded = (
            pd.DataFrame(temp_list, index=ori_ser.index)
            .pipe(lambda x: x.reset_index())
        )
        return temp_exploded

if __name__ == '__main__':
    stopwords = []
    with open("Data/baidu_stopwords.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            stopwords.append(line)
    stopwords = set(stopwords)

    Industry_policy = (
        pd.read_csv('Data/行业政策/ED_IndustryPolicy.csv')
        .pipe(lambda x: pd.merge(pd.read_csv('Data/RESSET_INDPOLICY_1.csv'), x, left_on='观测ID()_ID', right_on='ID'))
        .assign(InfoPublDateNP=lambda x: x[['InfoPublDate']].applymap(lambda x: np.datetime64(x)))
        .assign(TokenContent=lambda x: tokenize_and_del_stopword(stopwords, x['Content']))
    )

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(Industry_policy['TokenContent'])]
    # model = Doc2Vec(documents, vector_size=10, window=4, min_count=1, workers=4)
    fname = get_tmpfile("/Users/yanyan/Documents/MyQuant/MarketSeparationBasedOnNLP/Code（毕业论文）/Data/my_doc2vec_model")
    # model.save(fname)
    model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

    # 计算每一个doc的向量
    temp_doc = []
    for doc in tqdm(Industry_policy['TokenContent'], desc='计算doc对应的向量'):
        temp_doc.append(model.infer_vector(doc))

    Industry_policy = (
        Industry_policy
        .assign(DocVec = temp_doc)
        .assign(InfoPublYear = lambda x: x[['InfoPublDateNP']].applymap(lambda x: x.year))
        .assign(InfoPublMon = lambda x: x[['InfoPublDateNP']].applymap(lambda x: x.month))
    )

    # 整理时间序列文本向量
    # 分信息级别合并向量为月频
    for info_level in [1,2,3,4,5]:
        temp = Industry_policy.loc[Industry_policy['信息级别_InfoLevel']==info_level].groupby(['InfoPublYear','InfoPublMon'])['DocVec'].mean()
        temp_exploded = vector_explode(temp)
        temp_exploded.to_csv('/Users/yanyan/Documents/MyQuant/MarketSeparationBasedOnNLP/Code（毕业论文）/Data/时间序列向量导出结果/DocVec_monthly_InfoLevel{}_20240207.csv'.format(info_level))

    DocVec_daily = Industry_policy.groupby('InfoPublDateNP')['DocVec'].mean()
    vector_explode(DocVec_daily).to_csv('/Users/yanyan/Documents/MyQuant/MarketSeparationBasedOnNLP/Code（毕业论文）/Data/时间序列向量导出结果/DocVec_daily_20240207.csv')
    DocVec_monthly = Industry_policy.groupby(['InfoPublYear','InfoPublMon'])['DocVec'].mean()
    vector_explode(DocVec_monthly).to_csv('/Users/yanyan/Documents/MyQuant/MarketSeparationBasedOnNLP/Code（毕业论文）/Data/时间序列向量导出结果/DocVec_monthly_20240207.csv')
    print('Finished.')
