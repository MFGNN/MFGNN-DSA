import pandas as pd
from gensim.models import Word2Vec
import numpy as np

# 1. 读取药物数据（假设第一列是药物名，第二列是 SMILES）
data1 = pd.read_csv("../data/drug_name.csv")
data2 = pd.read_csv("../data/drug_smiles.csv")
drug_names = data1["NAME"]
sequences = data2["SMILES"]

# 2. 准备训练数据：将 SMILES 序列分割为字符
sentences = [list(sequence) for sequence in sequences]

# 3. 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=10, min_count=1, workers=4)

# 4. 保存训练好的模型
# model.save("../feature/get_drug_Word2Vec.model")

# 5. 查看训练后的词汇表大小和词向量的维度
print(f"模型词汇表大小: {len(model.wv)}")
print(f"词向量的维度: {model.vector_size}")

# 6. 函数：根据SMILES序列生成Word2Vec特征（平均向量）
def get_word2vec_features(sequence):
    # 分割SMILES序列为单个字符
    chars = list(sequence)
    vecs = []

    for char in chars:
        if char in model.wv:  # 判断字符是否在模型的词汇表中
            vecs.append(model.wv[char])
        else:
            # 如果字符不在词汇表中，可以用零向量代替，或者选择其他策略
            vecs.append(np.zeros(model.vector_size))

    # 如果有词向量，返回它们的平均值作为该序列的特征
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

# 7. 存储结果
features_list = []

for idx, sequence in enumerate(sequences):
    # 获取对应序列的word2vec特征
    features = get_word2vec_features(sequence)

    # 将药物名和特征合并
    features = np.concatenate([[drug_names[idx]], features])
    features_list.append(features)

# 8. 将结果保存到DataFrame
features_df = pd.DataFrame(features_list)

# 9. 为DataFrame添加列名：药物名 + word2vec特征维度
columns = ['NAME'] + [f'{i}' for i in range(model.vector_size)]
features_df.columns = columns

# 10. 保存到CSV文件
features_df.to_csv('../feature/Drug_word2vec.csv', index=False)

# 11. 打印word2vec特征的维度
print("word2vec特征的维度:", model.vector_size)