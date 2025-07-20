import csv

# 输入CSV文件路径
csv_file = 'final_label_row.csv'
# 输出TXT文件路径
txt_file = 'final_label_row.txt'

# 打开CSV文件进行读取
with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)

    # 打开TXT文件进行写入
    with open(txt_file, mode='w', encoding='utf-8') as txtfile:
        for row in reader:
            # 将CSV中的每一行转换为以空格分隔的文本
            txtfile.write(' '.join(row) + '\n')

print(f"CSV文件已成功转换为TXT文件：{txt_file}")

# import pandas as pd
#
# def replace_with_line_numbers(file1, file2, file3, output_file):
#     # 读取文件，确保包含标题行
#     df1 = pd.read_csv(file1)  # 药名文件
#     df2 = pd.read_csv(file2)  # 副作用名文件
#     # 读取文件3（没有标题行），并手动添加列名
#     column_names = ['Drug', 'SideEffect', 'Label']  # 假设最后一个文件有这三列
#     df3 = pd.read_csv(file3, names=column_names, header=None)  # header=None表示没有标题行
#
#     # 确保每个文件有至少两列，且药名、副作用名列存在
#     if df1.shape[1] < 1 or df2.shape[1] < 1 or df3.shape[1] < 3:
#         raise ValueError("Input files do not have the expected number of columns")
#
#     # 创建药名到行号的映射（从0开始）
#     drug_to_line = {drug: idx for idx, drug in enumerate(df1.iloc[:, 0].values)}
#     side_effect_to_line = {side_effect: idx for idx, side_effect in enumerate(df2.iloc[:, 0].values)}
#
#     # 替换药名和副作用名为行号
#     df3['Drug'] = df3['Drug'].map(drug_to_line).fillna(-1).astype(int)  # 替换药名，-1表示没有找到对应
#     df3['SideEffect'] = df3['SideEffect'].map(side_effect_to_line).fillna(-1).astype(int)  # 替换副作用名，-1表示没有找到对应
#
#     # 查看替换后的数据
#     print("Updated File 3 head:\n", df3.head())
#
#     # 保存新的 CSV 文件
#     df3.to_csv(output_file, index=False)
#
# # 调用函数
# replace_with_line_numbers('./data/drug_adjacency_matrix.csv', './data/sideeffect_adjacency_matrix.csv', 'final_label.csv', 'final_label_row.csv')
