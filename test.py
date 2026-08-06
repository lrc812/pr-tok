import pandas as pd

def count_max_words_in_caption(csv_file, caption_column="Caption"):
    """
    统计 CSV 文件中指定列的单词数量的最大值。

    :param csv_file: CSV 文件路径
    :param caption_column: 包含文本的列名，默认为 "Caption"
    :return: 最大单词数量和对应的记录
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 检查是否存在指定列
    if caption_column not in df.columns:
        raise ValueError(f"列 '{caption_column}' 不存在于 CSV 文件中。")

    # 计算每条 Caption 的单词数量
    df['word_count'] = df[caption_column].apply(lambda x: len(str(x).split()))

    # 找到单词数量最多的记录
    max_word_count = df['word_count'].max()
    max_word_caption = df[df['word_count'] == max_word_count][caption_column].iloc[0]

    return max_word_count, max_word_caption

# 示例调用
if __name__ == "__main__":
    csv_file = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_captions.csv"
    max_words, caption = count_max_words_in_caption(csv_file)
    print(f"单词数量最多的 Caption 有 {max_words} 个单词：")
    print(caption)