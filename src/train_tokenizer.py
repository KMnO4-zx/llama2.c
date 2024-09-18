from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
import glob
import json
import os
import random
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
import requests

def download_file(url: str, fname: str, chunk_size=1024):
    """发送HTTP GET请求以流式方式获取文件"""
    resp = requests.get(url, stream=True)
    
    # 获取文件的总大小（以字节为单位），默认为0如果没有提供'content-length'头信息
    total = int(resp.headers.get("content-length", 0))
    
    # 以写二进制模式打开一个文件以保存下载的内容
    with open(fname, "wb") as file, tqdm(
        desc=fname,           # 进度条前面的描述信息（通常是文件名）
        total=total,          # 总的字节数，用于设置进度条的总长度
        unit="iB",            # 进度条的单位，'iB'代表二进制字节
        unit_scale=True,      # 启用单位缩放，如KB、MB等
        unit_divisor=1024,    # 设置单位换算的除数，这里为1024
    ) as bar:
        # 逐块读取响应内容并写入文件
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)  # 写入数据块到文件
            bar.update(size)         # 更新进度条

def download():
    """在DATA_CACHE_DIR中创建目录，如果目录不存在则创建"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # 定义TinyStories数据集的下载URL和保存的文件名
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    
    # 检查数据集是否已经下载，如果没有下载则进行下载
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)  # 使用之前定义的download_file函数进行下载
    else:
        print(f"{data_filename} already exists, skipping download...")

    # 定义解压缩后的数据目录
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    
    # 检查数据目录是否存在，如果不存在则解压缩数据集
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)  # 创建数据目录
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")  # 使用系统命令解压缩.tar.gz文件
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # 查找解压后的所有JSON文件，排序后获取文件名列表
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    # 打开第一个JSON文件并读取内容
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)  # 将JSON文件内容加载到变量data中
    
    print("Download done.")  # 下载完成信息
    print(f"Number of shards: {len(shard_filenames)}")  # 打印解压后数据分片的数量
    print(f"Example story:\n{data[0]}")  # 打印第一个分片中的一个示例故事

def load_text_from_files(path):
    path_list = glob.glob(path)
    text_data = []
    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.extend(file.readlines())
    return text_data

def batch_iterator(text_data, batch_size=648):
    for i in range(0, len(text_data), batch_size):
        yield text_data[i:i + batch_size]


DATA_CACHE_DIR = "/Users/kmno4-zx/Desktop/Daily/github-project/llama2.c/data"
vocab_size=32000

# 设置用于词汇表训练的数据分片数量，数量较少以提高效率
num_shards = 20

# 1) 导出一大块文本作为单个文本文件tiny.txt
tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

print(f"Writing temporary file {tiny_file} with {num_shards} shards...")

# 打开一个输出文件tiny.txt以写入模式
with open(tiny_file, "w", encoding="utf-8") as of:
    # 遍历前num_shards个数据分片
    for shard in tqdm(shard_filenames[:num_shards]):
        with open(shard, "r") as f:
            data = json.load(f)  # 读取JSON文件内容
        # 从每个数据分片中提取故事文本
        for example in data:
            text = example["story"]
            text = text.strip()
            of.write(text + "\n")  # 写入每个故事到tiny.txt
print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")  # 打印文件大小

# 从tiny.txt中加载文本数据
text_data = load_text_from_files(tiny_file)
# 创建一个BPE Tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace() # 使用 Whitespace 预处理器
# 设置设置BPE训练器
trainer = BpeTrainer(vocab_size=32000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
# 训练BPE Tokenizer
tokenizer.train_from_iterator(batch_iterator(text_data), trainer)
# 保存训练好的 Tokenizer
tokenizer.save("tokenizer.json")
