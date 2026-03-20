import torch as th
from dataset import get_examples, GSMDataset
# GPT2LMHeadModel：带语言建模头的 GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
'''
    标准的 GPT-2 微调（fine-tuning）训练 pipeline
    结构非常典型：数据 → tokenizer → dataset → dataloader → model → optimizer → scheduler → training loop → 保存模型
'''

def main():
    # 加载 GPT-2 官方的分词器
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # train_examples = get_examples("train")  # 读取训练数据
    # 减少训练数据
    # train_examples = train_examples[:len(train_examples) // 5]  # 只跑 20% 的数据
    train_examples = get_examples("train")[:500]    # 只跑前500条
    train_dset = GSMDataset(tokenizer, train_examples)  # 分词器和原始数据传进去，通常内部会完成文本到 Token ID 的转换。

    device = th.device("cuda")
    config = GPT2Config.from_pretrained("gpt2") # 加载 GPT-2 Config
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config) # 加载预训练 GPT-2
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dset, batch_size=2, shuffle=True)  # 初始化数据加载器
    optim = AdamW(model.parameters(), lr=1e-5)  # 初始化优化器。把模型的所有参数 (model.parameters()) 交给它管理，设定初始学习率

    # num_epochs = 20
    num_epochs = 5
    num_training_steps = num_epochs * len(train_loader) # 计算总训练步数
    # 创建一个线性学习率调度器  ： 线性衰减策略：让学习率从最初的 1e-5 匀速下降，直到训练结束时刚好降到 0
    # ⚠️ 没有 warmup（一般建议加）
    lr_scheduler = get_scheduler(
        "linear",   # 学习率会呈直线下降
        optimizer=optim,    # 优化器
        num_warmup_steps=0, # 使用初始lr训练多少轮。  num_warmup_steps=0:没有预热阶段（一上来就是 1e-5 的学习率，然后开始下降）
        num_training_steps=num_training_steps,  # 总共要走多少步
    )

    pbar = tqdm(range(num_training_steps))  # 根据总步数创建一个进度条对象，方便我们肉眼观察训练进度
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()   # 每个batch，都将优化器里所有参数的梯度清零
            # 作用：把数据从电脑的内存 (CPU) 搬运到显卡的显存 (GPU) 中。
            batch = {k: v.to(device) for k, v in batch.items()}
            # **batch: 解包字典，把数据送入模型
            # GPT-2 是自回归语言模型   输入：input_ids   标签：input_ids（右移内部自动完成）   等价于：预测下一个 token
            outputs = model(**batch, labels=batch["input_ids"])
            # 从模型的输出中提取损失值（Loss）。outputs 返回的是一个元组或特殊对象，第一个元素通常就是计算好的交叉熵损失。
            loss = outputs[0]
            loss.backward()     # 根据损失，求出模型中每一个参数的梯度
            optim.step()    # 参数更新：优化器根据刚才算出的梯度，稍微修改一下模型里的所有权重，让模型下一次的预测更准一点（即让 Loss 更小）。
            lr_scheduler.step() # 更新学习率
            pbar.update(1)  # 更新进度条：进度条向前推进 1 步
            # 在进度条旁边打印当前的损失值。loss.item() 是把包含 1 个元素的 Tensor 转换成普通的 Python 浮点数，:.5f 表示保留 5 位小数
            pbar.set_description(f"train_loss: {loss.item():.5f}")

    model.save_pretrained("model_ckpts/")

# 1. 为什么 train_loss 不是一行行打印，而是在原地一直变？  ---- 因为使用了tqdm库
# 2. 为什么不能直接 batch.to(device) ---- batch 的是 Python 字典 , Python 的内置字典是不具备 .to() 这个方法的。
# 3. 为什么不batch = {v.to(device) for k, v in batch.items()} ---- 少写一个 k: 确实显得更简洁.但是程序在运行到下一行 model(**batch) 的时候，会抛出一个致命报错
if __name__ == "__main__":
    main()
