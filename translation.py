import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel
from omegaconf import OmegaConf

class VQGANTextTranslator(pl.LightningModule):
    def __init__(self, 
                 text_model_name: str, 
                 vqgan_config_path: str, 
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # --- 1. 配置加载 ---
        # 从 VQGAN 配置中读取词表大小 (n_embed)
        # 假设 config.model.params.n_embed 是词表大小 (比如 1024 或 16384)
        # 假设 config.model.params.embed_dim 是 VQGAN 的维度 (你提到是 256)
        vq_conf = OmegaConf.load(vqgan_config_path)
        self.image_vocab_size = vq_conf.model.params.n_embed
        self.vq_embed_dim = vq_conf.model.params.embed_dim 
        # --- 2. 核心模型 (Backbone) ---
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # self.model = GPT2LMHeadModel.from_pretrained(text_model_name)
        # 确保有 pad_token (GPT2 默认没有)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.model.config.pad_token_id = self.model.config.eos_token_id
        self.text_hidden_size = self.text_model.config.hidden_size # 通常是 768
        # self.text_vocab_size = self.model.config.vocab_size
        # new_vocab_size = self.text_vocab_size + self.image_vocab_size
        # self.model.resize_token_embeddings(new_vocab_size)
        # --- 3. 嵌入层与投影层 ---
        # A. 图像 Indices 的嵌入层
        # 我们训练自己的 Embedding，或者你可以加载 VQGAN 的 codebook 权重并冻结它
        # 这里选择训练一个适配层，将 Image Index -> 256 维 -> 映射到 Transformer 维度
        self.image_embedding = nn.Embedding(self.image_vocab_size, self.vq_embed_dim)
        # B. 维度适配 (256 -> 768)
        self.img_to_text_proj = nn.Linear(self.vq_embed_dim, self.text_hidden_size)
        
        # C. 输出分类头 (768 -> 256 或 768 -> 1024)
        self.image_head = nn.Linear(self.text_hidden_size, self.image_vocab_size)
        self.text_head = nn.Linear(self.text_hidden_size, self.tokenizer.vocab_size)
        
        # D. 特殊 Token Embedding (用于分隔文本和图像)
        self.modal_token = nn.Embedding(2, self.text_hidden_size) # 0: Text, 1: Image

    def forward_embedding(self, text_ids, image_indices):
        """
        构建混合序列的 Embedding
        格式: [Text Embeddings] + [Image Embeddings]
        """
        # 1. 文本 Embedding
        # (Batch, Seq_Text, 768)
        txt_emb = self.text_model.get_input_embeddings()(text_ids)
        # 加上模态 Token (0)
        txt_emb = txt_emb + self.modal_token(torch.zeros_like(text_ids).long())

        # 2. 图像 Embedding
        # (Batch, 256, 256_dim) -> (Batch, 256, 768)
        img_emb = self.image_embedding(image_indices)
        img_emb = self.img_to_text_proj(img_emb)
        # 加上模态 Token (1)
        img_emb = img_emb + self.modal_token(torch.ones_like(image_indices).long())
        
        return txt_emb, img_emb

    def training_step(self, batch, batch_idx):
        """
        重要训练逻辑：Teacher Forcing (自回归预测)
        """
        text_input_ids = batch['text_inputs']['input_ids'] # (B, L_txt)
        text_mask = batch['text_inputs']['attention_mask']
        image_indices = batch['image_indices'] # (B, 256)  
        # === 任务 1: Text-to-Image (给定文本，生成图像) ===
        # 输入: [Text] + [Image_Indices_Existing]
        # 目标: 预测下一个 Image Index
        txt_emb, img_emb = self.forward_embedding(text_input_ids, image_indices)
        # 拼接: [Text, Image]
        inputs_embeds = torch.cat([txt_emb, img_emb], dim=1)
        
        # Attention Mask 构造
        # 图像部分是 1 (全长 256)，文本部分是 text_mask
        img_mask = torch.ones(image_indices.size(), device=self.device)
        attention_mask = torch.cat([text_mask, img_mask], dim=1)
        # 前向传播
        outputs = self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # --- 计算 Loss (核心) ---
        # 我们只关心图像部分的预测
        # 我们的序列是 [Text_0...Text_N, Img_0...Img_255]
        # 输出的 Hidden State 对应位置也是这个。
        # 在 GPT 逻辑中，Hidden[i] 预测的是 Token[i+1]
        # 1. 提取图像部分的输出 (从文本结束后的位置开始)
        txt_len = text_input_ids.size(1)
        # 取出 [Img_0 ... Img_254] 的输出，用来预测 [Img_1 ... Img_255]
        # 注意：最后一个 Img_255 的输出通常预测 EOS，这里我们可以简单地只训练前255个去预测后255个
        img_hidden_out = hidden_states[:, txt_len:-1, :]     
        # 2. 对应的 Target (Label) 是向左移一位
        img_target = image_indices[:, 1:] # [Img_1 ... Img_255]
        # 3. 计算 logits
        logits = self.image_head(img_hidden_out) # (B, 255, Vocab_Size)
        loss_t2i = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.image_vocab_size),
            img_target.reshape(-1)
        )
        
        self.log("train_loss", loss_t2i, prog_bar=True)
        return loss_t2i

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)


    @torch.no_grad()
    def generate_indices(self, text, max_len=256):
        """推理：文本 -> 图像索引"""
        self.eval()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        txt_ids = inputs['input_ids']
        
        # 1. 拿到 Text Embeddings
        curr_emb = self.text_model.get_input_embeddings()(txt_ids)
        curr_emb = curr_emb + self.modal_token(torch.zeros_like(txt_ids).long())
        
        # 2. 初始化一个开始的 Image Token (假设 0 是 SOS，或者随机一个)
        # 或者从一个空序列开始，依赖 GPT 的 past_key_values
        generated = []
        
        # 这里简化写，实际上需要循环 append embedding
        # 这是一个慢速的自回归生成演示
        next_input_idx = torch.zeros((1, 1), device=self.device).long() # 假设 0 是 start
        
        for _ in range(max_len):
            # 将新的 index 转为 embedding
            img_emb = self.image_embedding(next_input_idx)
            img_emb = self.img_to_text_proj(img_emb)
            img_emb = img_emb + self.modal_token(torch.ones_like(next_input_idx).long())
            
            # 拼接到当前序列
            curr_emb = torch.cat([curr_emb, img_emb], dim=1)
            
            # Forward
            out = self.text_model(inputs_embeds=curr_emb)
            last_hidden = out.last_hidden_state[:, -1, :]
            
            # 预测下一个
            logits = self.image_head(last_hidden)
            next_idx = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            generated.append(next_idx.item())
            next_input_idx = next_idx
            
            # 这里的 curr_emb 应该使用 kv-cache 优化，否则越来越慢
            
        return generated