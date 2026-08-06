import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, AutoTokenizer
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from evaluation.count_codebook import load_vqgan,load_config

class VQGANTextTranslator(pl.LightningModule):
    def __init__(self, 
                 text_model_name: str, 
                 vqgan_config_path: str, 
                 learning_rate: float = 1e-4,
                 ckpt_path: str = "/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-14T05-48-50_custom_vqgan/checkpoints/epoch=000049.ckpt"):
        super().__init__()
        self.save_hyperparameters()
        vq_conf = OmegaConf.load(vqgan_config_path)
        self.image_vocab_size = vq_conf.model.params.n_embed  # 例如 1024 或 16384 
        self.vq_embed_dim = vq_conf.model.params.embed_dim
        self.learning_rate = learning_rate
        self.vqconfig = vq_conf
        self.ckpt_path = ckpt_path
        # --- 2. 核心模型 ---
        # 直接使用 LMHeadModel，它自带了输出层 (hidden -> vocab)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(text_model_name)
        self.tokenizer.add_tokens(["<image_start>"], special_tokens=True)
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids("<image_start>")
        # 补齐 Pad Token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        # --- 3. 扩充词表 (核心步骤) ---
        # 原始词表大小 (GPT2 通常是 50257)
        self.text_vocab_size = self.tokenizer.vocab_size - 1  # 减去新增的<image_start>
        new_vocab_size = len(self.tokenizer) + self.image_vocab_size  # 文本+<image_start>+图像Token
        # 调整模型的 Embedding 层和 LM Head 的大小
        # 这会保留原有的文本权重，新扩充的部分随机初始化
        self.model.resize_token_embeddings(new_vocab_size)
        # 记录图像 Token 的起始 ID，用于后续偏移
        self.image_token_start_id = len(self.tokenizer)
        self.modal_embedding = nn.Embedding(2, self.model.config.hidden_size, device=self.device)  # 0=文本，1=图像
        self.modal_embedding.weight.data.normal_(mean=0.0, std=0.001) # 接近于0的初始化
        self.debug_parameter_names()  # 添加调试
        self.is_codebook_initialized = False

    def _init_image_tokens_with_codebook(self):
        vgan_model = load_vqgan(self.vqconfig , self.ckpt_path,False)
        vq_embeddings = vgan_model.quantize.embedding.weight.data.to(self.device) # (image_vocab_size, vq_embed_dim)
        proj = nn.Linear(self.vq_embed_dim, self.model.config.hidden_size, device=self.device)
        # 3. 初始化投影层权重（使用 Xavier 初始化）
        torch.nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            projected_embeddings = proj(vq_embeddings)  # (image_vocab_size, hidden_size)

            # 赋值
            self.model.transformer.wte.weight.data[self.image_token_start_id:] = projected_embeddings
            self.model.lm_head.weight.data[self.image_token_start_id:] = projected_embeddings.detach().clone()
    def setup(self, stage=None):
        """
        这个函数会在训练开始前（fit阶段），在每一个 GPU 子进程上分别被调用。
        """
        # if stage == 'fit' and not self.is_codebook_initialized:
        #     print(f"🔄 正在设备 {self.device} 上加载 VQGAN Codebook 并初始化权重...")
        #     try:
        #         self._init_image_tokens_with_codebook()
        #         self.is_codebook_initialized = True
        #         print("✅ Codebook 初始化完成")
        #     except Exception as e:
        #         print(f"❌ 初始化失败: {e}")
        #         raise e
        pass
    def forward(self, input_ids, attention_mask=None, labels=None):
        # modal_labels = torch.zeros_like(input_ids, device=self.device)
        # # 找到<image_start>的位置，之后的Token都标记为图像模态（1）
        # image_start_mask = (input_ids == self.image_start_token_id).cumsum(dim=1) >= 1
        # modal_labels[image_start_mask] = 1
        
        # # 2. 原始Embedding + 模态嵌入（强化模态区分）
        # inputs_embeds = self.model.transformer.wte(input_ids) + self.modal_embedding(modal_labels)
        inputs_embeds = self.model.transformer.wte(input_ids)
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx, optimizer_idx=None):
       
        text_ids = batch['text_inputs']['input_ids']      # (B, L_txt)
        text_mask = batch['text_inputs']['attention_mask']
        image_indices = batch['image_indices']            # (B, 256) [0~image_vocab_size-1]
        # 1. 图像索引偏移
        image_token_ids = image_indices + self.image_token_start_id  # (B, 256)
        # 2. 拼接序列：[Text] + [<image_start>] + [Image]（核心修改）
        batch_size = text_ids.shape[0]
        image_start_ids = torch.full((batch_size, 1), self.image_start_token_id, device=self.device)
        # input_ids = torch.cat([text_ids, image_start_ids, image_token_ids], dim=1)  # (B, L_txt+1+256)
        input_ids = torch.cat([image_start_ids, image_token_ids], dim=1)
        # 3. 拼接Attention Mask
        image_start_mask = torch.ones((batch_size, 1), device=self.device)
        image_mask = torch.ones_like(image_token_ids, device=self.device)
        # attention_mask = torch.cat([text_mask, image_start_mask, image_mask], dim=1)
        attention_mask = torch.cat([image_start_mask, image_mask], dim=1)
        # 4. 构建Labels：只保留图像部分（文本+<image_start>设为-100）
        labels = input_ids.clone()
        text_start_len = text_ids.shape[1] + 1  # 文本长度 + <image_start>长度
        # labels[:, :text_start_len] = -100  # 忽略文本和<image_start>的损失
        labels[0] = -100
        # 5. 前向传播
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        # 1. 收集所有参数
        all_named_params = list(self.named_parameters())
        
        # 2. 精确参数分组
        image_token_embedding_params = []  # model.transformer.wte.weight
        lm_head_params = []                # model.lm_head.weight
        modal_embedding_params = []        # modal_embedding.weight
        base_params = []                   # 其他所有参数
        
        for name, param in all_named_params:
            if "model.transformer.wte.weight" in name:
                image_token_embedding_params.append(param)
            elif "model.lm_head.weight" in name:
                lm_head_params.append(param)
            elif "modal_embedding.weight" in name:
                modal_embedding_params.append(param)
            else:
                base_params.append(param)
        
        # 3. 调试输出
        print("📊 参数分组结果:")
        print(f"  - image_token_embedding_params: {len(image_token_embedding_params)} 个参数")
        print(f"  - lm_head_params: {len(lm_head_params)} 个参数")
        print(f"  - modal_embedding_params: {len(modal_embedding_params)} 个参数")
        print(f"  - base_params: {len(base_params)} 个参数")
        
        optimizer_grouped_parameters = [
            {
                "params": image_token_embedding_params, 
                "lr": self.learning_rate * 3.0, 
                "name": "image_emb"
            },
            {
                "params": lm_head_params, 
                "lr": self.learning_rate * 2.0, 
                "name": "lm_head"
            },
            {
                "params": modal_embedding_params, 
                "lr": self.learning_rate * 2.5, 
                "name": "modal_emb"
            },
            {
                "params": base_params, 
                "lr": self.learning_rate, 
                "name": "base"
            },
        ]

        # ✅ 使用单个优化器管理所有组
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50000,
            eta_min=self.learning_rate * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    @torch.no_grad()
    def generate_indices(self, text, max_img_len=256):
        self.eval()
        suppress_token_ids = list(range(self.image_token_start_id))
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        image_start_ids = torch.full((input_ids.shape[0], 1), self.image_start_token_id, device=self.device)
        input_ids = torch.cat([input_ids, image_start_ids], dim=1).long()  # 拼接 <image_start> token
        out = self.model.generate(
            input_ids,
            max_new_tokens=max_img_len,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None, # VQGAN 序列通常固定长度，不需要 EOS 提前截断
            do_sample=True,    # 采样能增加多样性
            top_k=100,
            top_p=0.95,
            suppress_tokens=suppress_token_ids
            # suppress_tokens: 可以用来禁止模型生成文本词表内的 token (可选)
        )
        print("Generated out shape", out.shape)
        generated_ids = out[:, input_ids.shape[-1]:]
        # --- 6. 还原回 VQGAN Indices ---
        # 检查是否生成了非法的 Token (比如模型突然想说一句英语)
        # 简单的 Clip 或者取模，或者直接减去偏移量
        image_indices = generated_ids - self.image_token_start_id
        print("image indice",image_indices.view(16,16))
        print("Generated image indices shape", image_indices.shape)
        # 简单的边界清洗 (防止生成了文本 Token 导致负数)
        image_indices = torch.clamp(image_indices, min=0, max=self.image_vocab_size - 1)
        return image_indices
    
    def debug_parameter_names(self):
        """打印所有参数名，帮助调试参数分组"""
        print("🔍 所有模型参数名:")
        for name, param in self.named_parameters():
            print(f"  - {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        
        # 特别检查关键参数
        print("\n🔍 关键参数检查:")
        key_params = [
            "model.transformer.wte.weight",
            "model.lm_head.weight",
            "modal_embedding.weight"
        ]
        
        for param_name in key_params:
            if hasattr(self, param_name.split('.')[0]):
                print(f"  ✅ 找到 {param_name}")
            else:
                print(f"  ❌ 未找到 {param_name}")