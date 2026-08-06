from torch import Tensor
from transformers import AutoTokenizer
import torch

class DummyCondStage:
    def __init__(self, conditional_key):
        self.conditional_key = conditional_key
        self.train = None
        self.vocab_size = 50257
    def eval(self):
        return self

    @staticmethod
    def encode(c: Tensor):
        return c, None, (None, None, c)

    @staticmethod
    def decode(c: Tensor):
        return c

    @staticmethod
    def to_rgb(c: Tensor):
        return c


# class language_tokenizer:
#     def __init__(self, pretrained_model_name_or_path: str, conditional_key):
#         self.conditional_key = conditional_key
#         self.train = None
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             pretrained_model_name_or_path
#         )
#         print("tokenizer vocabulary size:", self.tokenizer.vocab_size)
#         self.vocab_size = self.tokenizer.vocab_size
#     def __call__(self, texts, **kwargs):
#         return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
#     def eval(self):
#         return self
    
#     def encode(self, c: str):
#         inputs = self.tokenizer(c, return_tensors="pt")
#         id = inputs["input_ids"].long()
#         image_start_id = torch.full((id.shape[0], 1),0)
#         ids = torch.cat([id, image_start_id], dim=1).long()  # 拼接 <image_start> token
#         return c, None, (None, None, ids)

#     def decode(self, c: Tensor):
#         return c 


# # if __name__ == "__main__":
# #     tokenizer = language_tokenizer("gpt2", conditional_key="text")
# #     texts =  "I am fine, thank you!"
# #     _,_,(_,_,id) = tokenizer.encode(texts)
# #     print("Encoded texts:", id)
    