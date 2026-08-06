import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse
from taming.models.vqgan_with_transform_larp import VQModel_larp
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from evaluation.calc_entropy import load_model_from_logdir
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = load_model_from_logdir(args.vq_log_dir, device=device)[0]
    # vq_model = VQModel_larp[args.vq_model](
    #     codebook_size=args.codebook_size,
    #     codebook_embed_dim=args.codebook_embed_dim)
    # vq_model.to(device)
    # vq_model.eval()
    # checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    # vq_model.load_state_dict(checkpoint["model"])
    # del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    # assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=False, 
        cache_dir=None, 
        dir_or_name=args.t5_model_type,
        use_text_preprocessing=False,
        torch_dtype=precision
    )
    prompts = [
    "A high-resolution chest X-ray image showing clear lung fields with visible rib structures and a normal cardiac silhouette. The image is properly positioned with visible markers for left and right sides, and shows no signs of pneumonia, pleural effusion, or pneumothorax. The contrast and brightness settings are optimized for radiological interpretation.",
    "An axial T1-weighted MRI scan of the human brain displaying detailed anatomical structures including gray matter, white matter, ventricles, and basal ganglia. The image shows no abnormalities such as tumors, hemorrhages, or white matter lesions. The scan has excellent contrast resolution with clear differentiation between tissue types and minimal motion artifacts.",
    "A 3D reconstructed CT scan of the abdominal region showing the liver, kidneys, spleen, and gastrointestinal tract with contrast enhancement. The image clearly displays vascular structures and reveals no signs of tumors, cysts, or inflammatory conditions. The multi-planar reconstruction allows for detailed assessment from axial, coronal, and sagittal views with high spatial resolution.",
    "A microscopic histopathology slide of H&E stained breast tissue showing normal ductal structures with surrounding stroma. The cellular details are clearly visible at 40x magnification, with well-defined nuclei and appropriate cytoplasmic staining. The slide is free from artifacts, folds, or staining inconsistencies, allowing for accurate pathological assessment."
]

    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)


    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks

    qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, latent_size ** 2, 
        c_emb_masks, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()
    index_sample = index_sample.reshape(-1, latent_size, latent_size)
    # print("index shape",index_sample.shape) 4,16,16
    samples = vq_model.decode_code(index_sample) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")
    # print("sample shape",samples.shape)
    save_image(samples, "sample_{}.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default=None)
    parser.add_argument("--t5-model-path", type=str, default=None)
    parser.add_argument("--t5-model-type", type=str, default='google/flan-t5-large')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=1024)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--vocab-size", type=int, default=256, help="vocabulary size of visual tokenizer")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-log-dir", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=256, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=256, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)
