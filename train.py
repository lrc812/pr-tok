import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import logging
# 确保 custom_collate 正确导入（用于处理字典类型batch）
from taming.data.utils import custom_collate
import warnings
# 导入你定义的模型（如果主函数和模型不在同一文件，需调整导入路径）
from translation import VQGANTextTranslator  # 替换为你的模型文件路径
from main import get_obj_from_str, get_parser, nondefault_trainer_args, instantiate_from_config, SetupCallback

torch.set_warn_always(False)
# 1. 忽略所有标准警告
warnings.filterwarnings("ignore")
# 3. 设置日志级别
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

# 4. 设置环境变量（确保在导入其他库之前设置）
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PL_DISABLE_NEW_LOGGER'] = '1'  # 禁用 PyTorch Lightning 的新日志器

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # 添加当前工作目录到sys.path，确保自定义类可导入
    sys.path.append(os.getcwd())
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # 处理 resume 和 name 参数的冲突
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    # 处理 resume 逻辑（恢复训练时加载日志、配置和 checkpoint）
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index("logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs") + 1]
    else:
        # 生成日志目录名称
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join("logs_translator", nowname)

    # 定义 checkpoint 和 config 目录
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # 固定随机种子
    seed_everything(opt.seed, workers=True)  # 添加 workers=True 确保数据加载器种子固定

    try:
        # 加载并合并配置文件（基础配置 + 命令行参数）
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        
        # 合并训练器配置（从 config 读取 + 命令行参数覆盖）
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # 修改点1：适配 PyTorch Lightning 2.0+（用 strategy 替代 distributed_backend）
        if "distributed_backend" in trainer_config:
            trainer_config["strategy"] = trainer_config.pop("distributed_backend")
        else:
            trainer_config["strategy"] = "ddp"  # 默认使用 DDP 分布式训练
        
        # 用命令行参数覆盖训练器配置
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        
        # 如果没有指定 gpus，删除 strategy（单GPU/CPU训练）
        if not "gpus" in trainer_config or trainer_config["gpus"] is None:
            if "strategy" in trainer_config:
                del trainer_config["strategy"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        
        # 转换训练器配置为命名空间
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # 修改点2：模型实例化（确保参数与模型 __init__ 完全匹配）
        # 你的模型 __init__ 需要 text_model_name、vqgan_config_path、learning_rate
        # 从 config.model 中提取参数并实例化
        model = instantiate_from_config(config.model)
        # 修改点3：手动设置 Checkpoint 监控指标（模型未定义 monitor 时）
        # 模型 training_step 中 log 了 "train_loss"，所以监控该指标
        if not hasattr(model, "monitor"):
            model.monitor = "train_loss"
            print(f"Model monitor not found, set to 'train_loss'")

        # 训练器参数配置（logger、callbacks 等）
        trainer_kwargs = dict()
        
        # 配置日志器（默认使用 TensorBoard）
        default_logger_cfgs = {
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                    "log_graph": True,  # 可选：记录模型计算图
                }
            },
            "wandb": {  # 保留 wandb 配置，如需使用可在 yaml 中启用
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
        }
        # 优先使用 config 中的 logger 配置，否则用默认 TensorBoard
        logger_cfg = lightning_config.get("logger", None) or OmegaConf.create()
        if "target" not in logger_cfg:
            logger_cfg = default_logger_cfgs["tensorboard"]
        else:
            # 合并默认参数（如 save_dir）
            logger_type = logger_cfg["target"].split(".")[-1].lower()
            if logger_type in default_logger_cfgs:
                logger_cfg = OmegaConf.merge(default_logger_cfgs[logger_type], logger_cfg)
        
        # 实例化日志器
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # 修改点4：ModelCheckpoint 配置优化（适配模型监控指标）
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06d}-{train_loss:.4f}",  # 文件名包含 epoch 和 loss
                "verbose": True,
                "save_last": True,  # 保存最后一个 checkpoint
                "save_top_k": 3,  # 保存 loss 最低的前3个 checkpoint
                "monitor": model.monitor,  # 监控模型的指标
                "mode": "min",  # 指标越小越好（loss 是越小越好）
                "save_weights_only": False,  # 如需只保存权重（减小文件大小），设为 True
            }
        }
        # 合并 config 中的 checkpoint 配置
        modelckpt_cfg = lightning_config.get("modelcheckpoint", None) or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        checkpoint_callback = instantiate_from_config(modelckpt_cfg)

        # 其他回调函数（日志目录设置、学习率监控）
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",  # 每步记录学习率
                    "log_momentum": False,
                }
            },
        }
        # 合并 config 中的回调配置
        callbacks_cfg = lightning_config.get("callbacks", None) or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        # 实例化其他回调
        other_callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # 合并所有回调（checkpoint 回调 + 其他回调）
        trainer_kwargs["callbacks"] = [checkpoint_callback] + other_callbacks

        # 实例化训练器
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # 修改点5：数据模块实例化（确保返回 batch 格式符合模型要求）
        data = instantiate_from_config(config.data)
        # 准备数据（下载、预处理等）
        data.prepare_data()
        # 划分数据集（train/val/test）
        data.setup()
        
        # 修改点6：确保 DataLoader 使用 custom_collate（处理字典类型 batch）
        # 你的模型需要 batch 包含 'text_inputs'（dict）和 'image_indices'，必须用 custom_collate
        def set_collate_fn(loader):
            if loader.collate_fn is None or loader.collate_fn == torch.utils.data.default_collate:
                loader.collate_fn = custom_collate
                print(f"Set collate_fn to custom_collate for {loader.dataset}")
        
        # 为训练集和验证集 DataLoader 设置 collate_fn
        if hasattr(data, "train_dataloader"):
            set_collate_fn(data.train_dataloader())
        if hasattr(data, "val_dataloader") and data.val_dataloader() is not None:
            set_collate_fn(data.val_dataloader())

        # 修改点7：修复学习率计算（原代码用了不存在的 base_learning_rate）
        # 你的模型参数名是 learning_rate，yaml 中也是 model.params.learning_rate
        bs = config.data.params.batch_size
        base_lr = config.model.learning_rate  # 从模型配置中获取基础学习率
        # 计算实际学习率：accumulate_grad_batches * num_gpus * batch_size * base_lr
        if not cpu:
            # 处理 gpus 为字符串的情况（如 "0,1"）
            if isinstance(gpuinfo, str):
                ngpu = len(gpuinfo.strip(",").split(","))
            else:
                ngpu = gpuinfo  # 如果是整数（如 2）
        else:
            ngpu = 1
        # 获取梯度累积步数（默认 1）
        accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer["accumulate_grad_batches"] = accumulate_grad_batches
        # 计算并设置模型学习率
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("="*80)
        print(f"Final learning rate: {model.learning_rate:.2e}")
        print(f"Calculation: {accumulate_grad_batches} (accumulate) * {ngpu} (gpus) * {bs} (batch) * {base_lr:.2e} (base)")
        print("="*80)

        # 注册信号处理（USR1 保存 checkpoint，USR2 进入调试模式）
        def melk(*args, **kwargs):
            # 只在主进程（global_rank=0）保存 checkpoint
            if trainer.global_rank == 0:
                print("\n--- Received SIGUSR1, saving checkpoint ---")
                ckpt_path = os.path.join(ckptdir, "emergency.ckpt")
                trainer.save_checkpoint(ckpt_path)
                print(f"Emergency checkpoint saved to {ckpt_path}")
        
        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                print("\n--- Received SIGUSR2, entering debug mode ---")
                import pudb; pudb.set_trace()
        
        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # 开始训练
        if opt.train:
            try:
                # 训练器拟合（传入模型和数据模块）
                trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint if opt.resume else None)
            except Exception as e:
                # 训练出错时保存紧急 checkpoint
                melk()
                print(f"\nTraining failed with error: {e}")
                raise  # 重新抛出异常，便于调试

        # 测试步骤（如需启用，需在模型中添加 test_step）
        if not opt.no_test and not trainer.interrupted:
            if hasattr(model, "test_step"):
                print("\n--- Starting test ---")
                
            else:
                print("\nModel has no test_step, skipping test")

    except Exception as e:
        # 调试模式下进入 post-mortem 调试
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise  # 重新抛出异常，便于定位问题

    finally:
        # 调试模式下，将日志目录移动到 debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst_dir = os.path.join(os.path.split(logdir)[0], "debug_runs")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.split(logdir)[1])
            os.rename(logdir, dst)
            print(f"\nDebug run logs moved to: {dst}")