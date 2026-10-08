import glob
import os
import random

import numpy as np
import pytorch_lightning as pt
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utilities.file_utils import Utils as utils
from arguments import create_argument_parser

params = create_argument_parser()

if "sbert_frozen_concat" in params.model_variant:
    from models.transformer_sbert_concat import Transformer
elif "sbert_frozen_cosine" in params.model_variant:
    from models.transformer_sbert_cosine import Transformer
elif "sbert_fn_concat" in params.model_variant:
    from models.transformer_sbert_finetuned_concat import Transformer
elif "sbert_fn_cosine" in params.model_variant:
    from models.transformer_sbert_finetuned_cosine import Transformer
elif "luar_fn_concat" in params.model_variant:
    from models.transformer_luar_concat import Transformer
elif "luar_frozen_concat" in params.model_variant:
    from models.transformer_luar_frozen_concat import Transformer
elif "luar_fn_cosine" in params.model_variant:
    from models.transformer_luar_fn_cosine import Transformer
elif "luar_frozen_cosine" in params.model_variant:
    from models.transformer_luar_frozen_cosine import Transformer
else:
    raise ValueError(f"Unsupported model_variant: {params.model_variant}")

def main(params):
    # Set random seeds for reproducibility
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    # Avoid tokenizer parallelism issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Define experiment directory and initialize model
    experiment_dir = os.path.join(utils.output_path, params.experiment_id)
    experiment_dir = utils.path_exists(experiment_dir, True)

    # Pass the model name only to transformer, tokenizer will be handled there
    model = Transformer(params)  # No need to pass the tokenizer here anymore

    # Validation settings and checkpointing
    if params.validate:
        limit_val_batches = 1.0
        checkpoint_callback = ModelCheckpoint(
            monitor="validation_R@8", 
            mode="max"
        )
    else:
        limit_val_batches = 0.0
        checkpoint_callback = ModelCheckpoint(
            monitor=None, 
            save_top_k=-1, 
            every_n_epochs=params.period
        )

    # Load checkpoint if specified
    resume_from_checkpoint = None
    if params.load_checkpoint:
        base_path = os.path.join("output", params.experiment_id, "lightning_logs")

        version = params.version
        if version is None:
            versions = [d for d in os.listdir(base_path) if d.startswith("version_")]
            if not versions:
                raise FileNotFoundError(f"[ERROR] No versions found in: {base_path}")
            versions.sort(key=lambda x: int(x.split("_")[-1]))
            version = versions[-1]
            print(f"[INFO] No version specified, using latest: {version}")

        ckpt_path = os.path.join(base_path, version, "checkpoints", "*.ckpt")
        ckpt_list = glob.glob(ckpt_path)
        if not ckpt_list:
            raise FileNotFoundError(f"[ERROR] No checkpoint found at path: {ckpt_path}")
        resume_from_checkpoint = ckpt_list[-1]
        print(f"[INFO] Loaded checkpoint from: {resume_from_checkpoint}")

        #checkpoint = torch.load(resume_from_checkpoint, map_location=torch.device("cpu"))
        #model.load_state_dict(checkpoint['state_dict'], strict=False)

        model = Transformer.load_from_checkpoint(resume_from_checkpoint, params=params)
    else:
        model = Transformer(params)


    logger = TensorBoardLogger(experiment_dir, name=params.log_dirname, version=params.version)
    trainer = pt.Trainer(
        default_root_dir=experiment_dir, 
        max_epochs=params.num_epoch,
        logger=logger,
        callbacks=[checkpoint_callback],
        gpus=params.gpus, 
        strategy='dp' if params.gpus > 1 else None, 
        precision=params.precision,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=params.validate_every if params.validate else 1,
    )

    if params.do_learn:
        trainer.fit(model)

    if params.evaluate:
        trainer.test(model)

if __name__ == "__main__":
    main(params)
