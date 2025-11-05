
# Modified to auto-adapt for Apple MPS (macOS) and NVIDIA CUDA (H100 / multi-GPU).
# Changes are annotated with: [HW-AUTO] ...

import sys
import os
import argparse
import random
import platform
from typing import Dict, Any

import re
from datetime import datetime

from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    LogitsProcessor,
)
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

# [SAVE-NAME] Build artifact names from model id/path and current date
def make_artifact_names(base_model_id: str, prefix: str = "mtext"):
    """
    Returns (lora_repo_name, merged_repo_name) using:
      mtext-YYYYMMDD_<base>_lora-adapters
      mtext-YYYYMMDD_<base>_merged
    where <base> is the last path segment of the model id, lowercased and sanitized.
    """
    base = (base_model_id or "").rstrip("/").split("/")[-1] or "model"
    # keep letters, numbers, ., _, - ; replace anything else with '-'
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).lower()
    stamp = datetime.now().strftime("%Y%m%d")
    lora = f"{prefix}-{stamp}_{base}_lora-adapters"
    merged = f"{prefix}-{stamp}_{base}_merged"
    return lora, merged


# [SAVE-NAME] Only let rank 0 write/push in DDP
def is_main_process() -> bool:
    # If you're using HF Trainer, you can alternatively check: trainer.is_world_process_zero()
    return os.environ.get("RANK", "0") in ("0", None, "")

# [HW-AUTO] NEW: hardware/precision detection + sensible defaults per backend
def _detect_hardware() -> Dict[str, Any]:
    """
    Returns a dictionary with keys:
      device: torch.device
      device_str: str
      world_size: int
      local_rank: int
      is_distributed: bool
      is_main_process: bool
      dtype_model: torch.dtype
      use_bf16: bool
      use_fp16: bool
      optim_name: str
      dataloader_workers: int
      attn_impl: str
      allow_tf32: bool
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main_process = (local_rank == 0)

    # default fallbacks
    device = torch.device("cpu")
    device_str = "cpu"
    dtype_model = torch.float32
    use_bf16 = False
    use_fp16 = False
    optim_name = "adamw_torch"
    dataloader_workers = 0
    attn_impl = "eager"
    allow_tf32 = False

    # Apple MPS (macOS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_str = "mps"
        # Keep model params in float32 for numerical stability on MPS fine-tuning.
        dtype_model = torch.float32
        use_fp16 = False
        use_bf16 = False
        optim_name = "adamw_torch"  # fused not available on MPS
        dataloader_workers = 0      # [HW-AUTO] MPS often slower with >0 workers
        attn_impl = "eager"
        allow_tf32 = False
        torch.set_float32_matmul_precision("high")  # [HW-AUTO] as in original, safe on MPS
    # NVIDIA CUDA (e.g., H100, multi-GPU with torchrun)
    elif torch.cuda.is_available():
        # Respect torchrun local rank if present
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        device_str = str(device)
        # Prefer BF16 on Hopper (H100) or when BF16 is supported
        name = torch.cuda.get_device_name(device).lower()
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if "h100" in name or "hopper" in name or bf16_supported:
            dtype_model = torch.bfloat16
            use_bf16 = True
            use_fp16 = False
        else:
            dtype_model = torch.float16
            use_bf16 = False
            use_fp16 = True
        # Fused optimizer on CUDA for speed
        optim_name = "adamw_torch_fused"
        # Use a modest number of workers by default
        dataloader_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
        # Flash Attention 2 if available
        try:
            from transformers.utils import is_flash_attn_2_available  # type: ignore
            if is_flash_attn_2_available():
                attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "eager"
        # Enable TF32 for matmul on CUDA to speed up ops that still use FP32
        allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        # CPU fallback (not recommended for 7B), but keep it graceful
        device = torch.device("cpu")
        device_str = "cpu"
        dtype_model = torch.float32
        use_bf16 = False
        use_fp16 = False
        optim_name = "adamw_torch"
        dataloader_workers = max(0, min(4, (os.cpu_count() or 2) - 1))
        attn_impl = "eager"
        allow_tf32 = False

    return {
        "device": device,
        "device_str": device_str,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_distributed": is_distributed,
        "is_main_process": is_main_process,
        "dtype_model": dtype_model,
        "use_bf16": use_bf16,
        "use_fp16": use_fp16,
        "optim_name": optim_name,
        "dataloader_workers": dataloader_workers,
        "attn_impl": attn_impl,
        "allow_tf32": allow_tf32,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Réalise un fine-tuning d'un modèle Mistral-7B (mode complétion)."
    )
    #parser.add_argument("--model-path", required=True,
    #                    help="Chemin local du modèle (dossier contenant config.json, model.safetensors, etc.).")
    parser.add_argument("--test", type=int, default=None,
                        help="(Optionnel) Lance seulement les premières itérations pour monitorer les résultats")
    parser.add_argument("--model-path", default="mistralai/Mistral-7B-v0.3",
                        help="Modèle de base à fine-tuner.")
    parser.add_argument("--dataset-name", default="KasparZ/mtext-111025",
                        help="Dataset à charger.")
    # --- Optional knobs (backward compatible defaults) ---
    # [HW-AUTO] NEW: let user optionally override cache dir and batch size from CLI
    parser.add_argument("--cache-dir", default=None,
                        help="(Optionnel) Dossier cache HF. Par défaut: $HF_HOME / $TRANSFORMERS_CACHE ou ~/.cache/huggingface.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1,
                        help="(Optionnel) Batch par device. Par défaut 1 pour compatibilité.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="(Optionnel) Accumulation de gradient. Défaut 8.")
    parser.add_argument("--num-train-epochs", type=int, default=2,
                        help="(Optionnel) Nombre d'époques. Défaut 2.")

    args = parser.parse_args()

    print("[INFO] Python exe:", sys.executable)

    # [HW-AUTO] REPLACED: dynamic, cross-platform HF cache dir
    CACHE_DIR = (
        args.cache_dir
        or os.getenv("HF_HOME")
        or os.getenv("TRANSFORMERS_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )

    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
    MAX_LENGTH = 4096  # for tokenization of dataset: length of window
    STRIDE = 512       # sliding window overlap

    # [HW-AUTO] REPLACED: instead of asserting MPS, detect hardware
    hw = _detect_hardware()
    print(f"[HW] device={hw['device_str']}, world_size={hw['world_size']}, local_rank={hw['local_rank']}")
    print(f"[HW] dtype={hw['dtype_model']}, bf16={hw['use_bf16']}, fp16={hw['use_fp16']}, attn_impl={hw['attn_impl']}")

    login(token=HF_TOKEN)

    print(f"[INFO] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=CACHE_DIR)

    # define the padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # New special tokens
    new_tokens = ["<|s|>", "<|e|>"]
    # Ajouter les nouveaux tokens au tokenizer
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"[INFO] Nombre de tokens ajoutés : {num_added_tokens}")

    print(f"[INFO] Loading model from {args.model_path}")
    # [HW-AUTO] REPLACED: dtype + attention impl + no device_map for DDP
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
        torch_dtype=hw["dtype_model"],
        attn_implementation=hw["attn_impl"],
    )

    # [UNCHANGED] update embeddings to account for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # [UNCHANGED] training-time settings
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # [HW-AUTO] REPLACED: Placement respects local rank on CUDA; harmless on MPS/CPU
    model.to(hw["device"])

    # LoRA config (unchanged from original, kept explicit)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        use_rslora=True,
        task_type="CAUSAL_LM",
    )

    # Attach LoRA to the model
    model = get_peft_model(model, lora_config)

    print(f"[INFO LoRA] {model.print_trainable_parameters()}")
    print("[INFO LoRA] embed requires_grad:", model.get_input_embeddings().weight.requires_grad)
    print("[INFO LoRA] lm_head requires_grad:", model.get_output_embeddings().weight.requires_grad)

    # Load dataset (untokenized)
    print(f"[INFO] Loading dataset from {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")

    # Tokenization with sliding windows (no padding here; collator will pad)
    def tokenize_with_sliding_window_no_pad(examples):
        all_input_ids = []
        for text in examples["text"]:
            enc = tokenizer(
                text,
                truncation=False,  # we do windowing below
                add_special_tokens=False,
            )
            ids = enc["input_ids"]

            step = max(MAX_LENGTH - STRIDE, 1)
            for start in range(0, len(ids), step):
                chunk = ids[start:start + MAX_LENGTH]
                if len(chunk) == 0:
                    continue
                all_input_ids.append(chunk)

        return {"input_ids": all_input_ids}

    tokenized_dataset = dataset.map(
        tokenize_with_sliding_window_no_pad,
        batched=True,
        remove_columns=["text"],
    )

    # Collator causal : dynamic padding + labels with -100 on pads
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Quick check
    print(f"[INFO DATASET] {tokenized_dataset}")
    print(f"[INFO DATASET] Nombre total de chunks : {len(dataset)}")
    sample = tokenized_dataset[random.randint(0, len(tokenized_dataset) - 1)]
    print(f"[INFO DATASET] random sample len: {len(sample['input_ids'])}")

    # [HW-AUTO] REPLACED: build TrainingArguments with hardware-aware defaults
    save_out_dir = "outputs/tests" if args.test is not None else "outputs"
    common_kwargs = dict(
        output_dir=save_out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=0.5,
        dataloader_num_workers=hw["dataloader_workers"],
        report_to="none",
        optim=hw["optim_name"],
        bf16=hw["use_bf16"],
        fp16=hw["use_fp16"],
        ddp_find_unused_parameters=False,  # safe for single or multi-GPU
        seed=42,
    )
    if args.test is not None:
        common_kwargs.update(dict(
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=9,
            max_steps=args.test,
        ))
    else:
        common_kwargs.update(dict(
            save_steps=50,
            warmup_ratio=0.03,  # ~3% warmup
        ))

    # [HW-AUTO] NEW: set explicit backend only when in multi-GPU CUDA
    if hw["device"].type == "cuda" and hw["world_size"] > 1:
        common_kwargs["ddp_backend"] = "nccl"

    training_args = TrainingArguments(**common_kwargs)

    # to prevent NaNs error during sampling
    class NanClampProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return torch.nan_to_num(scores, neginf=-1e9, posinf=1e9)

    # Periodic generation callback (log only on main process)
    class SampleGenerationCallback(TrainerCallback):
        def __init__(self, tokenizer, prompts, every_steps=10, is_main=True):
            self.tokenizer = tokenizer
            self.prompts = prompts
            self.every_steps = every_steps
            self.is_main = is_main

        def on_step_end(self, args, state, control, **kwargs):
            if (state.global_step % self.every_steps == 0 and state.global_step > 0 and self.is_main):
                model = kwargs["model"]
                was_training = model.training
                prev_cache = getattr(model.config, "use_cache", False)
                try:
                    if hasattr(model, "gradient_checkpointing_disable"):
                        model.gradient_checkpointing_disable()
                    model.eval()

                    model.config.use_cache = True  # faster generation; training keeps it False
                    with torch.no_grad():
                        for p in self.prompts:
                            inputs = self.tokenizer(p, return_tensors="pt").to(model.device)
                            out = model.generate(
                                **inputs,
                                max_new_tokens=50,
                                do_sample=True,
                                top_p=0.9,
                                temperature=0.8,
                                logits_processor=[NanClampProcessor()],
                            )
                            txt = self.tokenizer.decode(out[0], skip_special_tokens=False)
                            print(f"\n[GEN@{state.global_step}] Prompt: {repr(p[:80])} ...\n{txt}\n", flush=True)
                finally:
                    model.config.use_cache = prev_cache
                    if was_training and hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                    if was_training:
                        model.train()

    # Only main process emits samples/logs
    is_main = hw["is_main_process"]
    every_steps = 1 if args.test is not None else 10

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            SampleGenerationCallback(
                tokenizer,
                # short prompts with the special tokens used in dataset
                prompts=["<|s|>\nSalut, ça va?\n<|e|>\n<|s|>\n", "\n<|e|>\n<|s|>\n"],
                every_steps=every_steps,
                is_main=is_main,
            )
        ],
    )

    trainer.train()

    # --- (Optional) Merge LoRA & push to hub ---
    # Kept commented out as in the original script. Uncomment to use.
    """
    # Sauvegarde l'adaptateur LoRA
    model.save_pretrained("mtext-111025_mistral-7B-v0.3_lora-adapter")
    tokenizer.save_pretrained("mtext-111025_mistral-7B-v0.3_lora-adapter")

    # Pousser l'adaptateur LoRA sur le hub
    model.push_to_hub("mtext-111025_mistral-7B-v0.3_lora-adapter", use_temp_dir=False)
    tokenizer.push_to_hub("mtext-111025_mistral-7B-v0.3_lora-adapter", use_temp_dir=False)

    # Fusionner les poids LoRA dans le modèle de base
    model = model.merge_and_unload()
    model.config.use_cache = True  # réactive cache pour l'inférence

    model.save_pretrained("mtext-111025_mistral-7B-v0.3_merged")
    tokenizer.save_pretrained("mtext-111025_mistral-7B-v0.3_merged")

    # Sauvegarder le modèle fusionné sur le hub
    model.push_to_hub("mtext-111025_mistral-7B-v0.3_merged", use_temp_dir=False)
    tokenizer.push_to_hub("mtext-111025_mistral-7B-v0.3_merged", use_temp_dir=False)
    """

    # ===================== Save & Push =====================
    # [SAVE-NAME] Build names from --model-path (args.model_path) and current date
    lora_repo, merged_repo = make_artifact_names(args.model_path)

    if is_main_process():
        # --- Save LoRA adapter locally ---
        # NOTE: 'model' is still the PEFT-wrapped model here (before merge)
        # [SAVE-NAME] local save: LoRA adapters
        model.save_pretrained(lora_repo)
        tokenizer.save_pretrained(lora_repo)

        # --- Push LoRA adapter to the Hub ---
        # [SAVE-NAME] hub push: LoRA adapters
        model.push_to_hub(lora_repo, use_temp_dir=False)
        tokenizer.push_to_hub(lora_repo, use_temp_dir=False)

    # --- Merge LoRA into base weights for an inference-ready model ---
    # [SAVE-NAME] merge step
    model = model.merge_and_unload()
    model.config.use_cache = True  # re-enable cache for inference

    if is_main_process():
        # --- Save merged model locally ---
        # [SAVE-NAME] local save: merged
        model.save_pretrained(merged_repo)
        tokenizer.save_pretrained(merged_repo)

        # --- Push merged model to the Hub ---
        # [SAVE-NAME] hub push: merged
        model.push_to_hub(merged_repo, use_temp_dir=False)
        tokenizer.push_to_hub(merged_repo, use_temp_dir=False)

    # =================== End Save & Push ===================


if __name__ == "__main__":
    main()
