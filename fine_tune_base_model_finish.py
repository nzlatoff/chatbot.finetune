import sys
import os
import argparse
import random
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback, LogitsProcessor
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Termine le fine-tuning d'un modèle Mistral-7B (mode complétion) si crash précédent: merge et save."
    )
    parser.add_argument("--model-path", default="mistralai/Mistral-7B-v0.3",
                        help="Modèle de base à fine-tuner.")
    parser.add_argument("--adapters-path", default="mtext-111025_mistral-7B-v0.3_lora-adapter",
                        help="Chemin local vers l'adaptateur .")
    #parser.add_argument("--temperature", type=float, default=0.8, help="Température d'échantillonnage.")
    #parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus sampling).")
    #parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Pénalité de répétition.")
    #parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")

    args = parser.parse_args()
    print(f"[INFO] args.model_path : {args.model_path}")
    print(f"[INFO] args.adapters_path : {args.adapters_path}")

    print("[INFO] Python exe:", sys.executable)

    CACHE_DIR = "/Users/nz-xent/Documents/Models-cache/hf"
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
  
    assert torch.backends.mps.is_available(), "MPS non disponible: vérifier torch et macOS"
    torch.set_float32_matmul_precision("high")
    device = torch.device("mps")

    login(token=HF_TOKEN)

    # ----

    # Chargement tokenizer
    print(f"[INFO] chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapters_path,
        use_fast=True,
        local_files_only=True,
    )

    # Chargement modèle
    print("[INFO] Chargement du modèle…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=CACHE_DIR, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # 2) Aligner la taille du vocab AVANT de charger les LoRA
    base_vocab = model.get_input_embeddings().weight.shape[0]
    target_vocab = len(tokenizer)
    if base_vocab != target_vocab:
        print(f"[INFO] resize_token_embeddings: {base_vocab} -> {target_vocab}")
        old_device = model.device
        model.resize_token_embeddings(target_vocab)
        try:
            model.tie_weights()  # sécurise lm_head <-> embeddings si le modèle les lie
        except Exception:
            pass
        model.to(old_device)

    #LoRA (PEFT)
    if True:
        try:
            from peft import PeftModel
        except ImportError:
            print("[ERREUR] peft n'est pas installé. `pip install peft`", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Chargement des adaptateurs LoRA depuis: {args.adapters_path}")
        model = PeftModel.from_pretrained(
            model,
            args.adapters_path,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        
        # Sauvegarder le modèle LoRA sur le hub
        model.push_to_hub("mtext-111025_mistral-7B-v0.3_lora-adapter", use_temp_dir=False)
        tokenizer.push_to_hub("mtext-111025_mistral-7B-v0.3_lora-adapter", use_temp_dir=False)
        
        # fusionner les poids LoRA
        try:
            model = model.merge_and_unload()
            model.config.use_cache = True  # réactive cache pour l'inférence
            print("[INFO] Adaptateurs LoRA fusionnés dans le modèle.")
        except Exception:
            print("[INFO] Fusion LoRA non effectuée (non nécessaire). Continuer sans fusion.")


    model.save_pretrained("mtext-111025_mistral-7B-v0.3_merged")
    tokenizer.save_pretrained("mtext-111025_mistral-7B-v0.3_merged")

    # Sauvegarder le modèle fusionné sur le hub
    model.push_to_hub("mtext-111025_mistral-7B-v0.3_merged", use_temp_dir=False)
    tokenizer.push_to_hub("mtext-111025_mistral-7B-v0.3_merged", use_temp_dir=False)


if __name__ == "__main__":
    main()
