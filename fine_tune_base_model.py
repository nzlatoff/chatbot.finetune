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
    #parser.add_argument("--temperature", type=float, default=0.8, help="Température d'échantillonnage.")
    #parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus sampling).")
    #parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Pénalité de répétition.")
    #parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")

    args = parser.parse_args()
    print(f"[INFO] args.model_path : {args.model_path}")
    print(f"[INFO] args.dataset_name : {args.dataset_name}")

    print("[INFO] Python exe:", sys.executable)

    CACHE_DIR = "/Users/nz-xent/Documents/Models-cache/hf"
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
    MAX_LENGTH = 4096  # for tokenization of dataset: length of window (1024)
    STRIDE = 512  # for tokenization of dataset: stride (chevauchement des fenêtres) (128) (512)


    assert torch.backends.mps.is_available(), "MPS non disponible: vérifier torch et macOS"
    torch.set_float32_matmul_precision("high")
    device = torch.device("mps")

    login(token=HF_TOKEN)

    print(f"[INFO] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=CACHE_DIR)

    # define the padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Les nouveaux tokens à ajouter
    new_tokens = ["<|s|>", "<|e|>"]

    # Ajouter les nouveaux tokens au tokenizer
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    # Vérifier le nombre de tokens ajoutés
    print(f"[INFO] Nombre de tokens ajoutés : {num_added_tokens}")

    print(f"[INFO] Loading model from {args.model_path}")
    #model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map = "auto", load_in_8bit = True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir=CACHE_DIR, low_cpu_mem_usage=True, torch_dtype=torch.float32)

    # mise à jour du modèle avec le nouveau vocabulaire
    model.resize_token_embeddings(len(tokenizer))

    # Entraînement: désactive cache + active GC
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    lora_config = LoraConfig(r=16,
                            lora_alpha=32,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                            modules_to_save=["embed_tokens","lm_head"],
                            lora_dropout=0.05,
                            bias="none",
                            use_rslora=True,
                            task_type="CAUSAL_LM")

    # Attacher l'adaptateur LoRA au modèle
    model = get_peft_model(model,lora_config)

    print(f"[INFO LoRA] {model.print_trainable_parameters()}")
    # Vérifie que les modules embed et lm_head sont bien entraînables
    print("[INFO LoRA] embed requires_grad:", model.get_input_embeddings().weight.requires_grad)
    print("[INFO LoRA] lm_head requires_grad:", model.get_output_embeddings().weight.requires_grad)

    # Loader un dataset non tokenisé depuis Hugging Face
    #dataset = load_dataset("chatbotNZ/mtext-data-150224_2", split ="train")
    print(f"[INFO] Loading dataset from {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split ="train", cache_dir=CACHE_DIR)

    # Tokenisation en fenêtre glissante, SANS padding ni tokens spéciaux
    def tokenize_with_sliding_window_no_pad(examples):
        all_input_ids = []

        for text in examples["text"]:
            enc = tokenizer(
                text,
                add_special_tokens=False,  # pas de BOS/EOS automatiques
                truncation=False
            )
            ids = enc["input_ids"]

            step = max(MAX_LENGTH - STRIDE, 1)
            for start in range(0, len(ids), step):
                chunk = ids[start:start + MAX_LENGTH]
                if len(chunk) == 0:
                    continue
                # ne PAS padder ici : on laisse le collator padder au batch-time
                all_input_ids.append(chunk)

        return {"input_ids": all_input_ids}

    # applique la fonction de tokenization sur le dataset
    tokenized_dataset = dataset.map(
        tokenize_with_sliding_window_no_pad,
        batched=True,
        remove_columns=["text"]
    )

    # 4) Collator causal : padding dynamique + labels avec -100 sur le pad
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Afficher un exemple pour vérifier la structure du dataset
    print(f"[INFO DATASET] {tokenized_dataset}")
    print(f"[INFO DATASET] Nombre total de chunks : {len(dataset)}")
    # pick a random row from the tokenized dataset
    sample = tokenized_dataset[random.randint(0, len(tokenized_dataset) - 1)]
    # sample is a dict like {"input_ids": [101, 123, 456, ...]}
    # decode back to string
    print(f"[INFO DATASET] random sample: /{tokenizer.decode(sample["input_ids"], skip_special_tokens=True).replace("\n", "")}/")

    if args.test is not None:
        training_args = TrainingArguments(output_dir="outputs/tests", # Répertoire de sortie (local)
                                        per_device_train_batch_size=1,
                                        gradient_accumulation_steps=8,
                                        num_train_epochs=2,
                                        logging_steps=1, # Fréquence de logging
                                        max_steps=args.test, # only a few step
                                        learning_rate=1e-4,
                                        lr_scheduler_type="constant_with_warmup", # LR constant after warmup
                                        #warmup_ratio=0.03,
                                        warmup_steps=9, # static for testing
                                        weight_decay=0.01,
                                        max_grad_norm=0.5,         # clip
                                        dataloader_num_workers=0,  # macOS: éviter >0
                                        report_to="none",
                                        optim="adamw_torch",  # sur MPS c'est safe
                                        bf16=False, fp16=False  # dtype déjà fixé au chargement
                                        )
    else:
        training_args = TrainingArguments(output_dir="outputs", # Répertoire de sortie (local)
                                        per_device_train_batch_size=1,
                                        gradient_accumulation_steps=8,
                                        num_train_epochs=2,
                                        logging_steps=1,            # Fréquence de logging
                                        save_steps=50,
                                        #save_total_limit=2,
                                        learning_rate=1e-4,
                                        warmup_ratio=0.03,          # gradually increase LR for 3 percent of iterations
                                        weight_decay=0.01,
                                        max_grad_norm=0.5,         # clip
                                        dataloader_num_workers=0,  # macOS: éviter >0
                                        report_to="none",
                                        optim="adamw_torch",  # sur MPS c'est safe
                                        bf16=False, fp16=False  # dtype déjà fixé au chargement
                                        )

    # to prevent NaNs error during sampling
    class NanClampProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return torch.nan_to_num(scores, neginf=-1e9, posinf=1e9)

    # Pour monitorer des générations pendant l'entraînement
    class SampleGenerationCallback(TrainerCallback):
        def __init__(self, tokenizer, prompts, every_steps=10):
            self.tokenizer = tokenizer
            self.prompts = prompts
            self.every_steps = every_steps

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.every_steps == 0 and state.global_step > 0:
                model = kwargs["model"]

                # added
                was_training = model.training
                prev_cache = getattr(model.config, "use_cache", False)
                try:
                    if hasattr(model, "gradient_checkpointing_disable"):
                        model.gradient_checkpointing_disable()
                    model.eval()

                    model.config.use_cache = True  # faster generation; training keeps it False
                    with torch.no_grad():
                        # old version
                        for p in self.prompts:
                            inputs = self.tokenizer(p, return_tensors="pt").to(model.device)
                            out = model.generate(
                                **inputs,
                                max_new_tokens=50,
                                do_sample=True,     # génération aléatoire
                                top_p=0.9,
                                temperature=0.8,
                                logits_processor=[NanClampProcessor()]
                            )
                            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                            print(f"\n=== Step {state.global_step} ===")
                            print(f"Prompt: {p}")
                            print(f"Generated: {text}\n")
                # added
                finally:
                    # restore training settings
                    model.config.use_cache = prev_cache
                    if was_training:
                        model.train()
                        if hasattr(model, "gradient_checkpointing_enable"):
                            model.gradient_checkpointing_enable()

    every_steps = 1 if args.test is not None else 10
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    callbacks=[SampleGenerationCallback(tokenizer, ["\n<|e|>\n<|s|>\nFABRICE\nSalut, ça va?\n<|e|>\n<|s|>\n", "\n<|e|>\n<|s|>\n"], every_steps)])

    trainer.train()

    """
    # Sauvegarde l'adaptateur LoRA
    model.save_pretrained("mtext-111025_mistral-7B-v0.3_lora-adapter")
    tokenizer.save_pretrained("mtext-111025_mistral-7B-v0.3_lora-adapter")
    # Sauvegarder le modèle LoRA sur le hub
    login(token=HF_TOKEN)
    model.push_to_hub("mtext-111025_mistral-7B-v0.3_lora-adapter", use_temp_dir=False)
    tokenizer.push_to_hub("mtext-111025_mistral-7B-v0.3_lora-adapter", use_temp_dir=False)

    # Fusionner les poids LoRA dans le modèle de base
    model = model.merge_and_unload()
    model.config.use_cache = True  # réactive cache pour l'inférence

    model.save_pretrained("mtext-111025_mistral-7B-v0.3_merged")
    tokenizer.save_pretrained("mtext-111025_mistral-7B-v0.3_merged")

    # Sauvegarder le modèle fusionné sur le hub
    model.push_to_hub("mtext-111025_mistral-7B-v0.3_merged", use_temp_dir=False)
    tokenizer.push_to_hub("mtext-111025_mistral-7B-v0.3_merged", use_temp_dir=False)"""


if __name__ == "__main__":
    main()
