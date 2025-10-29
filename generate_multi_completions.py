#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch

def detect_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(
        description="Génère plusieurs complétions avec un modèle Mistral-7B local (mode complétion)."
    )
    parser.add_argument("--model-path", required=True,
                        help="Chemin local du modèle (dossier contenant config.json, model.safetensors, etc.).")
    parser.add_argument("--adapters-path", default=None,
                        help="(Optionnel) Chemin des adaptateurs LoRA (PEFT) si le FT est séparé.")
    parser.add_argument("--n", type=int, default=3, help="Nombre de complétions à générer.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Longueur max générée.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Température d'échantillonnage.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus sampling).")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Pénalité de répétition.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    parser.add_argument("--no-sampling", action="store_true",
                        help="Désactive l'échantillonnage (greedy). Ignore temperature/top-p.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Utile si le repo local a du code custom HF.")
    args = parser.parse_args()
    print(f"[INFO] args.model_path : {args.model_path}")
    print(f"[INFO] args.adapters_path : {args.adapters_path}")

    CACHE_DIR = "/Users/nz-xent/Documents/Models-cache/hf"

    torch.manual_seed(args.seed)

    device = detect_device()
    print(f"[INFO] Périphérique détecté : {device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # dtype recommandé : fp16 sur mps/cuda, fp32 sur cpu
    if device in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Chargement tokenizer
    print(f"[INFO] chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapters_path if args.adapters_path else args.model_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        #local_files_only=True,
    )

    # Gestion du token de padding pour causal LM si manquant
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # pratique pour génération batched

    # Chargement modèle
    print("[INFO] Chargement du modèle…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=CACHE_DIR, 
        torch_dtype=dtype,
        device_map="auto",         # nécessite accelerate, répartit sur mps/cuda si possible
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        #local_files_only=True,
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

    # Option LoRA (PEFT)
    if args.adapters_path:
        try:
            from peft import PeftModel
        except ImportError:
            print("[ERREUR] peft n'est pas installé. `pip install peft`", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Chargement des adaptateurs LoRA depuis: {args.adapters_path}")
        model = PeftModel.from_pretrained(
            model,
            args.adapters_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        # (Optionnel) fusionner les poids LoRA pour l’inférence pure
        try:
            model = model.merge_and_unload()
            print("[INFO] Adaptateurs LoRA fusionnés dans le modèle.")
        except Exception:
            print("[INFO] Fusion LoRA non effectuée (non nécessaire). Continuer sans fusion.")

    # Déplacement manuel si device_map n'a pas tout géré (sécurité)
    if device == "cuda":
        model = model.to("cuda")
    elif device == "mps":
        model = model.to("mps")
    else:
        model = model.to("cpu")

    model.eval()

    # --- Mode interactif (REPL) ---
    turn = 0
    help_text = (
        "\n[REPL] Entrez un prompt puis Entrée pour générer."
        "\n       Commandes: /n <int>, /max <int>, /temp <float>, /top_p <float>, /rep <float>, /greedy, /sample, /show, /help, /quit"
        f"\n       Valeurs: n={args.n}, max={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}, rep={args.repetition_penalty}, sampling={'off' if args.no_sampling else 'on'}\n"
    )
    print(help_text)

    def show_params():
        print(f"[PARAMS] n={args.n} | max={args.max_new_tokens} | temp={args.temperature} | top_p={args.top_p} | rep={args.repetition_penalty} | sampling={'off' if args.no_sampling else 'on'}")

    def strip_prefix(full_text, prefix):
        return full_text[len(prefix):] if full_text.startswith(prefix) else full_text

    import threading, signal, time
    from transformers import TextIteratorStreamer, StoppingCriteria

    class StopOnEvent(StoppingCriteria):
        def __init__(self, event): self.event = event
        def __call__(self, input_ids, scores, **kwargs): return self.event.is_set()

    gen_time_limit = None  # en secondes; configurable via /time

    while True:
        try:
            line = input("\nPrompt (/help pour l’aide, /quit pour sortir) > ").rstrip()
        except KeyboardInterrupt:
            print("\n[INFO] Interrompu. Bye.")
            break

        if not line:
            continue
        if line in ("/quit", "/q", ":q", "exit"):
            print("[INFO] Bye.")
            break
        if line in ("/help",):
            print(help_text)
            continue
        if line in ("/show",):
            show_params(); continue
        if line.startswith("/time "):
            try:
                val = float(line.split()[1])
                gen_time_limit = max(0.0, val)
                print(f"[OK] time limit = {gen_time_limit or 'None'} s")
            except:
                print("[ERR] /time <float_seconds>")
            continue
        if line.startswith("/n "):
            try: args.n = max(1, int(line.split()[1])); print(f"[OK] n={args.n}")
            except: print("[ERR] /n <int>")
            continue
        if line.startswith("/max "):
            try: args.max_new_tokens = max(1, int(line.split()[1])); print(f"[OK] max={args.max_new_tokens}")
            except: print("[ERR] /max <int>")
            continue
        if line.startswith("/temp "):
            try: args.temperature = float(line.split()[1]); print(f"[OK] temp={args.temperature}")
            except: print("[ERR] /temp <float>")
            continue
        if line.startswith("/top_p "):
            try: args.top_p = float(line.split()[1]); print(f"[OK] top_p={args.top_p}")
            except: print("[ERR] /top_p <float>")
            continue
        if line.startswith("/rep "):
            try: args.repetition_penalty = float(line.split()[1]); print(f"[OK] rep={args.repetition_penalty}")
            except: print("[ERR] /rep <float>")
            continue
        if line == "/greedy":
            args.no_sampling = True; print("[OK] sampling=off"); continue
        if line == "/sample":
            args.no_sampling = False; print("[OK] sampling=on"); continue

        prompt = line
        turn += 1
        # graine différente par tour pour diversifier (si même prompt)
        torch.manual_seed(args.seed + turn)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
        )
        if args.no_sampling:
            gen_kwargs.update(dict(do_sample=False))
        else:
            gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))

        print("[INFO] Génération… (Ctrl+C pour stopper)")
        # --- Streaming + arrêt Ctrl+C ---
        stop_event = threading.Event()
        stopper = StopOnEvent(stop_event)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # On génère n séquences en série pour garder un contrôle fin (et éviter les gros batchs qui bloquent).
        for k in range(args.n):
            started_at = time.time()
            thread = threading.Thread(
                target=lambda: model.generate(
                    **inputs,
                    **gen_kwargs,
                    num_return_sequences=1,
                    streamer=streamer,
                    stopping_criteria=[stopper],
                    max_time=gen_time_limit,  # None si non défini
                ),
                daemon=True,
            )
            thread.start()

            print("\n" + "="*80)
            print(f"--- Complétion {k+1} / {args.n} ---\n")
            buf = []
            try:
                for tok in streamer:
                    # Affiche au fil de l’eau
                    print(tok, end="", flush=True)
                    buf.append(tok)
                    # Limite temps dure si demandée (pour macOS/MPS où Ctrl+C peut traîner)
                    if gen_time_limit and (time.time() - started_at) > gen_time_limit:
                        stop_event.set()
                        break
            except KeyboardInterrupt:
                # Stop immédiat de la génération en cours
                stop_event.set()
                print("\n[INFO] Arrêt demandé (Ctrl+C).")
            finally:
                thread.join(timeout=1.0)
                print("\n" + "="*80)

if __name__ == "__main__":
    main()
