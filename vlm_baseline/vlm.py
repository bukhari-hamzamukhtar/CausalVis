"""
vlm_baseline/vlm.py  —  Qwen2.5-VL behind an append-only answer cache
====================================================================

A trimmed copy of the prompter written for the OCVLP course project, which itself follows
VLP's Qwen2_5Prompter (Wuest et al., arXiv:2511.18964): text first, then images, at most
224*224 pixels per image when several are sent, seeded before every generation.

Two backends:
  hf      load the model and generate            (Kaggle GPU)
  replay  answer only from the cache, never generate  (laptop)

Every answer is one JSON line keyed by model, seed, prompt and image paths, so a job that
stops halfway can be restarted and nothing is ever asked twice.
"""

import hashlib
import json
import os
import time

MODEL_IDS = {"Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
             "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct"}
_MODELS = {}


def key_of(model, seed, prompt, paths):
    h = hashlib.sha1()
    h.update(json.dumps([model, seed, prompt, list(paths)], ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


class AnswerCache:
    def __init__(self, files, write_file=None):
        self.table, self.write_file = {}, write_file
        for f in files:
            if f and os.path.exists(f):
                for line in open(f, encoding="utf-8"):
                    if line.strip():
                        rec = json.loads(line)
                        self.table[rec["key"]] = rec
        if write_file:
            os.makedirs(os.path.dirname(os.path.abspath(write_file)), exist_ok=True)

    def get(self, k):
        rec = self.table.get(k)
        return None if rec is None else rec["response"]

    def put(self, rec):
        self.table[rec["key"]] = rec
        if self.write_file:
            with open(self.write_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load(model_name, dtype_name):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    dtype = {"float16": torch.float16, "mixed": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16, "auto": "auto"}[dtype_name]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_IDS[model_name], torch_dtype=dtype, device_map="auto").eval()
    if dtype_name == "mixed":            # T4 has no bfloat16 and the vision tower overflows in fp16
        model.visual.to(torch.float32)
    processor = AutoProcessor.from_pretrained(MODEL_IDS[model_name])
    processor.tokenizer.padding_side = "left"
    return model, processor


class Prompter:
    def __init__(self, model="Qwen2.5-VL-7B-Instruct", seed=0, cache=None,
                 backend="replay", dtype="mixed", max_pixels=224 * 224):
        self.model_name, self.seed, self.cache = model, seed, cache
        self.backend, self.dtype, self.max_pixels = backend, dtype, max_pixels
        self.calls, self.gpu_seconds, self.tokens = 0, 0.0, 0

    def ask(self, prompt_text, paths=(), max_new_tokens=256, key_paths=None):
        """key_paths names the images for the cache key. It lets a job write frames wherever
        the machine has room while the key stays the same, so a run started on one machine
        can be finished on another."""
        paths = list(paths)
        names = list(paths if key_paths is None else key_paths)
        k = key_of(self.model_name, self.seed, prompt_text, names)
        hit = self.cache.get(k) if self.cache else None
        if hit is not None:
            return hit
        if self.backend == "replay":
            raise KeyError("answer not cached: %r %s" % (prompt_text[:60], paths[:1]))
        text, n_tokens, secs = self._generate(prompt_text, paths, max_new_tokens)
        if self.cache:
            self.cache.put({"key": k, "model": self.model_name, "seed": self.seed,
                            "paths": names, "prompt": prompt_text, "response": text,
                            "new_tokens": n_tokens, "seconds": round(secs, 2)})
        return text

    def _generate(self, prompt_text, paths, max_new_tokens):
        import torch
        from qwen_vl_utils import process_vision_info
        pair = (self.model_name, self.dtype)
        if pair not in _MODELS:
            _MODELS[pair] = _load(*pair)
        model, processor = _MODELS[pair]
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        content = [{"type": "text", "text": prompt_text}]
        content += [{"type": "image", "image": p, "max_pixels": self.max_pixels} for p in paths]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        secs = time.time() - t0
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        n_new = int(trimmed[0].shape[0]) if trimmed else 0
        self.calls += 1
        self.gpu_seconds += secs
        self.tokens += n_new
        return (out[0] if out else ""), n_new, secs
