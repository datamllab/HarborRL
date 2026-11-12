"""Small exact-token Transformers adapter; imported only by training/evaluation."""
import hashlib

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from harborrl.trajectories.native import digest


class ByteTokenizer:
    """Test-only byte tokenizer with explicit BOS/EOS/PAD and no hidden template."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def encode(self, text):
        return [1, *[value + 3 for value in text.encode("utf-8")]]

    def decode(self, tokens):
        return bytes(token - 3 for token in tokens if token >= 3).decode("utf-8", errors="replace")

    def identity(self):
        return {"tokenizer_digest": digest("test-byte-v1"), "template_digest": digest("literal-prompt-with-bos-v1"),
                "vocabulary_digest": digest(list(range(259))), "vocab_size": 259}


class LocalTokenizer:
    def __init__(self, checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.eos_token_id is None or self.pad_token_id is None:
            raise ValueError("Tokenizer requires explicit EOS and PAD IDs")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def identity(self):
        tok = self.tokenizer
        if not tok.is_fast:
            raise ValueError("Exact identity requires a serializable fast tokenizer")
        return {"tokenizer_digest": digest(tok.backend_tokenizer.to_str()),
                "template_digest": digest({"serialization": "literal-prompt-add-special-tokens-v1",
                    "chat_template": tok.chat_template, "special_tokens": tok.special_tokens_map}),
                "vocabulary_digest": digest(sorted(tok.get_vocab().items())), "vocab_size": len(tok)}


def load_student(config, device, *, teacher=False):
    spec = config["teacher"] if teacher else config["student"]
    torch.manual_seed(config["training"]["seed"] + int(teacher))
    if spec["kind"] == "tiny_qwen":
        dimensions = config["student"]
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=259, hidden_size=dimensions["hidden_size"],
            intermediate_size=dimensions["hidden_size"] * 2, num_hidden_layers=dimensions["layers"],
            num_attention_heads=dimensions["heads"], num_key_value_heads=dimensions["heads"],
            max_position_embeddings=config["sampling"]["max_context"], attention_dropout=0.0,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, use_sliding_window=False,
            _attn_implementation="eager"))
        tokenizer = ByteTokenizer()
    else:
        tokenizer = LocalTokenizer(spec["checkpoint"])
        model = AutoModelForCausalLM.from_pretrained(spec["checkpoint"], local_files_only=True,
                                                    trust_remote_code=False, torch_dtype=torch.float32,
                                                    attn_implementation="eager")
    if model.config.vocab_size != tokenizer.identity()["vocab_size"]:
        raise ValueError("Model vocabulary and tokenizer must match exactly (no padded-vocab inference)")
    if config["sampling"]["max_context"] > model.config.max_position_embeddings:
        raise ValueError("Configured context exceeds model capacity")
    model.to(device).eval()
    if teacher:
        model.requires_grad_(False)
    return model, tokenizer


def parameter_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def response_logprobs(model, prompts, responses, pad_id):
    """Causal shift: response[0] is scored by the final prompt position."""
    device = next(model.parameters()).device
    sequences = [p + r for p, r in zip(prompts, responses, strict=True)]
    maximum = max(map(len, sequences))
    ids = torch.full((len(sequences), maximum), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for index, tokens in enumerate(sequences):
        ids[index, :len(tokens)] = torch.tensor(tokens, device=device)
        attention[index, :len(tokens)] = 1
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits.float()
    results = []
    for index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        logprobs = logits[index, len(prompt) - 1:len(prompt) + len(response) - 1].log_softmax(-1)
        results.append(logprobs.gather(-1, torch.tensor(response, device=device).unsqueeze(-1)).squeeze(-1))
    return results


@torch.no_grad()
def generate(model, tokenizer, prompt_ids, maximum, context, seed):
    if not prompt_ids or len(prompt_ids) + maximum > context:
        raise ValueError("Prompt plus response budget exceeds context; no implicit truncation")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens, probabilities = [], []
    for _ in range(maximum):
        ids = torch.tensor([prompt_ids + tokens], device=device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
        logprobs = logits.log_softmax(-1)
        token = int(torch.multinomial(logprobs.exp(), 1, generator=generator))
        tokens.append(token)
        probabilities.append(float(logprobs[token]))
        if token == tokenizer.eos_token_id:
            return tokens, probabilities, "eos"
    return tokens, probabilities, "length"
