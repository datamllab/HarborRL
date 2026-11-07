# Models

## Qwen

`qwen` is the only registered model family. A configuration selects a Slime model
preset through `model.args_file`; the value must begin with `qwen` and contain only
letters, numbers, underscores, and hyphens.

```yaml
model:
  args_file: qwen3-8B
  checkpoint: /models/qwen3-8b
  reference: /models/qwen3-8b-reference
```

The example preset is a configuration name consumed by the pinned backend stack. A
matching prefix alone does not prove that the checkpoint exists, fits in memory, or
is compatible with a particular backend installation.

## Protocol path

| Boundary | Implemented contract |
| --- | --- |
| Serving backend | SGLang |
| Harness request protocol | Anthropic Messages v1 |
| Generated tool protocol | `qwen-tool-call-v1` |
| Training backend | `slime-v0.3.2-native-v1` |

The adapter renders messages with the tokenizer's chat template using
`enable_thinking=False`, requires real nonnegative token IDs, and parses Qwen tool
calls from generated text.

## Identity checks

The training schema requires two lowercase SHA-256 fingerprints:

- `gateway.tokenizer_digest` identifies the tokenizer behavior used by rollout.
- `gateway.template_digest` identifies the chat template.

The configuration wizard can calculate both using the selected GPU Python and local
or SSH checkpoint. It loads tokenizer files offline with
`trust_remote_code=False`; it does not download a model.

## Limits

- No other model family is registered.
- Memory fit is a live deployment concern, not a schema guarantee.
- Tokenizer and template fingerprints are checked against runtime evidence.
- Native consistency enforcement currently supports dense models with PP=CP=1;
  see [train / inference consistency](../train-inference-consistency.md).
