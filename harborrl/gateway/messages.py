"""Strict text/tool subset of Anthropic Messages and buffered SSE encoding."""
import json
import re

from harborrl.trajectories.native import canonical_sampling, finite


class ProtocolError(ValueError):
    pass


def unavailable_tool_error(name):
    """Exact native Claude CLI observation for an unavailable tool."""
    return f"<tool_use_error>Error: No such tool available: {name}</tool_use_error>"


def keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ProtocolError(f"unsupported fields in {label}")


def text_blocks(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ProtocolError("text content must be a string or blocks")
    out = []
    for block in value:
        keys(block, ("type", "text", "cache_control"), "text block")
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise ProtocolError("only text blocks are enabled")
        out.append(block["text"])
    return "\n".join(out)


def convert(payload, *, model, max_output_tokens=8192, count_only=False):
    keys(payload, ("model", "messages", "system", "tools", "tool_choice", "max_tokens",
                   "temperature", "top_p", "top_k", "stop_sequences", "stream", "metadata",
                   "thinking", "context_management", "output_config"), "request")
    if payload.get("model") != model:
        raise ProtocolError("model alias is not bound to this policy profile")
    if type(payload.get("stream", False)) is not bool:
        raise ProtocolError("stream must be boolean")
    max_tokens = payload.get("max_tokens", 1 if count_only else None)
    if type(max_tokens) is not int or not 0 < max_tokens <= max_output_tokens:
        raise ProtocolError("max_tokens outside profile budget")
    temperature, top_p, top_k = payload.get("temperature", 1.0), payload.get("top_p", 1.0), payload.get("top_k", -1)
    # Initial raw-model recipe fixes the distribution, regardless of JSON numeric type.
    if any(not finite(v) for v in (temperature, top_p)) or temperature != 1 or top_p != 1 or type(top_k) is not int or top_k != -1:
        raise ProtocolError("initial profile requires temperature=1, top_p=1, top_k=-1")
    stop = payload.get("stop_sequences", [])
    if not isinstance(stop, list) or any(not isinstance(x, str) or not x for x in stop):
        raise ProtocolError("invalid stop sequences")
    if payload.get("metadata") is not None:
        keys(payload["metadata"], ("user_id",), "metadata")
    # Claude Code 2.1.x emits these controls even when adaptive thinking is
    # disabled.  Accept only the exact no-op forms for the thinking-disabled
    # Qwen chat template; no unknown policy semantics can cross the gateway.
    if payload.get("thinking") not in (None, {"type": "adaptive"}):
        raise ProtocolError("unsupported thinking control")
    if payload.get("context_management") not in (
            None, {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}):
        raise ProtocolError("unsupported context management control")
    effort = payload.get("output_config")
    if effort is not None and (not isinstance(effort, dict) or set(effort) != {"effort"}
                               or effort["effort"] not in ("low", "medium", "high", "xhigh")):
        raise ProtocolError("unsupported output configuration")
    if not isinstance(payload.get("tools", []), list):
        raise ProtocolError("tools must be a list")
    tools, names = [], set()
    for tool in payload.get("tools", []):
        keys(tool, ("name", "description", "input_schema", "cache_control"), "tool")
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in names or not isinstance(tool.get("input_schema"), dict):
            raise ProtocolError("invalid or duplicate tool")
        names.add(name)
        tools.append({"type": "function", "function": {"name": name, "description": tool.get("description", ""), "parameters": tool["input_schema"]}})
    # Forced tool sampling requires a separate backend contract; fail closed.
    choice = payload.get("tool_choice", {"type": "auto"})
    if choice != {"type": "auto"}:
        raise ProtocolError("only automatic tool choice is enabled")
    messages = []
    if "system" in payload:
        messages.append({"role": "system", "content": text_blocks(payload["system"])})
    history = payload.get("messages")
    if not isinstance(history, list) or not history:
        raise ProtocolError("nonempty messages required")
    pending, seen = {}, set()
    for item in history:
        keys(item, ("role", "content"), "message")
        role = item.get("role")
        if role not in ("system", "user", "assistant"):
            raise ProtocolError("unsupported message role")
        blocks = item.get("content")
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if not isinstance(blocks, list):
            raise ProtocolError("message content must be blocks or string")
        texts, calls, results = [], [], []
        for block in blocks:
            if not isinstance(block, dict):
                raise ProtocolError("invalid content block")
            kind = block.get("type")
            if kind == "text":
                texts.append(text_blocks([block]))
            elif kind == "tool_use" and role == "assistant":
                keys(block, ("type", "id", "name", "input", "cache_control"), "tool_use")
                call_id = block.get("id")
                name = block.get("name")
                if (not isinstance(call_id, str) or not call_id or call_id in seen
                        or not isinstance(name, str) or not name or not isinstance(block.get("input"), dict)):
                    raise ProtocolError("invalid tool_use identity or input")
                seen.add(call_id)
                pending[call_id] = name
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": block["name"], "arguments": json.dumps(block["input"], ensure_ascii=False, allow_nan=False)}})
            elif kind == "tool_result" and role == "user":
                keys(block, ("type", "tool_use_id", "content", "is_error", "cache_control"), "tool_result")
                call_id = block.get("tool_use_id")
                if call_id not in pending or type(block.get("is_error", False)) is not bool:
                    raise ProtocolError("orphan or duplicate tool_result")
                name = pending.pop(call_id)
                # Preserve error flag as part of model-visible tool observation.
                content = text_blocks(block.get("content", ""))
                if name not in names and (block.get("is_error") is not True
                                          or content != unavailable_tool_error(name)):
                    raise ProtocolError("undeclared tool requires its exact native rejection")
                if block.get("is_error"):
                    content = "[tool_error]\n" + content
                results.append({"role": "tool", "tool_call_id": call_id, "content": content})
            else:
                raise ProtocolError(f"unsupported content block: {kind}")
        if results:
            messages.extend(results)
        if texts or calls:
            msg = {"role": role, "content": "\n".join(texts)}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
    if pending:
        raise ProtocolError("missing tool results in generation context")
    return {"messages": messages, "tools": tools, "sampling": canonical_sampling({
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "max_new_tokens": max_tokens, "stop": stop})}


def parse_qwen_output(text, *, response_id, tools, allow_unknown_tools=False):
    """Deterministic Qwen tool markers; malformed model output is never repaired."""
    if "<think>" in text or "</think>" in text:
        raise ProtocolError("thinking output is not enabled by this profile")
    names = {}
    for tool in tools:
        name = tool["function"]["name"]
        alias = name.casefold()
        if alias in names:
            raise ProtocolError("ambiguous generated tool alias")
        names[alias] = name
    blocks, offset = [], 0
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    remainder = re.sub(r"<tool_call>\s*(.*?)\s*</tool_call>", "", text, flags=re.DOTALL)
    if "<tool_call>" in remainder or "</tool_call>" in remainder:
        raise ProtocolError("incomplete tool call")
    for index, match in enumerate(matches):
        if text[offset:match.start()]:
            blocks.append({"type": "text", "text": text[offset:match.start()]})
        try:
            call = json.loads(match[1], parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite tool argument")))
        except ValueError as exc:
            raise ProtocolError("invalid generated tool JSON") from exc
        keys(call, ("name", "arguments"), "generated tool")
        generated_name = call.get("name")
        if generated_name == "finish" and call.get("arguments") == {}:
            if text[match.end():]:
                raise ProtocolError("finish marker must be the final output")
            return blocks
        canonical_name = names.get(generated_name.casefold()) if isinstance(generated_name, str) else None
        if canonical_name is None and (not allow_unknown_tools or not isinstance(generated_name, str) or not generated_name):
            raise ProtocolError(f"unknown generated tool: {generated_name!r}")
        if canonical_name is None:
            canonical_name = generated_name  # Preserve the invalid action for native CLI rejection.
        if not isinstance(call.get("arguments"), dict):
            raise ProtocolError("generated tool arguments are not an object")
        blocks.append({"type": "tool_use", "id": f"toolu_{response_id}_{index}", "name": canonical_name, "input": call["arguments"]})
        offset = match.end()
    if text[offset:] or not blocks:
        blocks.append({"type": "text", "text": text[offset:]})
    return blocks


def sse_events(message):
    """Buffer generation before delivery; emit valid SSE, not token-latency streaming."""
    start = {**message, "content": [], "stop_reason": None, "stop_sequence": None,
             "usage": {**message["usage"], "output_tokens": 0}}
    yield "message_start", {"type": "message_start", "message": start}
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            initial = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            initial = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False, allow_nan=False)}
        yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": initial}
        yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
        yield "content_block_stop", {"type": "content_block_stop", "index": index}
    yield "message_delta", {"type": "message_delta", "delta": {
        "stop_reason": message["stop_reason"], "stop_sequence": message["stop_sequence"]},
        "usage": {"output_tokens": message["usage"]["output_tokens"]}}
    yield "message_stop", {"type": "message_stop"}
