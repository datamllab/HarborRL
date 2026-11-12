"""Inference-only local HTTP service for a pinned Qwen math process model."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from harborrl.math import prm
from harborrl.trajectories.native import digest


class Scorer:
    def __init__(self, config, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.frozen = prm.freeze(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'], trust_remote_code=True,
                                                       local_files_only=True)
        self.model = AutoModel.from_pretrained(
            config['checkpoint'], torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, attn_implementation='sdpa').to(device).eval()
        self.model.requires_grad_(False)
        self.marker_id = self.tokenizer.encode(prm.MARKER, add_special_tokens=False)
        if len(self.marker_id) != 1:
            raise ValueError('PRM marker must encode to one token')
        if prm.freeze(config) != self.frozen:
            raise ValueError('PRM checkpoint changed while loading the scorer')

    def info(self):
        frozen = not self.model.training and all(not parameter.requires_grad for parameter in self.model.parameters())
        return {key: self.frozen[key] for key in ('identity', 'revision', 'adapter', 'checkpoint')} | {
            'frozen': frozen, 'is_generation': False,
            'max_context': self.model.config.max_position_embeddings}

    def score(self, payload):
        import torch

        if not isinstance(payload, dict) or set(payload) != {'identity', 'query', 'steps'}:
            raise ValueError('PRM score requires identity, query and steps')
        expected = prm.request_payload(payload['identity'], payload['query'], payload['steps'])
        if expected['identity'] != self.frozen['identity'] or not self.info()['frozen']:
            raise ValueError('PRM scoring requires the frozen manifest identity')
        messages = [{'role': 'system', 'content': prm.SYSTEM},
                    {'role': 'user', 'content': payload['query']},
                    {'role': 'assistant', 'content': prm.MARKER.join(payload['steps']) + prm.MARKER}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.model.config.max_position_embeddings:
            raise ValueError('PRM scoring context exceeded; truncation is not permitted')
        positions = [index for index, token in enumerate(ids) if token == self.marker_id[0]]
        if len(positions) != len(payload['steps']):
            raise ValueError('PRM marker positions differ from the submitted steps')
        with torch.inference_mode():
            inputs = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            logits = self.model(input_ids=inputs, use_cache=False).logits
            if logits.shape != (1, len(ids), 2) or not torch.isfinite(logits).all():
                raise ValueError('PRM must return finite two-class token logits')
            values = logits[0, positions].float().softmax(-1)[:, 1].cpu().tolist()
        return {'identity': self.frozen['identity'], 'request_digest': digest(expected),
                'scorer_input_digest': digest(ids), 'probabilities': values}


def serve(scorer, host, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            content = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            self.reply(200, scorer.info()) if self.path == '/model_info' else self.reply(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/score':
                return self.reply(404, {'error': 'not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError('PRM request size must be between 1 byte and 1 MiB')
                result = scorer.score(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError, KeyError) as exc:
                return self.reply(400, {'error': str(exc)})
            self.reply(200, result)

    with HTTPServer((host, port), Handler) as server:
        print(json.dumps({'ready': True, **scorer.info()}), flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=30091)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = {'checkpoint': args.checkpoint, 'revision': args.revision,
              'endpoint': f'http://127.0.0.1:{args.port}'}
    serve(Scorer(config, args.device), args.host, args.port)


if __name__ == '__main__':
    main()
