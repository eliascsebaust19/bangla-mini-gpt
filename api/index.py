"""
Vercel Serverless Function for Bangla Mini-GPT.
Powered by lightweight pure NumPy inference (no PyTorch bloat).
"""
import json
import math
import os
import sys
import numpy as np
from flask import Flask, render_template, request, Response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
NPZ_PATH = os.path.join(BASE_DIR, "mini_gpt.npz")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Load Vocab
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()} if isinstance(list(vocab_data["itos"].keys())[0], str) else vocab_data["itos"]
chars = vocab_data["chars"]


def decode(ids):
    return "".join(itos.get(i, itos.get(str(i), "")) for i in ids)


# Pure NumPy Transformer Engine
class MiniGPTNumPy:
    def __init__(self, npz_path=NPZ_PATH):
        data = np.load(npz_path)
        self.w = {k: data[k] for k in data.files}
        self.n_layer = 4
        self.n_head = 6
        self.n_embd = 192
        self.head_dim = self.n_embd // self.n_head
        self.block_size = 128

    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * gamma + beta

    def _gelu(self, x):
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def forward(self, idx_arr):
        T = len(idx_arr)
        pos = np.arange(T)
        x = self.w["tok_emb.weight"][idx_arr] + self.w["pos_emb.weight"][pos]
        for i in range(self.n_layer):
            p = f"blocks.{i}"
            xn = self._layer_norm(x, self.w[f"{p}.ln1.weight"], self.w[f"{p}.ln1.bias"])
            qkv = xn @ self.w[f"{p}.attn.qkv.weight"].T
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, self.n_head, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(T, self.n_head, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(T, self.n_head, self.head_dim).transpose(1, 0, 2)
            att = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            mask = np.triu(np.full((T, T), -np.inf), k=1)
            att += mask
            att = self._softmax(att, axis=-1)
            out = att @ v
            out = out.transpose(1, 0, 2).reshape(T, self.n_embd)
            proj = out @ self.w[f"{p}.attn.proj.weight"].T + self.w[f"{p}.attn.proj.bias"]
            x = x + proj
            xn2 = self._layer_norm(x, self.w[f"{p}.ln2.weight"], self.w[f"{p}.ln2.bias"])
            h = self._gelu(xn2 @ self.w[f"{p}.mlp.0.weight"].T + self.w[f"{p}.mlp.0.bias"])
            mlp_out = h @ self.w[f"{p}.mlp.2.weight"].T + self.w[f"{p}.mlp.2.bias"]
            x = x + mlp_out
        xf = self._layer_norm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        return xf @ self.w["head.weight"].T


print("NumPy Model load kora hocche...")
model = MiniGPTNumPy()
print("Model load সম্পন্ন!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    mode = data.get("mode", "ai").lower()

    mode_prefix = {
        "gf": "[প্রেমিকা] ",
        "friend": "[বন্ধু] ",
        "wife": "[বউ] ",
        "ai": "[এআই] ",
    }.get(mode, "")

    # Normalize nukta characters
    prompt = prompt.replace("ড়", "ড\u09bc").replace("ঢ়", "ঢ\u09bc").replace("য়", "য\u09bc")
    temperature = float(data.get("temperature", 0.4))
    max_tokens = int(data.get("max_tokens", 80))

    temperature = max(0.1, min(temperature, 1.5))
    max_tokens = max(10, min(max_tokens, 150))

    def sse_stream():
        formatted_prompt = f"{mode_prefix}প্রশ্ন: {prompt}\nউত্তর:"
        clean_chars = [c for c in formatted_prompt if c in stoi]
        if not clean_chars:
            idx = [0]
        else:
            idx = [stoi[c] for c in clean_chars]

        accumulated = ""
        for _ in range(max_tokens):
            cond = idx[-model.block_size:]
            logits = model.forward(np.array(cond))[-1] / max(temperature, 1e-4)
            probs = model._softmax(logits)
            next_id = int(np.random.choice(len(probs), p=probs))
            idx.append(next_id)

            char = decode([next_id])
            if char == "\n" and len(accumulated.strip()) > 0:
                break
            accumulated += char
            yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    response = Response(sse_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/auto_train", methods=["POST", "GET"])
def auto_train_info():
    return {
        "status": "info",
        "message": "Vercel Serverless মোডে লাইভ চ্যাট সক্রিয় রয়েছে। নতুন ডেটা দিয়ে মডেল ট্রেইনিং লোকাল কম্পিউটারে করতে হবে।"
    }


@app.route("/api/stream_auto_train", methods=["GET", "POST"])
def stream_auto_train_info():
    def dummy_stream():
        yield f"data: {json.dumps({'type': 'log', 'text': 'Vercel Serverless ক্লাউডে ডিপ্লয় করা রয়েছে।'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'text': 'লাইভ চ্যাট শতভাগ সক্রিয়! নতুন ট্রেইনিং লোকাল পিসিতে চালু করুন।'}, ensure_ascii=False)}\n\n"
    return Response(dummy_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
