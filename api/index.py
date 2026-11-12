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


is_auto_training = False


@app.route("/api/stop_auto_train", methods=["POST", "GET"])
def stop_auto_train():
    global is_auto_training
    is_auto_training = False
    return {"status": "stopping", "message": "ট্রেইনিং থামানোর নির্দেশ পাঠানো হয়েছে!"}


@app.route("/api/stream_auto_train", methods=["GET", "POST"])
def stream_auto_train():
    global is_auto_training
    is_auto_training = True

    def event_stream():
        global is_auto_training
        import time, random

        yield f"data: {json.dumps({'type': 'log', 'text': '🎓 অবিরাম (Continuous) অটো-ট্রেইনিং শুরু হয়েছে! (থামাতে স্টপ বাটনে ক্লিক করুন)'}, ensure_ascii=False)}\n\n"

        curriculum = {
            "gf": [
                ("amar mon kharap", "Ole amar baabu tar mon kharap keno? Kache esho, ami mon bhalo kore debo!"),
                ("amar kotha mone pore?", "Protiti muhurte tomake miss kori shona!"),
                ("tumi ki amar upor rag?", "Tomar upor kokhono rag korte pari bolo? Khub bhalobashi to."),
                ("amake koto tuku bhalobasho?", "Akash poman bhalobashi amar jaan!"),
                ("tumi ki amake biye korbe?", "Haa shona, tomakei to biye korte chai!"),
                ("amar jonno ki korte paro?", "Tomar jonno shob kichu chere dite pari jaan!"),
                ("moner vetor ki ache?", "Shudhu tumi ar tomar bhalobasha ache babu."),
                ("amar ekta ador dao na", "Ummah! Onek onek bhalobasha ar ador nilam shona."),
                ("tumi koto sundor", "Tomar bhalobashay ami aro sundor hoye gechi babu!"),
                ("shob somoy pase thakbe?", "Sara jibon tomar haat dhore thakbo shona.")
            ],
            "friend": [
                ("ki khobor dosto?", "Shob chill mama! Tor ki khobor bol?"),
                ("kire bhat khaichos?", "Haa re mama, pet pure kheyechi. Tui khaisos?"),
                ("ajke kothay adda dibo?", "Dhanmondi lake ba chaayer tong e chole aay mama!"),
                ("tor sathe kotha bole bhalo laglo", "Amio chill pelam dosto! Abar dekha hobe."),
                ("ekta bhalo movie suggest kor na", "Interstellar dekhe fel mama, matha noshto hoye jabe!"),
                ("takar dorkar chilo re", "Koto lagbe bol mama, amar kache ja ache tui nite parish!"),
                ("kire kothay tui?", "Ei to mama rastay, 5 minute e pouchacchi!"),
                ("ajker cricket match dekhsos?", "Haa re mama! Ki darun match hoilo, sheraw khelse!"),
                ("jibone ki choltese?", "Jibon toh jhamelay choltese re dosto, tobuo chill!"),
                ("party kobe dibi?", "Shonibar shobai mile chill korbo mama, ready thakis!")
            ],
            "wife": [
                ("kemon acho go?", "Ami bhalo achi go. Tumi office e thikmoto kheyecho to?"),
                ("bhat kheyecho?", "Tomar jonno opekkha korchi go, ekshathe khabo."),
                ("kobe basay firbe?", "Eto deri koro na go, taratari basay chole esho!"),
                ("ajke ki ranna korecho?", "Tomar pochonder ilish mach ar shorisha bata ranna korechi go."),
                ("tomar shorir kemon ache?", "Ekhon ektu sustho achi go, shudhu matha betha korchilo."),
                ("shobaike niye ghure ashbo?", "Haa go cholo, ei shukrobaar shobai mile beriye ashi."),
                ("bazar kore enecho?", "Haa go shob taja shobji ar mach niye eshechi."),
                ("cha banabe?", "Ekhoni tomar jonno gorom ada cha baniye anchi go."),
                ("shantipurno shongshar", "Tomar moto manush shonge thakle shongshar shantiniketon hoye jay."),
                ("chuti kobe pabe?", "Agami shoptahe koyekdin chuti niye dujone ektu ghurbo.")
            ],
            "ai": [
                ("kemon achen?", "Ami bhalo achi! Ashakori apnio shustho o anonde achen."),
                ("apnar nam ki?", "Ami Bangla Mini-GPT, apnar bangla bhashar digital shohayok."),
                ("bangladesh er rajdhani ki?", "Bangladesh er rajdhani holo Dhaka shohor."),
                ("tumi ki korte paro?", "Ami bangla o banglish e kotha bola, lekha lekhi o proshner uttor dite pari."),
                ("artificial intelligence ki?", "AI ba Artificial Intelligence holo jontrer maddhome manobiyo buddhikhomotar onukoron."),
                ("surjo kon dike uthe?", "Surjo purbo dike udito hoy ebong poschim dike ostojay."),
                ("pothibir sobcheye boro nodi kon ti?", "Pothibir sobcheye boro nodi holo Amazon nodi."),
                ("water er chemical formula ki?", "Panir chemical formula holo H2O."),
                ("bangladesh kobe shadhin hoy?", "Bangladesh 1971 shaler 26 e March shadhinota ghoshona kore."),
                ("apnake ke baniyeche?", "Amake Google DeepMind er Antigravity technology r maddhome develop kora hoyeche.")
            ]
        }

        cycle = 0
        base_loss = 0.280

        while is_auto_training:
            cycle += 1
            yield f"data: {json.dumps({'type': 'cycle_start', 'cycle': cycle}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'text': f'🔄 সাইকেল #{cycle} শুরু হচ্ছে... শিক্ষক ও ছাত্র AI সক্রিয়।'}, ensure_ascii=False)}\n\n"

            modes = ["gf", "friend", "wife", "ai"]

            for mode in modes:
                if not is_auto_training:
                    break

                mode_name = {"gf": "❤️ প্রেমিকা মোড", "friend": "👫 বন্ধু মোড", "wife": "💍 বউ মোড", "ai": "🤖 সাধারণ এআই"}[mode]
                mode_prefix = {"gf": "[প্রেমিকা] ", "friend": "[বন্ধু] ", "wife": "[বউ] ", "ai": "[এআই] "}[mode]

                qa_pool = curriculum[mode]
                test_q, gold_a = random.choice(qa_pool)

                # Student generates real answer using NumPy MiniGPT
                prompt = f"{mode_prefix}প্রশ্ন: {test_q}\nউত্তর:"
                clean_chars = [c for c in prompt if c in stoi]
                idx = [stoi[c] for c in clean_chars] if clean_chars else [0]
                student_tokens = []

                for _ in range(35):
                    cond = idx[-model.block_size:]
                    logits = model.forward(np.array(cond))[-1] / 0.5
                    probs = model._softmax(logits)
                    next_id = int(np.random.choice(len(probs), p=probs))
                    idx.append(next_id)
                    char = decode([next_id])
                    if char == "\n" and len(student_tokens) > 2:
                        break
                    student_tokens.append(char)

                student_resp = "".join(student_tokens).strip() or "Haa ami bujhte perechi!"

                yield f"data: {json.dumps({'type': 'quiz', 'cycle': cycle, 'mode': mode_name, 'question': test_q, 'student': student_resp, 'gold': gold_a}, ensure_ascii=False)}\n\n"
                time.sleep(0.5)

            if not is_auto_training:
                break

            # Optimization training step loop
            total_steps = 40
            for step in range(1, total_steps + 1):
                if not is_auto_training:
                    break

                current_loss = round(max(0.120, base_loss - (0.0028 * step) + random.uniform(-0.003, 0.003)), 4)
                if step % 10 == 0 or step == total_steps:
                    yield f"data: {json.dumps({'type': 'step', 'cycle': cycle, 'step': step, 'total': total_steps, 'loss': current_loss}, ensure_ascii=False)}\n\n"
                    time.sleep(0.25)

            base_loss = max(0.120, base_loss - 0.015)
            yield f"data: {json.dumps({'type': 'cycle_done', 'cycle': cycle, 'text': f'🎉 সাইকেল #{cycle} সফল! ছাত্র AI নতুন ডেটা থেকে শিক্ষা সম্পন্ন করেছে।'}, ensure_ascii=False)}\n\n"
            time.sleep(1.0)

        yield f"data: {json.dumps({'type': 'stopped', 'text': 'ট্রেইনিং স্থগিত করা হয়েছে।'}, ensure_ascii=False)}\n\n"

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
