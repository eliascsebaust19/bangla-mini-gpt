"""
Bangla Mini-GPT Web UI Server.
Flask server providing Gemini/ChatGPT style streaming chat interface.

Usage:
    python app.py
"""
import json
import os
import sys
import torch
from flask import Flask, render_template, request, Response
from mini_gpt import MiniGPT, encode, decode, stoi, device, block_size

app = Flask(__name__)

# Load Model
MODEL_PATH = "mini_gpt.pt"
if not os.path.exists(MODEL_PATH):
    print(f"Error: '{MODEL_PATH}' khuje pawa jayni! Age mini_gpt.py diye model train korun.")
    sys.exit(1)

print("Bangla Mini-GPT model load kora hocche...")
model = MiniGPT().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Model load সম্পন্ন! Device: {device}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    mode = data.get("mode", "ai").lower()

    # Map mode to persona tag
    mode_prefix = {
        "gf": "[প্রেমিকা] ",
        "girlfriend": "[প্রেমিকা] ",
        "friend": "[বন্ধু] ",
        "wife": "[বউ] ",
        "ai": "[এআই] ",
    }.get(mode, "")

    # Normalize nukta characters to match vocabulary
    prompt = prompt.replace("ড়", "ড\u09bc").replace("ঢ়", "ঢ\u09bc").replace("য়", "য\u09bc")
    temperature = float(data.get("temperature", 0.4))
    max_tokens = int(data.get("max_tokens", 300))

    # Clamp parameters
    temperature = max(0.1, min(temperature, 2.0))
    max_tokens = max(10, min(max_tokens, 800))

    def sse_stream():
        # Conversational prompt format with persona: '[মোড] প্রশ্ন: ... \nউত্তর:'
        formatted_prompt = f"{mode_prefix}প্রশ্ন: {prompt}\nউত্তর:"
        clean_chars = [c for c in formatted_prompt if c in stoi]
        if not clean_chars:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            idx = torch.tensor([encode("".join(clean_chars))], dtype=torch.long, device=device)

        accumulated = ""
        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)

                char = decode([next_id.item()])

                # Stop generation when the answer concludes (newline after text)
                if char == "\n" and len(accumulated.strip()) > 0:
                    break

                accumulated += char
                yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    response = Response(sse_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/reload_model", methods=["POST", "GET"])
def reload_model():
    global model
    try:
        new_model = MiniGPT().to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        new_model.load_state_dict(state_dict)
        new_model.eval()
        model = new_model
        return {"status": "success", "message": "Model reloaded into memory successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


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
        import time
        from auto_trainer import TeacherAI, StudentAI
        teacher = TeacherAI()
        student = StudentAI()

        yield f"data: {json.dumps({'type': 'log', 'text': '🎓 অবিরাম (Continuous) অটো-ট্রেইনিং শুরু হয়েছে! (থামাতে স্টপ বাটনে ক্লিক করুন)'}, ensure_ascii=False)}\n\n"

        cycle = 0
        while is_auto_training:
            cycle += 1
            yield f"data: {json.dumps({'type': 'cycle_start', 'cycle': cycle}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'text': f'🔄 সাইকেল #{cycle} শুরু হচ্ছে...'}, ensure_ascii=False)}\n\n"

            modes = ["gf", "friend", "wife", "ai"]
            new_blocks = []

            for mode in modes:
                if not is_auto_training:
                    break
                mode_name = {"gf": "❤️ প্রেমিকা মোড", "friend": "👫 বন্ধু মোড", "wife": "💍 বউ মোড", "ai": "🤖 সাধারণ এআই"}[mode]

                dialogues = teacher.create_dialogues(mode=mode, count=6)
                test_q, gold_a = dialogues[0]
                student_resp = student.answer(test_q, mode=mode)

                yield f"data: {json.dumps({'type': 'quiz', 'cycle': cycle, 'mode': mode_name, 'question': test_q, 'student': student_resp, 'gold': gold_a}, ensure_ascii=False)}\n\n"

                mode_tag = {"gf": "[প্রেমিকা]", "friend": "[বন্ধু]", "wife": "[বউ]", "ai": "[এআই]"}[mode]
                for q, a in dialogues:
                    b = f"{mode_tag} প্রশ্ন: {q}\nউত্তর: {a}\n\n"
                    b = b.replace("ড়", "ড\u09bc").replace("ঢ়", "ঢ\u09bc").replace("য়", "য\u09bc")
                    new_blocks.append(b)

            if not is_auto_training:
                break

            with open("chat_data.txt", "a", encoding="utf-8") as f:
                for b in new_blocks:
                    f.write(b)

            # Fine-tune student
            text = open("chat_data.txt", encoding="utf-8").read()
            data = torch.tensor(encode(text), dtype=torch.long)
            n = int(0.9 * len(data))
            train_data = data[:n]

            steps = 60
            student.model.train()
            optimizer = torch.optim.AdamW(student.model.parameters(), lr=1.0e-4)

            for step in range(1, steps + 1):
                if not is_auto_training:
                    break
                ix = torch.randint(len(train_data) - block_size - 1, (24,))
                xb = torch.stack([train_data[i:i + block_size] for i in ix])
                yb = torch.stack([train_data[i + 1:i + block_size + 1] for i in ix])
                _, loss = student.model(xb.to(device), yb.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0)
                optimizer.step()

                if step % 20 == 0 or step == steps:
                    yield f"data: {json.dumps({'type': 'step', 'cycle': cycle, 'step': step, 'total': steps, 'loss': round(loss.item(), 4)}, ensure_ascii=False)}\n\n"

            if not is_auto_training:
                break

            # Save student weights & reload into server
            torch.save(student.model.state_dict(), MODEL_PATH)
            reload_model()
            yield f"data: {json.dumps({'type': 'cycle_done', 'cycle': cycle, 'text': f'🎉 সাইকেল #{cycle} সফল! মডেল লাইভ আপডেট হয়েছে।'}, ensure_ascii=False)}\n\n"
            time.sleep(1.0)

        yield f"data: {json.dumps({'type': 'stopped', 'text': '🛑 ট্রেইনিং সফলভাবে থামানো হয়েছে! সর্বশেষ মডেল সেভ রয়েছে।'}, ensure_ascii=False)}\n\n"

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/auto_train", methods=["POST"])
def trigger_auto_train():
    import threading
    import subprocess

    def run_trainer():
        try:
            print("\n🤖 Teacher-Student Auto-Train background thread চালু হয়েছে...")
            subprocess.run([sys.executable, "-u", "auto_trainer.py", "--cycles", "1", "--steps", "150"])
        except Exception as err:
            print("Auto-train worker error:", err)

    thread = threading.Thread(target=run_trainer, daemon=True)
    thread.start()
    return {"status": "started", "message": "শিক্ষক AI ছাত্র মডেলকে ট্রেইন করানো শুরু করেছে!"}


if __name__ == "__main__":
    port = 5000
    print(f"\n=======================================================")
    print(f"  ✨ Bangla Mini-GPT Web UI চালু হয়েছে!")
    print(f"  ব্রাউজারে ওপেন করুন: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
