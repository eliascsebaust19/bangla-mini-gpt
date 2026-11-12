"""
Autonomous Teacher-Student AI Auto-Trainer for Mini-GPT.

How it works:
1. Teacher AI (Gemini API or Built-in Synthetic Teacher) creates fresh, creative Q&A dialogues.
2. Student AI (Mini-GPT) attempts to answer the Teacher's questions.
3. Teacher inspects the student's output, provides gold-standard corrections, and gathers training data.
4. Auto-Trainer automatically fine-tunes Mini-GPT on the new batch and updates mini_gpt.pt!
5. Notifies the live web server to reload the newly upgraded brain.

Usage:
    python auto_trainer.py
    python auto_trainer.py --api-key YOUR_GEMINI_API_KEY --cycles 3
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import torch

from mini_gpt import MiniGPT, encode, decode, stoi, device, block_size

# Default paths
MODEL_PATH = "mini_gpt.pt"
DATA_PATH = "chat_data.txt"


class TeacherAI:
    """The master teacher that generates curriculum and corrects student."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            print("✨ Teacher AI: Gemini API সক্রিয় রয়েছে (Live Online Teacher)!")
        else:
            print("💡 Teacher AI: Built-in Synthetic Teacher সক্রিয় (অফলাইন মোড)।")

    def generate_with_gemini(self, prompt):
        """Calls Gemini API via standard urllib without extra packages."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 1000
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"Gemini API তে সাময়িক ত্রুটি ({e}), অফলাইন জেনারেটর ব্যবহার করা হচ্ছে।")
            return None

    def create_dialogues(self, mode="gf", count=15):
        """Generates structured dialogues for the given persona mode."""
        if self.api_key:
            prompt = (
                f"You are a dataset generator for training a tiny conversational Bengali model.\n"
                f"Generate exactly {count} short question-answer dialogues in persona: '{mode}' (either GF, Friend, Wife, or AI assistant).\n"
                f"Make half in colloquial Banglish (English letters) and half in Bengali script.\n"
                f"Format EACH pair strictly as:\n"
                f"Q: [question]\nA: [short answer]\n---\n"
                f"Only output the dialogues, nothing else."
            )
            raw = self.generate_with_gemini(prompt)
            if raw:
                pairs = []
                blocks = raw.split("---")
                for b in blocks:
                    q, a = None, None
                    for line in b.strip().splitlines():
                        if line.startswith("Q:"):
                            q = line[2:].strip()
                        elif line.startswith("A:"):
                            a = line[2:].strip()
                    if q and a:
                        pairs.append((q, a))
                if pairs:
                    return pairs

        # Fallback to Built-in Dynamic Curriculum Generator
        return self._builtin_generator(mode, count)

    def _builtin_generator(self, mode, count):
        """Rich dynamic template generator with strictly matched Q&A pairs."""
        if mode == "gf":
            pool = [
                ("amar mon kharap", "Ole amar baabu tar mon kharap keno? Kache esho, ami mon bhalo kore debo!"),
                ("আমার মন খারাপ।", "ওলে আমার বাবুটার মন খারাপ কেন? কি হয়েছে আমাকে বলো তো, আমি তোমার মন ভালো করে দেব!"),
                ("kemon acho?", "Ami bhalo achi jaanu! Tumi kemon acho bolo? Tomar kotha khub mone porchilo."),
                ("কেমন আছো জানু?", "আমি ভালো আছি জানু! তুমি কেমন আছো বলো? তোমার কথা খুব মনে পড়ছিল।"),
                ("kheyecho baabu?", "Haa ami kheyechi baabu! Tumi pet bhore bhat kheyecho to?"),
                ("খেয়েছ তুমি?", "হ্যাঁ আমি খেয়েছি বাবু! তুমি ঠিকমতো খেয়েছ তো?"),
                ("bhat khaicho?", "Haa ami bhat kheyechi baabu! Tumi shomoymoto kheye nao to."),
                ("ki koro jaan?", "Tomar kothai bhabchilam jaanu! Tumi ki korcho bolo to?"),
                ("কি করছো?", "তোমার কথাই ভাবছিলাম বাবু! তুমি কি করছো বলো তো?"),
                ("amake bhalobasho?", "Khub bhalobashi tomay! Tumi chara amar ar ke ache bolo?"),
                ("আমাকে ভালোবাসো?", "খুব ভালোবাসি তোমায়! তোমায় ছাড়া আমি আর কিছুই বুঝি না।"),
                ("রাগ করছো?", "না রে বাবু, একটুও রাগ করিনি! আমি কি তোমার ওপর রাগ করতে পারি কখনো?"),
            ]
        elif mode == "friend":
            pool = [
                ("kire mama", "Are mama! Shob chill, tor ki obostha bol?"),
                ("কেমন আছিস দোস্ত?", "আরে দোস্ত! আমি একদম জোস, তুই কেমন আছিস বল?"),
                ("kemon achis?", "Are dosto! Ami ekdom joss, tui kemon achis bol?"),
                ("ki khobor dosto?", "Shob darun dosto! Tui bol, ki choltese jibone?"),
                ("কি খবর?", "সব চিল মামা! তোর কি খবর বল? কোনো নতুন কাহিনী আছে নাকি?"),
                ("bhat khaicho?", "Haa re mama bhat kheyechi! Tui khaisosh to?"),
                ("chol ghure ashi", "Chol mama! Ekhoni ready hoye ja, kothay jabi bol?"),
                ("taka dhar dibi?", "Are bhai! Amar nijer pocket e to batash ghurtese! Tui পারলে আমাকে ধার দে!"),
                ("mon bhalo nei re", "Dhur mama! Eto pyara nis na to, chol চা খাই আর আড্ডা দিই!"),
                ("ki koros?", "Boshe boshe tor message er opekkha kortechilam dosto!"),
                ("কিরে মামা!", "আরে মামা! তুই কেমন আছিস বল?"),
            ]
        elif mode == "wife":
            pool = [
                ("কেমন আছো গো?", "আমি ভালো আছি গো। তোমার শরীর ঠিক আছে তো? কাজে বেশি চাপ নিও না।"),
                ("kemon acho?", "Ami bhalo achi go. Tomar shorir thik ache to? Kaje beshi chap nio na."),
                ("কখন ফিরবে বাসায়?", "তাড়াতাড়ি ফিরে এসো গো, দেরি করো না। আমি তোমার পছন্দের খাবার রান্না করে রাখছি।"),
                ("kokhon firbe bashay?", "Taratari fire esho go, deri koro na. Ami ranna kore rakhchi."),
                ("খেয়েছো তুমি?", "আমি খাব গো, আগে তুমি সময়মতো খেয়ে নাও তো। সবসময় খাওয়ার কথা মনে করিয়ে দিতে হয় তোমাকে!"),
                ("kheyecho tumi?", "Haa ami kheyechi go! Tumi office e thikmoto kheyecho to?"),
                ("amar cha lagbe", "Ekdom gorom gorom aada cha baniye dichi go, ektu opekkha koro."),
                ("আজকে দেরি হবে।", "আবার দেরি? প্রতিদিন এত দেরি করলে হবে? কাজ শেষ করে সাবধানে তাড়াতাড়ি বাড়ি ফিরবে কিন্তু!"),
                ("ajke ektu deri hobe", "Abar deri? Kintu shabdhane taratari bari firbe!"),
                ("tumi amake bhalobasho?", "Bhalobashi bolei to eto bokajhoka kori ar eto jotno nei go!"),
            ]
        else:  # AI Mode
            pool = [
                ("কেমন আছেন?", "আমি ভালো আছি, ধন্যবাদ! আপনাকে কীভাবে সাহায্য করতে পারি?"),
                ("kemon achen?", "Ami bhalo achi, dhonnobad! Apnake kivabe sahajjo korte pari?"),
                ("তুমি কে?", "আমি একটি বাংলা মিনি জিপিটি (Mini-GPT) কৃত্রিম বুদ্ধিমত্তা মডেল।"),
                ("tumi ke?", "Ami ekta Bangla Mini-GPT AI model, ami Bangla o Banglish duitatei kotha bolte pari."),
                ("tomar naam ki?", "Amar naam Mini-GPT, ami apnar shathe kotha bolte pari."),
                ("বাংলাদেশের রাজধানী কি?", "বাংলাদেশের রাজধানী হলো ঢাকা।"),
                ("Bangladesh er rajdhani ki?", "Bangladesh er rajdhani holo Dhaka."),
                ("computer ki?", "Computer holo ekta electronic device ja druto tottho process kore."),
                ("কম্পিউটার কি?", "কম্পিউটার হলো একটি ইলেকট্রনিক যন্ত্র যা অতি দ্রুত তথ্য প্রক্রিয়াকরণ করে।"),
                ("কৃত্রিম বুদ্ধিমত্তা কি?", "কৃত্রিম বুদ্ধিমত্তা বা এআই হলো এমন এক প্রযুক্তি যা মানুষের মতো চিন্তা ও সিদ্ধান্ত নিতে সক্ষম।"),
                ("krittim buddhimotta ki?", "Krittim buddhimotta ba AI holo emon technology ja manusher moto chinta korte pare."),
            ]

        # Return strictly matched pairs (no random mismatching)
        selected = []
        for _ in range(count):
            selected.append(random.choice(pool))
        return selected


class StudentAI:
    """The student Mini-GPT that learns and takes tests."""

    def __init__(self, model_path=MODEL_PATH):
        self.model = MiniGPT().to(device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def answer(self, prompt, mode="ai", max_tokens=50, temperature=0.4):
        mode_prefix = {
            "gf": "[প্রেমিকা] ",
            "friend": "[বন্ধু] ",
            "wife": "[বউ] ",
            "ai": "[এআই] "
        }.get(mode, "")

        formatted_prompt = f"{mode_prefix}প্রশ্ন: {prompt}\nউত্তর:"
        clean_chars = [c for c in formatted_prompt if c in stoi]
        if not clean_chars:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            idx = torch.tensor([encode("".join(clean_chars))], dtype=torch.long, device=device)

        accumulated = ""
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            char = decode([next_id.item()])
            if char == "\n" and len(accumulated.strip()) > 0:
                break
            accumulated += char

        return accumulated.strip()


class AutoTrainer:
    """Orchestrates Teacher-Student self-training loop."""

    def __init__(self, api_key=None):
        self.teacher = TeacherAI(api_key=api_key)
        self.student = StudentAI()

    def run_cycle(self, cycle_num=1, steps=250):
        print(f"\n=======================================================")
        print(f"  🚀 Auto-Training Cycle #{cycle_num} শুরু হচ্ছে!")
        print(f"=======================================================")

        modes = ["gf", "friend", "wife", "ai"]
        new_blocks = []

        # 1. Teacher generates curriculum and quizzes student
        for mode in modes:
            print(f"\n[শিক্ষক AI -> মোড: {mode.upper()}] নতুন পাঠ্যক্রম তৈরি করা হচ্ছে...")
            dialogues = self.teacher.create_dialogues(mode=mode, count=10)

            # Test student on the first question
            test_q, gold_a = dialogues[0]
            student_resp = self.student.answer(test_q, mode=mode)
            print(f"  ❓ শিক্ষক প্রশ্ন করলেন: '{test_q}'")
            print(f"  🎓 ছাত্রের উত্তর:      '{student_resp}'")
            print(f"  ✅ শিক্ষকের সঠিক উত্তর: '{gold_a}'")

            mode_tag = {
                "gf": "[প্রেমিকা]",
                "friend": "[বন্ধু]",
                "wife": "[বউ]",
                "ai": "[এআই]"
            }[mode]

            for q, a in dialogues:
                block = f"{mode_tag} প্রশ্ন: {q}\nউত্তর: {a}\n\n"
                # Normalize nukta
                block = block.replace("ড়", "ড\u09bc").replace("ঢ়", "ঢ\u09bc").replace("য়", "য\u09bc")
                new_blocks.append(block)

        # 2. Append new knowledge to dataset
        print(f"\n📚 সংগৃহীত মোট {len(new_blocks)} টি নতুন ডায়ালগ ডেটাসেটে যুক্ত হচ্ছে...")
        with open(DATA_PATH, "a", encoding="utf-8") as f:
            for b in new_blocks:
                f.write(b)

        # 3. Fine-tune student model
        print(f"\n⚡ ছাত্র Mini-GPT কে স্বয়ংক্রিয়ভাবে ট্রেইন করানো হচ্ছে ({steps} steps)...")
        text = open(DATA_PATH, encoding="utf-8").read()
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data, val_data = data[:n], data[n:]

        def get_batch():
            ix = torch.randint(len(train_data) - block_size - 1, (24,))
            x = torch.stack([train_data[i:i + block_size] for i in ix])
            y = torch.stack([train_data[i + 1:i + block_size + 1] for i in ix])
            return x.to(device), y.to(device)

        self.student.model.train()
        optimizer = torch.optim.AdamW(self.student.model.parameters(), lr=1.0e-4)

        for step in range(1, steps + 1):
            xb, yb = get_batch()
            _, loss = self.student.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0 or step == steps:
                print(f"  Step {step:>3}/{steps} — Loss: {loss.item():.4f}")

        # Save student weights
        torch.save(self.student.model.state_dict(), MODEL_PATH)
        self.student.model.eval()
        print(f"\n🎉 সাইকেল #{cycle_num} সম্পন্ন! নতুন জ্ঞান mini_gpt.pt এ সেভ হয়েছে।")

        # 4. Notify live server to reload
        self.notify_server_reload()

    def notify_server_reload(self):
        """Notifies local Flask server to reload weights in-memory instantly."""
        try:
            req = urllib.request.Request("http://127.0.0.1:5000/api/reload_model", data=b"{}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                print("🔄 লাইভ ওয়েব সার্ভারে নতুন মডেল রিলোড সফল হয়েছে!")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Auto-Train Mini-GPT with Teacher AI")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API Key (optional)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of training cycles")
    parser.add_argument("--steps", type=int, default=200, help="Steps per cycle")
    args = parser.parse_args()

    trainer = AutoTrainer(api_key=args.api_key)
    for c in range(1, args.cycles + 1):
        trainer.run_cycle(cycle_num=c, steps=args.steps)
        if c < args.cycles:
            time.sleep(2)


if __name__ == "__main__":
    main()
