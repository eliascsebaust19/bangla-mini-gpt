"""
Instruction / Dialogue Fine-Tuning for Mini-GPT.
Fine-tunes the base model on conversational 'প্রশ্ন: ... \nউত্তর: ...' data.

Usage:
    python finetune_chat.py
"""
import os
import torch
import torch.nn.functional as F
from mini_gpt import MiniGPT, encode, decode, block_size, device

# Config for fine-tuning
batch_size = 28
max_iters = 800
eval_interval = 200
eval_iters = 25
lr = 1.0e-4

# 1. Load Conversational Dataset
text = open("chat_data.txt", encoding="utf-8").read()
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    model = MiniGPT().to(device)

    # 2. Load pre-trained weights if available
    base_model_path = "mini_gpt.pt"
    if os.path.exists(base_model_path):
        print(f"Pre-trained weights '{base_model_path}' load kora hocche...")
        state_dict = torch.load(base_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights load shompurno!")
    else:
        print("Base weights pawa jayনি, shuru theke train hobe.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nConversational Fine-Tuning shuru hocche ({max_iters} steps, lr={lr})...")
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it:>4}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if it > 0:
                torch.save(model.state_dict(), "mini_gpt.pt")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Save fine-tuned checkpoint
    output_path = "mini_gpt.pt"
    # Backup original first
    if os.path.exists(base_model_path) and not os.path.exists("mini_gpt_base.pt"):
        import shutil
        shutil.copyfile(base_model_path, "mini_gpt_base.pt")
        print("Original base model 'mini_gpt_base.pt' te backup kora hoyeche.")

    torch.save(model.state_dict(), output_path)
    print(f"\nFine-tuned model '{output_path}' te save hoyeche! 🎉")

    # Quick test across personas & colloquial queries
    model.eval()
    test_prompts = [
        "[প্রেমিকা] প্রশ্ন: jannn\nউত্তর:",
        "[প্রেমিকা] প্রশ্ন: bhat khaicho?\nউত্তর:",
        "[প্রেমিকা] প্রশ্ন: ki koro\nউত্তর:",
        "[প্রেমিকা] প্রশ্ন: bhalo\nউত্তর:",
        "[বন্ধু] প্রশ্ন: kire mama\nউত্তর:",
        "[বন্ধু] প্রশ্ন: bhat khaicho?\nউত্তর:",
        "[বউ] প্রশ্ন: kokhon firbe bashay?\nউত্তর:",
        "[এআই] প্রশ্ন: Bangladesh er rajdhani ki?\nউত্তর:",
    ]
    print("\n" + "=" * 50)
    print("Multi-Persona & Banglish Test Results:")
    print("=" * 50)
    for tp in test_prompts:
        print(f"\n--- {tp.splitlines()[0]} ---")
        context = torch.tensor([encode(tp)], dtype=torch.long, device=device)
        out_ids = model.generate(context, max_new_tokens=80, temperature=0.4)[0].tolist()
        text = decode(out_ids)
        ans = text[len(tp):].split("\n")[0].strip()
        print("উত্তর:", ans)


if __name__ == "__main__":
    main()
