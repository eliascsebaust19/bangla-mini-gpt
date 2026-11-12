"""
Mini GPT Inference Script.
Loads pre-trained mini_gpt.pt checkpoint and generates text from a prompt.

Usage:
    python generate.py
    python generate.py "বাংলাদেশ একটি সুন্দর"
"""
import os
import sys
import torch
from mini_gpt import MiniGPT, encode, decode, stoi, device, block_size


def load_model(weights_path="mini_gpt.pt"):
    if not os.path.exists(weights_path):
        print(f"Error: '{weights_path}' khuje pawa jayni! Age mini_gpt.py diye model train korun.")
        sys.exit(1)
    model = MiniGPT().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate_text(model, prompt="", max_new_tokens=200, temperature=0.6, stream=True):
    # Conversational format
    if not prompt.startswith("প্রশ্ন:"):
        formatted_prompt = f"প্রশ্ন: {prompt}\nউত্তর:"
    else:
        formatted_prompt = prompt

    clean_chars = [c for c in formatted_prompt if c in stoi]
    if not clean_chars:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        idx = torch.tensor([encode("".join(clean_chars))], dtype=torch.long, device=device)

    accumulated = ""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        char = decode([next_id.item()])
        if char == "\n" and len(accumulated.strip()) > 0:
            break

        accumulated += char
        if stream:
            print(char, end="", flush=True)

    if stream:
        print("\n" + "-" * 50)
    return accumulated.strip()


def main():
    print("Model load kora hocche...")
    model = load_model()
    print("Model load hoyeche! (Device:", device + ")")

    # Jodi command line theke prompt dewa thake
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"\n--- Prompt: '{prompt}' ---")
        generate_text(model, prompt=prompt, max_new_tokens=300, temperature=0.7)
        return

    # Interactive mode
    print("\n" + "=" * 50)
    print("Mini GPT Bangla Text Generator")
    print("Je kono proshno ba shuru-te kichu likhun (ber hote 'exit' likhun)")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("\nআপনার প্রম্পট লিখুন: ").strip()
            if prompt.lower() in ("exit", "quit", "q"):
                print("বিদায়!")
                break
            if not prompt:
                prompt = "বাংলাদেশ"
                print(f"ডিফল্ট প্রম্পট ব্যবহার করা হচ্ছে: '{prompt}'")

            print("\n--- আউটপুট ---")
            generate_text(model, prompt=prompt, max_new_tokens=300, temperature=0.7)
        except (KeyboardInterrupt, EOFError):
            print("\nবিদায়!")
            break


if __name__ == "__main__":
    main()
