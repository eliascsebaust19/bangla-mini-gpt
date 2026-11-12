# 🇧🇩 Bangla Mini-GPT | Futuristic Emerald Bengali AI Assistant

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-00D6A0?style=for-the-badge&logo=vercel&logoColor=black)](https://bangla-mini-gpt.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.11+-00A878?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/Inference-Pure_NumPy-071C17?style=for-the-badge&logo=numpy&logoColor=00D6A0)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/Training-PyTorch-FF304F?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

বাংলা ভাষার জন্য স্ক্র্যাচ থেকে তৈরি একটি ক্যারেক্টার-লেভেল মিনি ল্যাঙ্গুয়েজ মডেল (Transformer Decoder) যা খাঁটি বাংলা এবং Banglish (রোমান হরফে বাংলা) উভয় ভাষায় সাবলীলভাবে চ্যাট করতে পারে। সম্পূর্ণ নতুন ফিউচারিস্টিক ডার্ক এমারেল্ড (Futuristic Emerald) ও সাইবারপাঙ্ক হলোোগ্রাফিক ইন্টারফেসে সাজানো।

🌐 **লাইভ ওয়েবসাইট:** [https://bangla-mini-gpt.vercel.app](https://bangla-mini-gpt.vercel.app)

---

## ✨ প্রধান বৈশিষ্ট্যসমূহ (Key Features)

- 💎 **ফিউচারিস্টিক এমারেল্ড ও হলোোগ্রাফিক UI (Next-Gen UI):**
  - ডার্ক এমারেল্ড প্যালেট (`#071C17`, `#0A241C`, `#0B211B`, `#00D6A0`) ও হলোোগ্রাফিক লাল অ্যাকসেন্ট (`#FF304F`)।
  - ডেস্কটপে স্মুথ হলোোগ্রাফিক কার্সার ও মাইক্রো-ইন্টারঅ্যাকশন।
  - মোবাইল ও ডেস্কটপে ফুললি রেসপনসিভ অফ-ক্যানভাস ড্রয়ার ও নিয়ন টগল বাটন।
  - নিচে স্থায়ীভাবে পিন্ড ডকড চ্যাট ক্যাপসুল (Docked Prompt Capsule)।
- 🎭 **৪টি অনন্য ব্যক্তিত্ব মোড (4 Persona Modes):**
  - 🤖 **সাধারণ এআই (General AI):** তথ্যমূলক, জ্ঞানমূলক ও সহযোগিতামূলক উত্তর
  - 👫 **বন্ধু মোড (Friend Mode):** আড্ডাবাজি, চিল ও দোস্তি কথপোকথন
  - ❤️ **প্রেমিকা মোড (GF Mode):** মিষ্টি, কেয়ারিং ও রোমান্টিক কথোপকথন
  - 💍 **বউ মোড (Wife Mode):** খুনসুটি, যত্ন ও সংসার বিষয়ক আলাপচারিতা
- 🔤 **বাংলা ও Banglish দ্বিমুখী সাপোর্ট:** বাংলা লিপির পাশাপাশি রোমান হরফে লেখা বাংলা (যেমন: "ki koro", "bhalo achi", "kemon acho") অনায়াসে বোঝে ও উত্তর দেয়।
- ⚡ **জিরো-ক্লাউড কস্ট Serverless Engine:** Vercel-এর ২৫০MB মেমোরি লিমিট বাইপাস করার জন্য সম্পূর্ণ **Pure NumPy Transformer** দিয়ে সার্ভারলেস ইনফারেন্স ইঞ্জিন তৈরি করা হয়েছে, ফলে কোনো ডেডিকেটেড GPU/সার্ভার ছাড়াই দ্রুত রেসপন্স দেয়।
- 🎓 **স্বয়ংক্রিয় শিক্ষক-ছাত্র AI অটো-ট্রেইনিং (Teacher-Student Auto-Trainer):** মডেল নিজেই প্রশ্ন-উত্তর তৈরি করে নিজের ওয়েট অপ্টিমাইজ করার ক্ষমতা রাখে এবং লাইভ ফ্রন্টএন্ড মডালে রিয়েল-টাইমে কুইজ ও লস দেখা যায়।
- 🎛️ **রিয়েল-টাইম মডেল প্যারামিটার কন্ট্রোল:** টেম্পারেচার (০.১ - ১.৫) এবং ম্যাক্স টোকেন স্লাইডার দিয়ে উত্তরের সৃজনশীলতা নিয়ন্ত্রণ।
- 🔊 **Text-to-Speech (TTS):** প্রতিটি উত্তরের খাঁটি বাংলা উচ্চারণ শোনার সুবিধা।
- 📋 **চ্যাট ডাউনলোড ও কপি:** এক ক্লিকে সম্পূর্ণ কথোপকথন টেক্সট ফাইলে এক্সপোর্ট বা ক্লিপবোর্ডে কপি করার সুবিধা।

---

## 🏗️ মডেল আর্কিটেকচার (Model Architecture)

- **মডেল টাইপ:** Decoder-only Transformer (Character-level GPT)
- **মোট প্যারামিটার:** ~১.৯৬ মিলিয়ন (~1.96M Parameters)
- **লেয়ার সংখ্যা (`n_layer`):** ৪
- **অ্যাটেনশন হেড (`n_head`):** ৬
- **এমবেডিং ডাইমেনশন (`n_embd`):** ১৯২
- **ব্লক সাইজ / কনটেক্সট উইন্ডো (`block_size`):** ১২৮
- **ভোকাবুলারি সাইজ:** ৪২৪টি ইউনিক ক্যারেক্টার (বাংলা ইউনিকোড + নুকতা + রোমান বর্ণমালা + পাঙ্কচুয়েশন)

---

## 📁 ফাইল ও প্রজেক্ট স্ট্রাকচার (Repository Structure)

```text
bangla-mini-gpt/
├── api/
│   └── index.py            # Vercel Serverless এন্ট্রি পয়েন্ট (Pure NumPy Inference + Auto-Train)
├── templates/
│   └── index.html          # ফিউচারিস্টিক এমারেল্ড Tailwind UI (রেসপনসিভ ড্রয়ার, ক্যাপসুল)
├── app.py                  # লোকাল Flask ফুল সার্ভার (PyTorch / CUDA এক্সিলারেশন সহ)
├── mini_gpt.py             # পাইটর্চ ট্র্যান্সফরমার কোর মডেল আর্কিটেকচার
├── auto_trainer.py         # স্বয়ংক্রিয় Teacher-Student ট্রেইনিং ইঞ্জিন
├── create_chat_data.py     # মাল্টি-পার্সোনা বাংলা ও Banglish ডায়ালগ জেনারেটর
├── finetune_chat.py        # কনভারসেশনাল ফাইন-টিউনিং স্ক্রিপ্ট
├── download_bangla.py      # উইকিপিডিয়া স্ক্র্যাপার স্ক্রিপ্ট
├── mini_gpt.npz            # এক্সপোর্টেড পিওর NumPy মডেল ওয়েটস (৭.২৮ MB)
├── vocab.json              # ক্যারেক্টার টোকেনাইজার ভোকাবুলারি ম্যাপ
├── requirements.txt        # Vercel সার্ভারলেস ডিপেন্ডেন্সি (Flask + NumPy)
├── requirements-train.txt  # লোকাল ট্রেইনিং ডিপেন্ডেন্সি (PyTorch + Torchvision)
├── vercel.json             # Vercel সার্ভারলেস কনফিগ
└── README.md               # সম্পূর্ণ প্রজেক্ট ডকুমেন্টেশন
```

---

## 🚀 লোকাল সেটআপ ও রান করার নিয়ম (Local Setup)

### ১. রিপোজিটরি ক্লোন করুন
```bash
git clone https://github.com/eliascsebaust19/bangla-mini-gpt.git
cd bangla-mini-gpt
```

### ২. ভার্চুয়াল এনভায়রনমেন্ট তৈরি ও ডিপেন্ডেন্সি ইনস্টল করুন
```bash
python -m venv .venv

# Windows-এ:
.venv\Scripts\activate

# Linux / Mac-এ:
source .venv/bin/activate

pip install -r requirements.txt
```

### ৩. ওয়েব অ্যাপ চালু করুন
```bash
# Pure NumPy সার্ভারলেস মোডে টেস্ট করতে:
python api/index.py

# অথবা PyTorch মডেল চালাতে (requirements-train.txt ইনস্টল থাকলে):
python app.py
```
ব্রাউজারে ওপেন করুন: `http://127.0.0.1:5000`

---

## 🛠️ মডেল ট্রেইনিং ও ফাইন-টিউনিং (Training from Scratch)

লোকাল কম্পিউটারে স্ক্র্যাচ থেকে মডেল ট্রেইনিং করতে চাইলে PyTorch ডিপেন্ডেন্সি ইনস্টল করুন:
```bash
pip install -r requirements-train.txt
```

1. **উইকিপিডিয়া স্ক্র্যাপিং:**
   ```bash
   python download_bangla.py
   ```
2. **বেস প্রি-ট্রেইনিং:**
   ```bash
   python mini_gpt.py
   ```
3. **কনভারসেশনাল ডেটাসেট তৈরি ও ফাইন-টিউনিং:**
   ```bash
   python create_chat_data.py
   python finetune_chat.py
   ```
4. **NumPy ওজনে কনভার্ট (Vercel-এর জন্য):**
   ```python
   # PyTorch .pt থেকে NumPy .npz এক্সপোর্ট:
   import torch, numpy as np
   weights = torch.load("mini_gpt.pt", map_location="cpu")
   np.savez_compressed("mini_gpt.npz", **{k: v.numpy() for k, v in weights.items()})
   ```

---

## 🌐 Vercel-এ ডিপ্লয়মেন্ট (Vercel Serverless Deployment)

মডেলটি যাতে Vercel-এ কোনো অতিরিক্ত সার্ভার খরচ ছাড়াই চলতে পারে, তার জন্য PyTorch ওয়েটসকে `mini_gpt.npz`-এ রূপান্তর করে `api/index.py`-তে পিওর NumPy দিয়ে ইনফারেন্স রান করানো হয়।

```bash
vercel --prod --yes
```

---

## 📄 লাইসেন্স
MIT License. যে কেউ শিক্ষা, গবেষণা ও ডেভেলপমেন্ট কাজে মুক্তভাবে ব্যবহার করতে পারবেন।
