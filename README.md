# 🇧🇩 Bangla Mini-GPT | Gemini-Style Bengali AI Assistant

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://bangla-mini-gpt.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/Inference-Pure_NumPy-013243?style=for-the-badge&logo=numpy)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/Training-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)

বাংলা ভাষার জন্য স্ক্র্যাচ থেকে তৈরি একটি ক্যারেক্টার-লেভেল মিনি ল্যাঙ্গুয়েজ মডেল (Transformer Decoder) যা খাঁটি বাংলা এবং Banglish (রোমান হরফে বাংলা) উভয় ভাষায় সাবলীলভাবে চ্যাট করতে পারে।

🌐 **লাইভ ওয়েবসাইট:** [https://bangla-mini-gpt.vercel.app](https://bangla-mini-gpt.vercel.app)

---

## ✨ প্রধান বৈশিষ্ট্যসমূহ (Key Features)

- 🎭 **৪টি অনন্য ব্যক্তিত্ব মোড (4 Persona Modes):**
  - ❤️ **প্রেমিকা মোড (GF Mode):** মিষ্টি রোমান্টিক কথোপকথন
  - 👫 **বন্ধু মোড (Friend Mode):** আড্ডাবাজি ও চিল কথপোকথন
  - 💍 **বউ মোড (Wife Mode):** খুনসুটি ও সংসার বিষয়ক আলাপচারিতা
  - 🤖 **সাধারণ এআই (General AI):** তথ্যমূলক ও সহায়তামূলক উত্তর
- 🔤 **বাংলা ও Banglish দ্বিমুখী সাপোর্ট:** বাংলা লিপির পাশাপাশি রোমান হরফে লেখা বাংলা (যেমন: "ki koro", "bhalo achi", "kemon acho") অনায়াসে বোঝে ও উত্তর দেয়।
- ⚡ **জিরো-ক্লাউড কস্ট Serverless Engine:** Vercel-এর ২৫০MB মেমোরি লিমিট বাইপাস করার জন্য সম্পূর্ণ **Pure NumPy Transformer** দিয়ে সার্ভারলেস ইনফারেন্স ইঞ্জিন তৈরি করা হয়েছে, ফলে কোনো ডেডিকেটেড GPU/সার্ভার ছাড়াই ইনস্ট্যান্ট রেসপন্স দেয়।
- 🎓 **স্বয়ংক্রিয় শিক্ষক-ছাত্র AI অটো-ট্রেইনিং:** মডেল নিজেই প্রশ্ন-উত্তর তৈরি করে নিজের ওয়েট আপডেট করার ক্ষমতা রাখে এবং এর লাইভ প্রসেস ফ্রন্টএন্ডে সরাসরি দেখা যায়।
- 📱 **ফুললি মোবাইল রেসপনসিভ UI:** গুগল জেমিনি ও চ্যাটজিপিটি স্টাইলের ডার্ক/লাইট মোড, স্লাইডার সেটিংস, এবং স্মুথ ড্রয়ার নেভিগেশন।
- 🔊 **Text-to-Speech (TTS):** প্রতিটি উত্তরের খাঁটি বাংলা উচ্চারণ শোনার সুবিধা।

---

## 🏗️ মডেল আর্কিটেকচার (Model Architecture)

- **মডেল টাইপ:** Decoder-only Transformer (Character-level GPT)
- **মোট প্যারামিটার:** ~১.৯৬ মিলিয়ন (~1.96M Parameters)
- **লেয়ার সংখ্যা (`n_layer`):** ৪
- **অ্যাটেনশন হেড (`n_head`):** ৬
- **এমবেডিং ডাইমেনশন (`n_embd`):** ১৯২
- **ব্লক সাইজ / কনটেক্সট উইন্ডো (`block_size`):** ১২৮
- **ভোকাবুলারি সাইজ:** ৪২৪টি ইউনিক ক্যারেক্টার (বাংলা ইউনিকোড + নুকতা + রোমান বর্ণমালা + পাঙ্কচুয়েশন)

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
python app.py
```
ব্রাউজারে ওপেন করুন: `http://127.0.0.1:5000`

---

## 🛠️ মডেল ট্রেইনিং ও ফাইন-টিউনিং (Training from Scratch)

লোকাল কম্পিউটারে মডেল ট্রেইনিং করতে চাইলে PyTorch ডিপেন্ডেন্সি ইনস্টল করুন:
```bash
pip install -r requirements-train.txt
```

1. **উইকিপিডিয়া স্ক্র্যাপিং:**
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

---

## 🌐 Vercel-এ ডিপ্লয়মেন্ট (Vercel Serverless Deployment)

মডেলটি যাতে Vercel-এ কোনো অতিরিক্ত সার্ভার খরচ ছাড়াই চলতে পারে, তার জন্য PyTorch ওয়েটসকে `mini_gpt.npz`-এ রূপান্তর করে `api/index.py`-তে পিওর NumPy দিয়ে ইনফারেন্স রান করানো হয়।

```bash
vercel --name bangla-mini-gpt --yes
```

---

## 📄 লাইসেন্স
MIT License. যে কেউ শিক্ষা, গবেষণা ও ডেভেলপমেন্ট কাজে ব্যবহার করতে পারবেন।
