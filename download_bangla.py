"""
Bangla Wikipedia theke article namiye input.txt banay.
Kono extra package lagbe na, shudhu Python er built-in library.

Usage:
    python download_bangla.py
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

TARGET_CHARS = 1_000_000   # koto character jogar korbe (1M ~ 3 MB file)
MIN_ARTICLE_CHARS = 500    # er cheye chhoto article bad
OUTPUT_FILE = "input.txt"
API = "https://bn.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "MiniGPT-StudentProject/1.0 (Bengali NLP learning project; contact: student@example.org)"
}

# Shuru te boro ebong shundor kichu article diye shuru korle druto bhalo text pawa jay
SEED_ARTICLES = [
    "বাংলাদেশ", "ঢাকা", "বাংলা ভাষা", "রবীন্দ্রনাথ ঠাকুর", "কাজী নজরুল ইসলাম",
    "বাংলাদেশের স্বাধীনতা যুদ্ধ", "বঙ্গবন্ধু শেখ মুজিবুর রহমান", "সুন্দরবন",
    "পদ্মা নদী", "হিমালয় পর্বতমালা", "সৌরজগৎ", "পৃথিবী", "সূর্য",
    "পদার্থবিজ্ঞান", "রসায়ন", "জীববিজ্ঞান", "ইতিহাস", "কম্পিউটার", "কৃত্রিম বুদ্ধিমত্তা"
]


def call_api(params, max_retries=5):
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 4 * (attempt + 1)
                print(f"\nWikipedia Rate limit (429), {wait_time}s opekkha korchi...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    return None


def random_titles(count=20):
    data = call_api({
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": count,
    })
    if not data or "query" not in data:
        return []
    return [item["title"] for item in data["query"]["random"]]


def fetch_text(title):
    data = call_api({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
    })
    if not data or "query" not in data:
        return ""
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def clean(text):
    text = re.sub(r"={2,}.*?={2,}", "", text)    # section heading soriye daw
    text = re.sub(r"\n{3,}", "\n\n", text)       # extra faka line komao
    return text.strip()


def main():
    seen = set()
    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # 1. First fetch seed articles for quick, high quality text
        print("Boro article gulo download kora shuru hocche...")
        for title in SEED_ARTICLES:
            seen.add(title)
            try:
                text = clean(fetch_text(title))
                if len(text) >= MIN_ARTICLE_CHARS:
                    out.write(text + "\n\n")
                    out.flush()
                    total += len(text)
                    print(f"{total:>9,} / {TARGET_CHARS:,}  {title}")
                time.sleep(0.8)
            except Exception as e:
                print("Skip:", title, e)
                time.sleep(2)
            if total >= TARGET_CHARS:
                break

        # 2. Then fill the rest with random articles
        print("\nRandom article download shuru hocche...")
        while total < TARGET_CHARS:
            try:
                titles = random_titles()
            except Exception as e:
                print("Title ante somossa, abar chesta korchi:", e)
                time.sleep(4)
                continue

            for title in titles:
                if title in seen:
                    continue
                seen.add(title)
                try:
                    text = clean(fetch_text(title))
                except Exception as e:
                    print("Skip:", title, e)
                    time.sleep(2)
                    continue

                if len(text) < MIN_ARTICLE_CHARS:
                    continue

                out.write(text + "\n\n")
                out.flush()
                total += len(text)
                print(f"{total:>9,} / {TARGET_CHARS:,}  {title}")
                if total >= TARGET_CHARS:
                    break
                time.sleep(0.8)

    print(f"\nShesh! {OUTPUT_FILE} e {total:,} character save hoyeche.")


if __name__ == "__main__":
    main()
