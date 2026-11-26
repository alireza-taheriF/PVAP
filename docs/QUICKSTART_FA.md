# راه‌اندازی سریع: تماس با یک لینک، روی مک

این راهنما PVAP را روی مک راه می‌اندازد تا یک نفر با باز کردن یک لینک روی گوشی‌اش با ایجنت حرف بزند. ایجنت در لحظه جواب می‌دهد، با صدای فارسیِ احساسی Ava حرف می‌زند، و بعد از هر تماس چیزهای تازه یاد می‌گیرد.

## ۱. نصب (یک بار)

```bash
brew install python@3.12 ollama cloudflared
git clone https://github.com/alireza-taheriF/PVAP && cd PVAP
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ava,asr]"

python -m ava_voice.tts.models fetch                  # تبدیل نوشتار به آوا + صدای پیش‌فرض
mkdir -p ~/models && cd ~/models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2
tar xjf sherpa-onnx-whisper-turbo.tar.bz2 && cd -

ollama pull aya-expanse:8b                            # مدل زبانی لوکال با فارسیِ خوب (حدود ۵ گیگ)
```

## ۲. تنظیم

```bash
cp pvap.example.toml pvap.toml
cp personas/example.toml personas/wife.toml
```

در `personas/wife.toml` این‌ها را پر کن:

- `[caller] id = "wife"`
- اسم طرف مقابل
- چند چیزی که ایجنت باید بداند (`facts`)
- `reveal`، که تعیین می‌کند ایجنت کی بگوید هوش مصنوعی است:
  - `"end"`: موقع خداحافظی (پیش‌فرض، مناسب سورپرایز)
  - `"turns:3"`: بعد از سه رفت‌وبرگشت
  - `"start"`: همان اول تماس

در `pvap.toml` مسیر `whisper_dir` را `~/models/sherpa-onnx-whisper-turbo` بگذار. اگر خواستی سراغ مدل ابری بروی، فقط `provider = "cloud"` را عوض کن و کلید API را در متغیر محیطی بگذار. فایل‌های `pvap.toml` و `personas/*.toml` وارد گیت نمی‌شوند.

## ۳. اول خودت تست کن

```bash
python -m pvap.web
```

در کروم یا سافاری مک این آدرس را باز کن: `http://localhost:8443/?who=wife&debug=1`. با `debug=1` متن گفت‌وگو هم روی صفحه دیده می‌شود. روی «پاسخ» بزن و حرف بزن.

## ۴. لینک برای گوشی

میکروفون گوشی فقط روی HTTPS کار می‌کند. ساده‌ترین راه یک تونل است. در ترمینال دیگری بزن:

```bash
cloudflared tunnel --url http://localhost:8443
```

آدرس `https://….trycloudflare.com` را که می‌دهد، با `?who=wife` در انتهایش بفرست. صفحه شبیه یک تماس ورودی است و طرف فقط روی «پاسخ» می‌زند.

اگر هر دو روی یک وای‌فای هستید، می‌توانی بدون تونل هم این کار را بکنی:

```bash
./tools/make_cert.sh
python -m pvap.web --cert cert.pem --key key.pem
```

بعد آدرس `https://<آی‌پی مک>:8443/?who=wife` را باز کن. گوشی یک بار هشدار امنیتی می‌دهد که باید ردش کنی.

نکته: اگر طرف مقابل با هندزفری حرف بزند، حذف اکو بهتر کار می‌کند. بدون هندزفری هم سیستم صدای خودِ ایجنت را از صدای او تشخیص می‌دهد.

## ۵. یادگیری از هر تماس (خودکار)

بعد از قطع شدن تماس، سه کار در پس‌زمینه انجام می‌شود:

- **متن کامل تماس** در `~/.pvap/memory/calls/wife/` ذخیره می‌شود.
- **حافظه:** مدل زبانی واقعیت‌های ماندگار تماس را بیرون می‌کشد و در `~/.pvap/memory/wife.json` نگه می‌دارد، مثل برنامه‌ها، علاقه‌ها و چیزهایی که باید پیگیری شوند. تماس بعدی این‌ها را می‌داند. فایل را هر وقت خواستی می‌توانی ویرایش یا پاک کنی.
- **تلفظ:** Ava جمله‌هایی را که در تماس گفته دوباره می‌گوید و با Whisper گوش می‌دهد. هر کلمه‌ای که درست شنیده نشود، با چند تلفظ دیگر امتحان می‌شود. تلفظی که واضحاً بهتر شنیده شود در `~/.config/ava-voice/learned_lexicon.tsv` ذخیره می‌شود. تلفظ‌هایی که خودت در `~/.config/ava-voice/lexicon.tsv` می‌نویسی همیشه بر تلفظ‌های یادگرفته‌شده اولویت دارند.

## ۶. صدای خودت

راهنمای کامل در `clone/TRAINING.md` در Ava Voice است. خلاصه‌اش:

1. حدود ۱۵ دقیقه ضبط با `python -m ava_voice.clone.record --voice ali`
2. ساخت دیتاست
3. آموزش روی GPU رایگان Colab، حدود ۱ تا ۲ ساعت
4. `python -m ava_voice.clone.install`

بعد در `pvap.toml` این‌ها را تنظیم کن:

```toml
voice_dir = "~/.cache/ava-voice/voice-ali"
speed = 1.0
pitch_semitones = 0
```

## سرعت و صدا

اگر صدا هنوز کند یا بم به نظرت رسید، در `[voice]` مقدار `speed` (مثلاً ۱.۲۵) و `pitch_semitones` (مثلاً ۳) را بالاتر ببر. شدت احساس با `intensity` تنظیم می‌شود.
