# SIM Entegrasyonu — IRONSIGHT Veri Kaynakları

> Bu doküman iki iş tanımlar:
> 1. SIM'in mevcut RSS sistemine eklenecek yeni feed URL'leri
> 2. SIM'e yeni eklenecek Telegram kanal scraping mekanizması

---

## 1. Mevcut RSS Sistemine Eklenecek Feed'ler

SIM'in RSS altyapısı (`pass_a_ingest.py`) zaten çalışıyor. Aşağıdaki URL'ler `config/settings.json` → `sources.static_feeds` array'ine eklenmelidir.

### İsrail Medyası
```
https://www.timesofisrael.com/feed/
https://www.jpost.com/rss/rssfeedsfrontpage.aspx
https://www.ynetnews.com/Integration/StoryRss2.xml
https://rcs.mako.co.il/rss/news-military.xml
https://rss.walla.co.il/feed/22
https://www.haaretz.com/srv/haaretz-latest-headlines
https://www.haaretz.com/srv/middle-east-news-rss
```

### İran Medyası
```
https://www.presstv.ir/rss.xml
https://www.presstv.ir/rss/rss-102.xml
https://www.presstv.ir/rss/rss-101.xml
```

### Uluslararası / Bölgesel
```
https://feeds.bbci.co.uk/news/world/middle_east/rss.xml
https://www.aljazeera.com/xml/rss/all.xml
https://feeds.reuters.com/Reuters/worldNews
https://www.thenationalnews.com/arc/outboundfeeds/rss/?outputType=xml
http://rss.cnn.com/rss/edition_meast.rss
https://moxie.foxnews.com/google-publisher/world.xml
https://feeds.content.dowjones.io/public/rss/RSSWorldNews
https://rss.nytimes.com/services/xml/rss/nyt/MiddleEast.xml
```

### Savunma / Askeri
```
https://breakingdefense.com/feed/
https://www.longwarjournal.org/feed
https://www.militarytimes.com/arc/outboundfeeds/rss/?outputType=xml
https://warontherocks.com/feed/
https://www.centcom.mil/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=808&max=20
https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=945&max=10
```

### Bağımsız
```
https://www.dropsitenews.com/feed
```

### Google News Arama Sorguları
```
https://news.google.com/rss/search?q=Iran+Israel+war+military&hl=en-US&gl=US&ceid=US:en
https://news.google.com/rss/search?q=Iran+missile+strike+drone&hl=en-US&gl=US&ceid=US:en
https://news.google.com/rss/search?q=%22Strait+of+Hormuz%22+OR+%22Red+Sea%22+military&hl=en-US&gl=US&ceid=US:en
```

---

## 2. Yeni Mekanizma: Telegram Kanal Scraping

SIM şu anda Telegram'dan veri çekmiyor. Aşağıda sıfırdan eklenecek mekanizma tanımlanmıştır.

### Nasıl Çalışır

Telegram'ın public embed sayfası scrape edilir. Bot API veya API key gerekmez.

Her post şu URL'den HTML olarak çekilir:
```
GET https://t.me/{kanal_adi}/{post_id}?embed=1&mode=tme
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
Timeout: 5 saniye
```

### HTML'den Veri Çıkarma

Mesaj metni şu regex ile alınır:
```regex
<div class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>
```

Tarih şu regex ile alınır:
```regex
<time[^>]*datetime="([^"]+)"
```

Çıkarılan text temizlenir:
- `<br>` → boşluk
- Tüm HTML tag'leri kaldırılır
- HTML entity'ler decode edilir (`&amp;` → `&`, `&lt;` → `<`, `&gt;` → `>`, `&quot;` → `"`, `&#39;` → `'`)
- Fazla boşluklar tek boşluğa indirilir

### Non-Latin Metin Çevirisi

İbranice, Arapça veya Farsça karakter tespiti:
```regex
[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]
```

Eşleşme varsa, ücretsiz Google Translate endpoint'i çağrılır (API key gerekmez):
```
GET https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={url_encoded_text}
Timeout: 3 saniye
```

Yanıt JSON array'dir. Çevrilmiş metin: `data[0]` içindeki her parçanın `[0]` indexi birleştirilir.

### Son Post ID'sini Bulma (Binary Search)

Telegram post ID'leri ardışık tam sayılardır ama kanalın toplam post sayısı bilinmez. İlk çağrıda binary search yapılır:

1. Probe noktalarını dene: `[500, 5000, 15000, 30000, 50000, 80000, 120000, 180000]`
   - Post varsa → `low = probe`, sonraki probe'a geç
   - Post yoksa → `high = probe`, dur
2. `high - low > 10` olduğu sürece `mid = (low + high) // 2` dene
3. Son 10 ID'yi tek tek tara, en yükseği son post ID

Sonraki pipeline çalışmalarında: bilinen son ID'den 20 ileriye bakılır, yeni post varsa cache güncellenir.

Son post ID cache'i DB'de `system_telemetry` tablosuna kaydedilmelidir (pipeline run'lar arası kalıcılık).

### Rate Limiting

- Kanallar arası bekleme: 0.5-1 saniye
- Google Translate çağrıları arası: 0.3 saniye
- Kanal başına max 3 post çekilir

### Takip Edilecek Kanallar

```json
[
  "OSINTdefender",
  "warfareanalysis",
  "rnintel",
  "GeoPWatch",
  "middle_east_spectator",
  "thecradlemedia",
  "dropsitenews",
  "france24_en",
  "PressTV",
  "iranintl_en",
  "AbuAliExpress",
  "QudsNen",
  "IDFofficial",
  "FarsNews_EN",
  "TasnimNewsEN",
  "SaberinFa",
  "defapress_ir",
  "sepah",
  "wamnews_en",
  "gulfnewsUAE",
  "aljazeeraglobal",
  "bintjbeilnews",
  "FotrosResistancee",
  "Alsaa_plus_EN",
  "Alertisrael",
  "HAMASW",
  "RocketAlert",
  "TimesofIsrael",
  "kianmeli1",
  "Alibk3",
  "Middle_East_Spectator"
]
```

### Çıktı Formatı

Her Telegram postu, SIM'in mevcut article formatına dönüştürülmelidir:

```json
{
  "title": "ilk 200 karakter",
  "link": "https://t.me/{kanal}/{post_id}",
  "pub_date": "ISO 8601 tarih (embed HTML'den)",
  "pub_dt": "datetime objesi",
  "description": "tam post metni (çevrilmişse çevrilmiş hali)",
  "source": "telegram",
  "domain": "t.me",
  "source_country": ""
}
```

Bu format, `pass_a_ingest.py`'daki mevcut RSS/GDELT/Nitter item yapısıyla uyumludur. Eklenen Telegram postları mevcut pipeline'dan (Pass B dedup → Pass C classify → ...) aynen geçer.

### Uyarılar

- `t.me` embed endpoint'i resmi API değildir; format değişebilir
- Google Translate `gtx` endpoint'i resmi değildir; yoğun kullanımda engellenebilir
- Sadece metin çekilir; fotoğraf/video alınamaz
- Binary search ilk çağrıda yavaştır (~15-20 HTTP isteği); cache kritik
