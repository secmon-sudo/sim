# IRONSIGHT OSINT Pipeline — İyileştirme Planı

## Context (Neden bu plan?)

Proje (`/home/ubuntu/Desktop/claude_code/sim`) çalışan bir OSINT/jeopolitik istihbarat
pipeline'ı: RSS/GDELT/advisory/Nitter kaynaklarından haber çekip 6 aşamalı bir hat
(Pass A→F) ile dedup → LLM sınıflandırma → skorlama → alert → arşiv yapıyor, ayrıca
haftalık tahmin (forecast) üretiyor. Kod çalışıyor ama bir gözden geçirme istendi:
fazlalıklar, riskler ve iyileştirmeler. **Streamlit (`streamlit_app/`) kapsam dışı —
kullanılmıyor.**

Bu plan bir **bulgu + aksiyon yol haritasıdır**. Her madde tek başına uygulanabilir;
hiçbiri mevcut davranışı zorunlu kılmadan kırmıyor. Sıralama risk önceliğine göre.

> Not: Keşif sırasında "logs/, forecast.md, IRONSIGHT_DATA_SOURCES.md, .pytest_cache
> git'e commit edilmiş" iddiası çıktı; **doğruladım, yanlış.** `git ls-files` bunların
> izlenmediğini gösteriyor — `.gitignore` doğru çalışıyor. Bu yüzden plana **alınmadı**.

---

## Doğrulanan ana bulgular

| # | Konu | Dosya | Önem |
|---|------|-------|------|
| 1 | Alert/notify, DB commit'ten ÖNCE gönderiliyor → yinelenen alert riski | `pass_d_score.py`, `pass_c_classify.py`, `pass_f_archive.py` | 🔴 |
| 2 | `pass_a_ingest.py` 1517 satır monolit; `run_pass_a()` ~213 satır | `pass_a_ingest.py` | 🔴 |
| 3 | Konfig sabitleri kod içinde dağınık (eşikler, ağırlıklar, kredibilite matrisi) | tüm pass'ler, `core/` | 🟠 |
| 4 | LLM hesapları + Türkçe yorumlar koda gömülü | `llm_router.py:154-214` | 🟠 |
| 5 | İki Telegram notifier'da kopya kimlik/env okuma kodu | `telegram_notifier.py`, `telegram_report_notifier.py` | 🟠 |
| 6 | N+1 sorgular (diversity skoru, sibling fetch) | `pass_d_score.py`, `pass_e_reconcile.py` | 🟠 |
| 7 | Kritik modüllerde test yok (orchestrator, pass_b/c/e/f, supabase_client) | `tests/` | 🟠 |
| 8 | `token_bucket` property'leri lock'suz okunuyor → race | `core/token_bucket.py:80-97` | 🟡 |
| 9 | Anchor normalize mantığı Pass D ve Pass E'de tekrar; confidence sırası 2 yerde | `pass_d/e`, `anchor.py` | 🟡 |
| 10 | URL dedup mantığı Pass A ve Pass B'de iki yerde | `pass_a_ingest.py`, `pass_b_dedup.py` | 🟡 |
| 11 | `requirements.txt` tamamen `>=`, üst sınır yok → CI kırılganlığı | `requirements.txt` | 🟡 |
| 12 | `settings.json` içinde kullanılmayan `archive_endpoint: ""` | `config/settings.json` | 🟢 |
| 13 | Geniş `except Exception` blokları hataları yutuyor | `czib_client.py`, `anchor.py`, vb. | 🟡 |
| 14 | RLS politikaları herkese `SELECT USING (true)` veriyor | `db/migrations/009_rls_policies.sql` | 🟠 (doğrulanmalı) |

---

## Faz 1 — Doğruluk / Veri bütünlüğü (önce bunlar)

### 1.1 "Önce gönder, sonra commit" sırasını düzelt (🔴)
**Sorun:** `pass_d_score.py:~342` Telegram alert'i DB commit'ten önce yolluyor;
suppression kaydı yazılmadan ağ hatası olursa aynı alert tekrar gidiyor. Aynı desen
Pass C telemetri (`:404`) ve Pass F telemetri/silme sırasında da var.
**Yapılacak:** Sıralamayı *commit → sonra side-effect* yap. Telegram için
"outbox" deseni: önce `alert_suppression` satırını commit et, sonra gönder; gönderim
dönüş değerini kontrol et ve başarısızsa logla (şu an dönüş değeri yutuluyor).
**Dosyalar:** `pass_d_score.py`, `pass_c_classify.py`, `pass_f_archive.py`.

### 1.2 Telemetri commit'i emanet bırakma
Telemetri kaydını ana event commit'iyle aynı transaction'a al ya da event commit
başarılı olduktan sonra yaz (`pass_c_classify.py:404`, `llm_client.py:150`,
`pass_f_archive.py:245-256`).

---

## Faz 2 — Konfigürasyon merkezileştirme (🟠)

Tekrarlayan asıl sorun: eşik/ağırlık/sabitler kod içinde. Tek bir `config/settings.json`
şemasına taşı, modüller başlangıçta okusun. Önce env override deseni zaten var
(`SIM_MATURATION_WINDOW_HOURS`) — onu standartlaştır.

Taşınacak sabitler (örnek, hepsi değil):
- `pass_a_ingest.py:36-39` → `content_sim_threshold`, `title_sim_threshold`
- `pass_d_score.py:26-64` → kaynak kredibilite matrisi + skor ağırlıkları (casualty/proximity/czib bonus)
- `storyline.py:50`, `storyline_clusterer.py:64,89`, `anchor.py:55` → benzerlik/zaman eşikleri
- `heartbeat.py:18`, `llm_client.py:58` → error limiti, timeout
- `pass_f_archive.py:32,40` → arşiv gün eşiği, batch size
- `forecast_engine.py:14-63` → profil çarpanları + TI ağırlıkları

**Strateji:** Tek seferde değil, modül modül. Her modüle `_cfg = load_settings()`
yardımcı fonksiyonu. Kredibilite matrisi gibi büyük tablolar uzun vadede DB tablosu
olabilir ama ilk adımda JSON yeterli.

### 2.1 LLM router hesaplarını koddan çıkar (🟠)
`llm_router.py:154-214` 8 hesabı sabit gömüyor + Türkçe yorumlar ("en akıllı", "son çare").
`config/llm_accounts.json` benzeri bir tabloya taşı (model, rpm, rpd, priority, env_key).
`build_llm_router()` bu listeyi okusun. Türkçe yorumları İngilizce'ye çevir (kod tabanı
İngilizce; forecast'in Türkçe label'ları ayrı konu — bkz 4.2).

---

## Faz 3 — Tekrar (DRY) temizliği (🟠/🟡)

### 3.1 Telegram credential yardımcısı
`telegram_notifier.py:47-51` ile `telegram_report_notifier.py:245-249,328-332`
aynı env okumayı yapıyor. Ortak `get_telegram_credentials()` çıkar. Ayrıca
`telegram_report_notifier.py:306` `sendDocument` retry'siz — `sendMessage`'deki
`_post_telegram` retry wrapper'ını dosya yüklemeye de uygula.

### 3.2 Anchor normalize + confidence sırasını tekilleştir
Pass D ve Pass E her ikisi de anchor normalize + severity/confidence yeniden hesaplıyor;
confidence rank haritası (`{"LOW":0,"MEDIUM":1,"HIGH":2}`) `pass_e_reconcile.py:67` ve
`anchor.py:68-75` içinde kopya. Tek kaynağa indir (`anchor.py`'de helper).

### 3.3 URL dedup mantığını tek yere
`pass_a_ingest.py:1402-1413` ve `pass_b_dedup.py:176-204` aynı URL-hash dedup'ı yapıyor.
Ortak fonksiyon; Pass A in-memory ön eleme yapsa bile aynı hash fonksiyonunu paylaşsın.

---

## Faz 4 — Performans ve sağlamlık (🟡)

### 4.1 N+1 sorguları
- `pass_d_score.py:152` diversity skoru her event için storyline domain sayımı yapıyor →
  storyline başına bir kez hesapla/cache'le.
- `pass_e_reconcile.py:50-59` sibling text'i event başına ayrı sorgu → storyline başına
  toplu fetch.

### 4.2 Forecast'teki Türkçe label'lar
`forecast_engine.py:267-271` trajectory'yi "Tırmanıyor/Azalıyor/Stabil" string olarak
döndürüyor; `weekly_forecast.py:295` bu stringi karşılaştırıyor. Enum'a çevir
(`TrajectoryDirection.RISING` vb.), görüntü stringini locale tablosuna bırak. Bu, label
değişirse karşılaştırmaların sessizce kırılmasını önler.

### 4.3 token_bucket race
`token_bucket.py:80-97` `daily_used`/`remaining_daily`/`utilization_pct` lock'suz
okuyor. Property okumalarını da `self._lock` altına al.

### 4.4 Geniş except blokları
`czib_client.py:73-143`, `anchor.py:62`, `storyline.py:68`, `flash_detector.py:31`
tüm `except Exception:` → en azından `logger.exception(...)` ekle; mümkünse beklenen
hata tipini (örn. `psycopg.errors.UniqueViolation`) ayır.

---

## Faz 5 — Test kapsamı (🟠)

Şu an 27 kaynak modülün ~12'sinde test yok. Kritik boşluklar (öncelik sırası):
1. `pass_b_dedup.py` — dedup mantığı (sessiz veri kaybı riski yüksek)
2. `orchestrator.py` — giriş noktası, en azından pass çağrı sırası + hata yutmama
3. `pass_c_classify.py` — `validate_and_parse()` JSON temizliği birim testleri
4. `pass_e_reconcile.py` — anchor upgrade → yeniden skorlama
5. `heartbeat.py` — lock kaybı / max consecutive error davranışı
6. `supabase_client.py` — bağlantı/secret yükleme (mock'la)

Mevcut `MockDB/MockCursor` deseni korunabilir; yeni testler aynı stili izlesin.
Hedef: dedup + orchestrator + parse mantığı için somut assertion'lar (görsel/IO değil).

---

## Faz 6 — Küçük temizlik (🟢)

- `config/settings.json:65` kullanılmayan `archive_endpoint: ""` → kaldır veya yorumla.
- `requirements.txt` → test edilen sürümlere pinle (`==`), en azından üst sınır ekle.
- `db/migrations/009_rls_policies.sql` — `USING (true)` herkese okuma veriyor. Sistem
  kapalıysa rol bazlı kısıtla. **Önce kullanım amacını teyit et** (kasıtlı public olabilir).

---

## Kapsam dışı
- `streamlit_app/` (kullanıcı kullanmıyor)
- Git tracking / .gitignore (zaten doğru — yanlış alarmdı)

---

## Doğrulama (her faz sonrası)

```
cd /home/ubuntu/Desktop/claude_code/sim
pytest -q                      # mevcut + yeni testler yeşil kalmalı
python -m src.pipeline.orchestrator   # (DB erişimi varsa) tek tur smoke
```
- Faz 1 için: alert gönderimini mock'layıp commit-öncesi/sonrası sırayı test et.
- Faz 2 için: settings.json'dan okunan değerin eski sabitle aynı olduğunu assert et
  (davranış değişmemeli — sadece kaynak taşınıyor).
- Faz 5 için: `pytest --cov=src` ile kapsam artışını ölç.

## Önerilen uygulama sırası
Faz 1 (doğruluk) → Faz 5'ten sadece pass_b testi → Faz 3 (DRY, düşük risk) →
Faz 2 (konfig) → Faz 4 → kalan testler → Faz 6. Faz 1 ve Faz 2 en yüksek değer.
