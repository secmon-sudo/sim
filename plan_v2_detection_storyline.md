# SIM — Tespit · Süzme · Sınıflandırma · Storyline İyileştirme Planı (V2)

> Bu plan, mevcut `plan.md` (genel kod kalitesi) ile **çakışmaz, onu tamamlar**.
> Burada odak **domain kalitesi**: havacılık güvenliği önceliğiyle haber tespiti,
> süzme, sınıflandırma ve storyline (olay geçmişi/hikâye) mekanizmalarının
> mükemmelleştirilmesi.

## Kapsam kararı (kullanıcı onayı)
- **Geniş kapsam korunur.** Jeopolitik / askeri / denizcilik / siber / protesto
  olayları kaçırılmayacak — bunlar önemli.
- **Havacılık ön planda.** Her olaya bir "havacılık-nexus" boyutu eklenir; skor,
  alert ve storyline havacılığı tehdit eden olayları öne çıkarır. Jeopolitik olaylar
  bağlam/erken-uyarı olarak kalır.
- **Asıl ağrı noktaları:** (1) duplikasyon, (2) eski tarihli olayların süzülmesi,
  (3) gereksiz/junk olayların elenmesi, (4) storyline'ın gerçek bir "geçmiş hikâye"
  gibi kurulması.
- **Emniyet vs Güvenlik (öneri, onay bekliyor):** Katalog şu an kasıtsız KAZA/emniyet
  olaylarını (`bird_strike`, `engine_failure`, `emergency_landing`,
  `depressurization`, `mayday`) kasıtlı güvenlik olaylarıyla karıştırıyor.
  **Öneri:** silme yerine `safety` olarak etiketle + düşük öncelik izine al; yalnız
  sabotaj/tehdit şüphesi varsa yükselt. (Faz 2.4)

---

## LLM TOKEN BÜTÇESİ (sert kısıt — her madde buna uymalı)
Kullanıcı 3-4 ayrı key ile free-tier limitlerini aşmaya çalışıyor. Gerçek bütçe
(`llm_router.py:154-212`):
- **İyi modeller:** 4 Groq slotu × 1000 RPD = ~**4.000 RPD** (gpt-oss-120b, llama-3.3-70b,
  llama-4-scout, qwen3-32b) + OpenRouter 2×200 = 400 RPD.
- **Son çare:** `llama-3.1-8b-instant` = 14.400 RPD (düşük kalite, bulk).
- **Asıl darboğaz RPM:** slot başına 30-60. Pass C her olay için 1 çağrı yapıyor.

**Tasarım kuralları:**
1. **Deterministik-önce.** Faz 0/2/4'ün tamamı + Faz 1.1-1.2 **sıfır LLM** ile yapılır
   (regex, imza, anchor/entity, tarih sınırı). Bunlar önce gelir.
2. **Filtreleme = token tasarrufu.** Junk eleme ve dedup'ı **Pass C'den ÖNCE** sıkılaştırmak
   gereksiz sınıflandırma çağrılarını keser → net token *kazancı*. Faz 1/2 token-pozitiftir.
3. **Embedding'de API kullanma.** Faz 3.1 anlamsal benzerlik gerekirse **yerel** model
   (sentence-transformers, sıfır API) veya hiç; varsayılan leksikal+entity (sıfır token).
4. **Narrative özet bütçeli.** Faz 3.2 "story so far" özetleri: olay başına DEĞİL,
   yalnız *aktif + yüksek-severity* storyline'lar için, **artımlı** (yeni olay gelince),
   run başına tavanlı (örn. ≤5), `8b-instant` bulk slotunda + DB cache. İçerik değişmezse
   yeniden üretme.
5. **Pass C en büyük tüketici.** Yeni LLM çağrısı eklemeden önce mevcut çağrı sayısını
   düşürmeyi (daha iyi ön-filtre) tercih et.

---

## DOĞRULANAN BULGULAR

### A. Süzme / Tespit hassasiyeti (junk eleme)
| # | Bulgu | Dosya | Önem |
|---|-------|-------|------|
| A1 | `_matches_security_keywords` ve `_HIGH_SIGNAL_TERMS` **substring** eşleşmesi yapıyor (`kw in text`). "war"→Warsaw/forward, "coup"→couple, "riot"→patriot, "dead"→deadline, "rocket"→rocketship → **yanlış pozitif**. Noise filtreleri zaten `\b` word-boundary kullanıyor; bunlar kullanmıyor. | `pass_a_ingest.py:335-390` | 🔴 |
| A2 | `is_noise` askeri-bypass çok geniş: "missile", "killed in", "casualties" geçen her şey noise filtresini atlıyor → "air crash investigation" belgeseli bile kurtulabiliyor. | `pass_a_ingest.py:300-329` | 🟠 |
| A3 | Relevans tamamen LLM `relevance_score`'una bağlı; deterministik ön-eşik yok. LLM hatası tüm hattı kirletir. | `pass_c_classify.py:289-339` | 🟠 |
| A4 | Havacılık önceliği için deterministik sinyal yok — "havalimanı/havayolu/hava sahası/personel" yakınlığı sadece anchor proximity bonusu ile dolaylı. | `pass_d_score.py`, prompt | 🟠 |

### B. Duplikasyon
| # | Bulgu | Dosya | Önem |
|---|-------|-------|------|
| B1 | Çapraz-kaynak aynı olay birleşmiyor. Dedup yalnız URL-hash (Pass B) + title/içerik benzerliği (Pass A). Farklı yayın aynı olayı farklı başlıkla verince `title_sim < 0.65` → ayrı event olarak kalıyor → **görünür duplikasyon**. | `pass_a_ingest.py:1282-1298`, `pass_b_dedup.py` | 🔴 |
| B2 | İçerik benzerliği `difflib.SequenceMatcher` ile **tam metin** üzerinde O(N·M); pahalı ve kırılgan (çeviri/kırpma sonrası kayar). Shingle/MinHash/SimHash yok. | `pass_a_ingest.py:1294-1297` | 🟠 |
| B3 | Dedup mantığı iki yerde (Pass A in-memory + Pass B URL-hash) — tek kaynağa inmeli. | `pass_a_ingest.py`, `pass_b_dedup.py` | 🟡 |
| B4 | Storyline kümeleme (görünür duplikasyonu çökertecek katman) yalnız haftalık forecast'te kullanılıyor; ana akış/alert tarafında olayları "tek hikâye altında" toplayıp tekrarı bastırma yok. | `storyline_clusterer.py` | 🟠 |

### C. Eski tarih / zaman bütünlüğü
| # | Bulgu | Dosya | Önem |
|---|-------|-------|------|
| C1 | `occurred_at_est` LLM'den geliyor ama **sınır kontrolü yok** — yıllar öncesi ya da gelecek tarih kabul ediliyor. Storyline penceresini ve forecast'i kirletir. | `pass_c_classify.py:144-172,366` | 🔴 |
| C2 | Yaş filtresi yalnız RSS yayın tarihine bakıyor (`max_article_age_days=2`); yeni yayınlanan ama **eski olayı** anlatan içerik (yıldönümü/retrospektif) yalnız zayıf noise kelimeleriyle ("anniversary", "years ago") yakalanıyor. Olay tarihine göre sert eleme yok. | `pass_a_ingest.py`, `settings.json:10` | 🟠 |
| C3 | Storyline zaman penceresi `time_certainty="unknown"` olayları için tanımsız; null `occurred_at_est` linklemeyi tamamen atlatıyor → izole eventler. | `storyline.py:61-69` | 🟡 |

### D. Storyline / hikâye kurma
| # | Bulgu | Dosya | Önem |
|---|-------|-------|------|
| D1 | **Konfig yok sayılıyor.** `pass_d` `link_storylines` → `should_link_storyline(event, existing)` çağrısı threshold/max_days **geçmiyor**; hardcoded `0.35`/`14` kullanılıyor. `settings.json` `jaccard_threshold=0.4`, `time_window_days=14`, `country_match_required=true` **etkisiz**. | `pass_d_score.py:212-221`, `storyline.py:50` | 🔴 |
| D2 | Eşik tutarsızlığı: `should_link`=0.35, config=0.4, `storyline_clusterer`=0.40. Tek kaynak yok. | birden çok | 🟠 |
| D3 | **İlk eşleşen** seçiliyor, en benzer değil. `link_storylines` threshold'u geçen ilk olaya bağlıyor → sıra-bağımlı yanlış linkleme. En yüksek benzerlik seçilmeli. | `pass_d_score.py:212-221` | 🟠 |
| D4 | Linkleme tamamen **leksikal** (4-6 kelimelik LLM hint üzerinde Jaccard). Anlamsal/entity/anchor temelli linkleme yok; LLM aynı hint'i üretemezse hikâye kopuyor. | `storyline.py` | 🟠 |
| D5 | Hint'e **tarih token'ı zorunlu** (Jun8). Bu Jaccard'a sızıyor: aynı olay iki günde Jun8/Jun9 → benzerlik düşer; farklı olaylar aynı gün → şişer. `AVIATION_STOPWORDS` ay/tarih token'larını ayıklamıyor. | `pass_c_classify.py:59-73`, `storyline.py:12-25` | 🟠 |
| D6 | `same_country` çok gevşek: null ISO her zaman eşleşiyor → ilgisiz olaylar global linklenebilir. `country_match_required` config'i uygulanmıyor. | `storyline.py:58` | 🟡 |
| D7 | Storyline = sadece paylaşılan `storyline_id`. **Anlatı yok**: zaman sıralı timeline, neden-sonuç/gelişim zinciri, "bu bir geçmiş hikâye" sentezi yok. Kullanıcının asıl istediği bu. | tüm storyline katmanı | 🟠 |
| D8 | Greedy centrist clustering ilk öğeyi centroid alıyor → sıra-bağımlı, yeniden-merkezleme yok; tek hint centroid kırılgan. | `storyline_clusterer.py:89-126` | 🟡 |

### E. Korelasyon bug'ları (mevcut plan ile kesişen)
- Alert, DB commit'ten **önce** gönderiliyor → tekrar alert riski (`pass_d_score.py`). [bkz plan.md 1.1]
- Geniş `except Exception` blokları hataları yutuyor (storyline, anchor, czib).

---

## FAZLI PLAN

### Faz 0 — Hızlı doğruluk düzeltmeleri (düşük risk, yüksek değer) ✅ TAMAMLANDI
> Uygulandı (2026-06-20). 83/83 test yeşil. Davranış doğrulamaları geçti.
> Ek: `config/keywords.json`'a havacılık-güvenliği terimleri eklendi (GPS jamming/
> spoofing, hava sahası kapatma/ihlali, uçağa ateş/uçak düşürme, MANPADS, perimeter
> ihlali, kule/pist saldırısı — EN+TR).
> Ek: 13 yeni RSS kaynağı eklendi (hepsi curl ile 200+geçerli XML doğrulandı) —
> çatışma erken-uyarı/istihbarat (Crisis Group, Bellingcat, Cipher Brief, Foreign
> Policy), ABD savunma (Defense One, The War Zone, Defense News), Ortadoğu/İran
> (Al-Monitor), Rusya-Ukrayna (Moscow Times, Meduza, Warsaw Institute), küresel
> (UN News, BBC World). Kredibilite matrisine + TRUSTED_DOMAINS'e işlendi. Genel
> kaynaklar `_matches_security_keywords` (artık word-boundary) ile süzülüyor → güvenli.

0.1 **Storyline konfigini bağla (D1/D2).** `link_storylines` config'ten
   `jaccard_threshold`, `time_window_days`, `country_match_required` okusun ve
   `should_link_storyline`'a geçirsin. Eşiği tek yerden yönet (`settings.json`).
   *Test:* config 0.4 iken 0.37 benzerlikli çift linklenmemeli.
0.2 **En-iyi-eşleşme linkleme (D3).** İlk yerine en yüksek Jaccard'lı mevcut olayı seç.
0.3 **occurred_at_est sınır kontrolü (C1).** `_parse_occurred_at`'e makullük sınırı:
   gelecek > +1 gün veya geçmiş > `max_event_age_days` (örn. 30) ise reddet → `None`
   + `time_certainty='unknown'`. *Test:* 2019 tarihli LLM çıktısı reddedilir.
0.4 **Substring → word-boundary (A1).** `_matches_security_keywords` ve
   `_HIGH_SIGNAL_TERMS` regex `\b...\b` ile eşleşsin (noise filtresindeki desen).
   *Test:* "Warsaw summit" artık "war" ile tetiklenmez; "drone strike on airport" tetikler.

### Faz 1 — Junk eleme & havacılık önceliği
1.1 **Deterministik relevans tabanı (A3) — [sıfır token, token-pozitif]. ✅ TAMAMLANDI**
   Uygulandı (2026-06-20). `deterministic_relevance()` + Pass C ön-eleme: hiç güvenlik
   sözcüğü içermeyen makaleler LLM çağrısı yapılmadan arşivleniyor (token tasarrufu);
   ayrıca yüksek-sinyal (explosion/airstrike/killed) varsa LLM'in "noise" demesi
   geçersiz kılınıyor (yanlış-negatif koruması). Config: `classification.*`. 9 yeni test.
1.2 **Askeri-bypass daraltma (A2). ✅ TAMAMLANDI.** Uygulandı (2026-06-20).
   `_BYPASS_CANCEL_PATTERN`: belgesel/film/kitap/podcast/op-ed/explainer/"in pictures"
   gibi medya-recap işaretleri askeri-bypass'ı iptal ediyor → "documentary about the
   missile strike" artık noise (önce "missile" onu kurtarıyordu). Canlı olaylar korunur.
   Konservatif: salt opinion/analysis noise'a zorlanmaz (canlı çatışma analizi
   düşürülmesin diye), Pass C zaten <30 puanlıyor.
1.3 **Havacılık-nexus boyutu (A4). ✅ TAMAMLANDI.** Uygulandı. `compute_aviation_bonus`
   (+15, config `scoring.aviation_nexus_bonus`): havacılık event_type'ı VEYA başlık/anchor'da
   havacılık sözcüğü VEYA LLM `aviation_impact=direct` → severity bonusu. Jeopolitik/
   denizcilik/siber/protesto olayları **yine yakalanıp skorlanır** (sadece bonus almaz)
   → geniş kapsam korunur, yalnız sıralama havacılık lehine. 9 yeni test.
1.4 **Sınıflandırma prompt'u havacılık-öncelikli (`pass_c`). ✅ TAMAMLANDI.** Aynı LLM
   çağrısı (ekstra token YOK); prompt'a `aviation_impact` (direct|indirect|none) alanı +
   "havacılık öncelik domaini" talimatı eklendi. `compute_aviation_bonus` bu alanı okur.

### Faz 2 — Duplikasyon çökertme
2.1 **Çapraz-kaynak near-dup (B1/B2). ✅ TAMAMLANDI (kısmi).** Uygulandı (2026-06-20).
   Pahalı tam-metin `SequenceMatcher` → word-shingle Jaccard (hız + truncation-robust);
   ek olarak başlık word-set Jaccard (reordered/syndicated başlıkları yakalar). Config:
   `dedup.title_token_jaccard_threshold`, `content_shingle_threshold`. 10 yeni test.
   **NOT (önemli):** Tam-paraphrase çapraz-kaynak ("explosion"↔"blast") leksikal dedup
   ile çözülemez (eşik düşürmek yanlış-birleştirir) → bu **storyline linkleme** işidir
   (Faz 3.1). Ingest dedup syndicated/reordered/repost vakalarını kapsar.
2.2 **Dedup tek kaynağa (B3).** URL-hash + içerik-imza tek modülde; Pass A ön-eleme,
   Pass B otoritatif. Ortak hash fonksiyonu.
2.3 **Storyline'ı ana akışta birleştirici yap (B4).** `storyline_clusterer`'ı sadece
   forecast'te değil, alert/gösterim öncesinde de çalıştır: aynı hikâyedeki N event
   tek "story card" altında toplansın, tekrar alert bastırılsın (suppression key =
   storyline_id, zaten kısmen var → güçlendir).
2.4 **Emniyet izini ayır. ✅ TAMAMLANDI (Seçenek A).** Uygulandı (2026-06-20).
   `SAFETY_EVENT_TYPES` (bird_strike, engine_failure, emergency_landing,
   depressurization) → `apply_safety_downrank`: severity `safety_severity_cap=40`'a
   çekilir (alert eşiğinin altı) ve `is_safety=TRUE` etiketlenir (migration 011,
   queryable). **Kitlesel-kayıp varsa cap kalkar** (fatal kaza yine yüzeye çıkar) —
   "güvenlik odağı" + "büyük olayı kaçırma" dengesi. Pass D + Pass E'de tutarlı.
   Silme YOK → kapsam korunur. Config `scoring.safety_*`. 5 yeni test.

### Faz 3 — Storyline'ı "geçmiş hikâye" olarak kurma (D4/D7/D8)
3.1 **Hibrit linkleme — [sıfır token]. ✅ TAMAMLANDI.** Uygulandı (2026-06-20).
   Leksikal Jaccard + **anchor-assist**: aynı normalize anchor + dar zaman penceresi
   (≤72s) → kelime farklı olsa bile link (çapraz-kaynak paraphrase-duplikasyonun
   çözümü); uzak zamanda leksikal overlap istenir (farklı olayları birleştirmez).
   Tarih token'ı Jaccard'dan çıkarıldı (`_DATE_TOKEN`: ay+gün, çıplak sayı; uçuş no
   "dl54" korunur). anchor `_fetch_recent_events_for_linking`'e + event dict'ine bağlandı.
   Config: `storyline.anchor_assist_threshold`, `anchor_assist_max_hours`. Yerel/semantik
   embedding **eklenmedi** (API token YOK kuralı; anchor+leksikal yeterli). 9 yeni test.
3.2 **Narrative timeline. ✅ TAMAMLANDI (yapısal + LLM prose).** Uygulandı (2026-06-20).
   (a) Sıfır-LLM: `core/storyline_narrative.py` — `build_timeline()` + `summarize_timeline()`
   (span, kaynak/ülke/anchor, peak severity, escalating/stable/deescalating trend).
   (b) LLM prose: `services/storyline_narrator.py` — "story so far" özeti, **cache'li**
   (`storyline_narratives` tablosu + signature; değişmeyen storyline'a çağrı yok),
   **bulk router'a izole** (`build_bulk_router`, 8b/14.4K RPD — iyi modeller Pass C'ye
   kalır), rasyonlu (min_severity=65, min_events=2, ≤10/run). Orchestrator'a Pass E
   sonrası hata-güvenli bağlandı. Migration 010. Config `narrative.*`. 6 yeni test.
3.3 **Centroid sağlamlaştırma (D8).** [Ertelendi] `storyline_clusterer` yalnız haftalık
   forecast'te; ana akışı etkilemiyor → düşük öncelik.
3.4 **Ülke eşleşmesini sıkılaştır (D6). ✅ TAMAMLANDI (3.1 ile).** `country_match_required`
   artık sert kapı: iki ülke de biliniyor ve farklıysa link yok; null ISO lenient kalır.

### Faz 4 — Eski tarih & zaman bütünlüğü (C2/C3)
4.1 **Olay-tarihi sert eleme.** Büyük ölçüde Faz 0.3 ile kapsandı: `occurred_at_est`
   >30 gün/gelecek LLM tahminleri parse'ta zaten `None`'a düşürülüyor; yayın tarihi de
   `max_article_age_days=2` ile sınırlı. Kalan açık düşük → ayrı Pass D guard'ı
   gereksiz (dead code) olacağı için eklenmedi.
4.2 **Retrospektif tespiti. ✅ TAMAMLANDI.** Uygulandı (2026-06-20). `_RETROSPECTIVE_PATTERN`
   `is_noise`'da askeri-bypass'tan ÖNCE çalışıyor → "10. yıldönümü hava saldırısı",
   "5 years ago bombing", "on this day" gibi geçmiş-olay özetleri askeri terim içerse
   bile eleniyor. 10 yeni test (dedup ile birlikte).
4.3 **`time_certainty='unknown'` politikası.** Linklemede null zaman için ülke+anchor
   zorunlu; aksi halde izole bırak (yanlış linklemeyi önle).

### Faz 5 — Bug & sağlamlık (mevcut plan ile ortak)
5.1 **Alert outbox sıralaması. ✅ TAMAMLANDI.** Uygulandı (2026-06-20). `dispatch_alert()`
   helper'ı: suppression **önce kaydedilir+commit edilir, sonra gönderilir** → gönderimle
   kayıt arasındaki çökme artık tekrar-alert üretemez. Gönderim başarısızsa suppression
   geri alınır (sibling event tekrar deneyebilir). 5 yeni test (sıralama doğrulanıyor).
5.2 **Geniş `except` temizliği.** [Kısmi/ertelendi] Yeni kodda dar `except`'ler ve
   `logger.exception` kullanıldı; eski modüllerdeki tarama düşük öncelik.
5.4 **"NEW LOCATION/COUNTRY" etiket sistemi düzeltmesi. ✅ TAMAMLANDI.** (`pass_d_score`
   quiet-hours → `telegram_notifier` etiketleri). (a) Sinyal kirliliği: `archived` (noise/
   eskiyen) olaylar artık "aktivite" SAYILMIYOR → gerçek ilk güvenlik olayı "NEW COUNTRY"
   etiketini doğru alıyor. (b) `is_safety` olayları aktivite saymıyor (sadece bird strike
   olan ülke güvenlik açısından "sessiz" kalır). (c) N+1 → tek `FILTER` sorgusu. (d) Pencere
   config'li (`alert.new_activity_window_hours`, `make_interval` ile injection-safe).
5.3 **Yeni testler. ✅ TAMAMLANDI.** 58 yeni test eklendi (storyline config+best-match+
   hibrit, occurred_at sınırı, word-boundary, near-dup, retrospektif, aviation bonus,
   narrator cache, alert sıralaması). Tümü mevcut `MockDB` stilinde.

---

## ÖNERİLEN SIRA
Faz 0 (doğruluk, küçük diff) → Faz 1.1-1.2 + Faz 4.1 (junk & eski tarih, en görünür kazanım)
→ Faz 2 (duplikasyon) → Faz 1.3-1.4 (havacılık önceliği) → Faz 3 (narrative storyline)
→ Faz 5 (bug/test). Faz 0 ve 2 en yüksek "hemen hissedilir" değer.

## DOĞRULAMA (her faz)
```
cd /home/ubuntu/Desktop/github_repos/sim
pytest -q                              # mevcut + yeni testler yeşil
python -m src.pipeline.orchestrator    # (DB varsa) tek tur smoke
```
- Davranış değiştirmeyen taşımalar için: eski sabit == yeni config değeri assert'i.
- Storyline değişiklikleri için: bilinen çift olay seti üzerinde link/no-link assertion'ları.
