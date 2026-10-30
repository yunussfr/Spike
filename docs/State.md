# Proje Durumu (Günlük)

Her gün sonunda güncellenir. En yeni gün en üstte.

**Şu an:** Faz 3 ve Faz 4 kodu yazıldı, ayarın çoğu bitti; tam koşular (6 oturum × 5 seed) henüz yapılmadı. Ayrıntı ve kalan işler: 2026-10-29 girişi → "Teknik borç".
**Açık kararlar:** θ(τ) tablosu hâlâ "geçici"; `γ_rec` = 1.0 (1/3/10 arası fark gürültü düzeyinde).

---

## 2026-10-29

**Yaptıklarım** (hiçbiri commit edilmedi)
- [x] **Veri:** 6 NeuroBench primat oturumu indirildi (≈4,3 GB, md5 doğrulandı, `data/primate/`). 20 ms ızgaraya işlenip önbelleğe alındı (`data/primate/cache/`, özet `summary.json`). Oturumlar 13–41 dk, 96/192 kanal, sorun yok.
- [x] **Ortak altyapı:**
  - `src/utils/online_norm.py`: `OnlineStandardizer`, `nlms_step`. Beyincik buna taşındı, davranış aynı.
  - `src/data/primate.py`: oturum yükleme, bölme, kanal başına ölçek.
  - `experiments/common.py`: rezervuar kurulumu, r toplama, parçalı ridge, R².
- [x] **Script'ler:**
  - `scripts/download_primate.py`, `scripts/preprocess_primate.py`
  - `scripts/diagnose_nhp.py` (teşhis tablosu), `scripts/tune_affect.py`, `scripts/make_tables.py`
  - `experiments/tune_nhp.py` (val ayarı, r önbelleği), `experiments/nhp_motor.py` (A, B, B+Z; `--workers` ile paralel), `experiments/affect_viz.py`
- [x] **Faz 3 kodu:** `src/decision.py` (`DecisionReadout`, normalize LMS) ve `src/agent.py` (adım 1–8, A/B; beyincik `a = ŷ` alır).
  - Testler: `tests/test_decision.py`, `tests/test_agent.py`, `test_invariants.py`'ye K3 kontrolleri.
- [x] **Faz 4 kodu:** `src/affect.py` (`AffectSpace`, `AffectTargets`) ve ajanın 9. adımı (`affect.enabled`, B+Z).
  - Testler: `tests/test_affect.py` ve sızıntı testi (duygu uzayı açıkken ŷ, x̂ ve ağırlıklar birebir aynı).
- [x] **Mimari:** ARCHITECTURE.md §3.1 (kanal başına ölçek), §3.4 (duygu uzayı ayrıntıları), §3.6 (karar bloğunda normalize LMS) ve §11 güncellendi.
- [x] **Dokümanlar:**
  - `docs/Desicion.md`: KR-001…020, geçmiş kararlar dahil.
  - `docs/Explain.md`: düzeltme notları + Faz 3 anlatımı (veri, ön işleme, normalize LMS, karar bloğu, ajan akışı, A/B/C, R²).
- [x] **Teşhis (KR-014):**
  - Rezervuar hız bilgisini taşıyor: ridge val R² 0,516, ham sayım 0,11, 10 adım geçmiş 0,51.
  - `c_in`/`L` değişince fark gürültü düzeyinde.
  - Ateşleme oranı gerçek veride %1,0 → **K7 ihlali**. Kullanıcı kararı: §3.2 kalibrasyon kuralı **değişmez**, ihlal raporlarda not edilir.
- [x] **Faz 3 ayarı (KR-020, yalnızca val):** `nhp.epochs: 12`, `decision.norm_rate: 0.0003`, η = 0,01 (K4 varsayılanı). Val R² 0,267 → 0,381.
- [x] **NeuroBench taban çizgisi** bulundu (arXiv 2304.04640, Tablo 4): SNN Indy 0,593, Loco 0,568; ANN 0,593 / 0,558.
- [x] Koşturulan testler geçti: 35 test (decision, agent, affect, cerebellum). Tam `pytest` en son değişikliklerden sonra koşturulmadı.

**Sonuç (şimdilik)**
- `indy_20170131_02`, ayar seed'i 100, ilk ayarsız koşu: A test R² 0,334 = B (beklenen, K3), ridge 0,493.
- Duygu uzayı (duman + ilk ayar): baş hataları düşüyor, z'den iyi/kötü ayrımı val AUC 0,74–0,84 (`lr_z` 0,03–0,1). Ancak z'nin %50–75'i doymuş (|z| > 0,99).

### Teknik borç / kalan işler (buradan devam)

**Hemen**
1. ~~Tam `uv run pytest`~~ 75/75 geçti.
2. ~~Duygu uzayı `lr_z` seçimi~~ **Tamamlandı (2026-10-30, KR-021).**
   - Seçilen: `lr_z = 0,03` + z ön-aktivasyonuna L2 cezası `λ = 0,03` (`affect.preact_l2`), baş ağırlıkları [1, 1, 1].
   - Val sonucu: AUC 0,807 (Indy) / 0,814 (Loco); doygunluk %60–68 → %1–3; üç baş da iki oturumda öğreniyor.
   - Yedek seçenek (`lr_z = 0,01`) gerekmedi.
   - Tam tablo Desicion.md KR-021'de. ARCHITECTURE.md §3.4'e `− λ·a` terimi eklendi.
   - Açık: Loco'da u başı zayıf (0,81 → 0,80).
   - Ayrıca: tam `pytest` 75/75 geçti (2026-10-29, L2 eklenmeden önce). Ardından affect/agent/invariants testleri yeniden geçti.
3. ~~KR-019 kaydı~~ yazıldı.

**Faz 3 tam koşu**
4. `uv run python experiments/nhp_motor.py --config configs/nhp_motor.yaml --set "nhp.configs=[A, B, B+Z]" --workers 8`
   - 6 oturum × 5 seed; tahmini süre ~1–2 saat; RAM: büyük oturumda süreç başına ~3 GB.
   - Kontrol: tüm hücreler dolu, `leak_free: true`, NaN yok.
5. `uv run python scripts/make_tables.py --run runs/<koşu>` → `table.md`. A/B tablosu, B − A notu, NeuroBench karşılaştırması.
6. Faz 3 kapanışı: State.md sonuç girişi; PHASES.md Faz 3 ✅; CLAUDE.md "aktif faz" → Faz 4; Explain.md'ye sonuç yorumu.

**Faz 4 tam koşu**
7. `uv run python experiments/affect_viz.py --run runs/<aynı koşu>` → `affect_*.png`, `affect_summary.json`.
8. Faz 4 çıkış kriterleri:
   - (1) üç baş hatası düşüyor (5 seed);
   - (2) z'de iyi/kötü ayrışıyor (AUC + 3B grafik);
   - (3) görev performansı Faz 3 ile aynı (`r2_B+Z == r2_B`, `leak_free`).
9. Explain.md'ye duygu uzayı bölümü (z, üç baş, hedef standartlaştırma, W_z güncellemesi, sızıntı testi); Faz 4 kapanış dokümanları.

**Açık konular / bilinen sınırlar**
- K7 ihlali (KR-014): kural değişmedi, tablolarda ✗ olarak raporlanıyor. Faz 5'te τ değişince durum yeniden ölçülmeli.
- Yerel kural ile ridge arasında ~0,14 R² farkı var (NLMS yavaş yakınsıyor). Faz 3 kriteri bunu istemiyor ama Faz 6 için not.
- Hız / GPU: adım başına ~4 ms (CPU, 1 iş parçacığı); GPU tek deneme için daha yavaş (KR-015). GPU'yu doldurmak için bağımsız denemeleri batch'le: aynı r'yi okuyan ayar kombinasyonları `[K, …]` ve 5 seed'in rezervuarları `[5, N, N]` (`bmm`). Tahmini 5–20× hız (ölçülmedi). Faz 5–6 ablation'larından önce yapılmalı.
- Uzun koşularda bilgisayar uykuya geçmemeli: 2026-10-29 gecesi ayar koşusu ~10 saat askıda kaldı. İç döngüde `drive` her T adımı için ayrı hesaplanıyor (~0,4 ms/adım), iyileştirilebilir.
- r önbelleği (`data/primate/cache/r_*.pt`, ~0,3–1 GB/oturum) yalnızca ayar seed'i için. Rapor seed'leri her koşuda yeniden hesaplanıyor.
- Bölme farkı (KR-018): NeuroBench makalesi 75/25, biz kodun 80/10/10'u. Karşılaştırma birebir değil.
- Commit: Faz 2 sonrası hiçbir şey commit edilmedi. Faz 3 kapanınca commit kullanıcıya sorulacak.

---

## 2026-10-28

**Yaptıklarım**
- **Faz 2 kapatıldı.** `Cerebellum` (`src/cerebellum.py`), `configs/chaotic.yaml`, `experiments/cerebellum_test.py`, `tests/test_cerebellum.py`, K3/K6 arayüz testi.
- Normalize edilmiş çevrimiçi LMS kuralı ARCHITECTURE.md §3.3'e işlendi ve `base.yaml` varsayılanı oldu; §11'e kayıt eklendi.

**Etkisi**
- Ham delta kuralı ıraksıyordu çünkü okuma vektörü `[f, v]` büyük ölçekli (`‖φ‖² ~ 1e6`); veri setinin kendisi sorun değildi. Girişi nedensel standartlaştırıp adımı `‖φ̃‖²`'ye bölmek bunu çözdü.
- Aynı kural Faz 3'te karar bloğunda da gerekecek (ortak yardımcı önerilir).

**Sonuç** (`runs/20261025-230246_chaotic_seeds0-1-2-3-4`, Mackey-Glass τ=17, 5 seed)
- Normalize kural: MSE 0.00132 (NRMSE 0.161); "önceki değeri tekrarla" 0.00736 → oran **0.179** (kriter ≤ 0.5 ✔).
- Ridge üst sınırı: MSE 0.00124 (NRMSE 0.156) → fark **%6.8** (kriter < %20 ✔).
- Çevrimiçi ve dondurulmuş okuma neredeyse aynı. Ateşleme oranı ≈ %4.1 (K7 ✔). `plain` kural 5 seed'de de NaN.
- Testler: 46 geçti.

---

## 2026-10-25

**Yaptıklarım**
- `docs/` klasörü açıldı; `PHASES.md` buraya taşındı, bu `State.md` eklendi.

**Etkisi**
- Dokümanlar tek yerde; günlük ilerleme artık kayıt altında.

**Sonuç**
- Kod değişikliği yok.

---

## 2026-10-24

**Yaptıklarım**
- **Faz 0:** uv + `pyproject.toml` (CUDA torch), YAML config (base devralma, `--set`), seed sabitleme, koşu kaydı (config/seed/git/metrik), `experiments/run.py` iskeleti.
- **Faz 1:** `SpikeEncoder` (delta + termometre kanalları) ve `LIFReservoir` yazıldı; 37 test (K1, K2, K4, K7 dahil); bellek testi (`experiments/memory_test.py`, item/window hedefi, ideal doğrusal referans).
- Bellek testi ilk denemede başarısız çıktı → teşhislerle mimaride 8 değişiklik (ARCHITECTURE.md §11):
  - `(1−α)` kaldırıldı (uzun τ'da nöronlar ateşlemiyordu, %0.08).
  - k-WTA "tam k" → "eşiği geçenlerden en fazla k".
  - Okuma `r = [f, v]` (bellek zar potansiyelinde duruyor).
  - τ aralığı 5–100 → 20–2000 ms (ortam adımı 20 ms).
  - θ sabit → θ(τ) tablosu (geçici).
  - Nöron başına `τ_i = τ·m_i` ve `c_i` (tek τ'da bellek ~2 adımda bitiyordu).
  - Yumuşak sıfırlama `V ← V − θ`.
  - `γ_rec` rekürrent kazancı eklendi.
  - Kodlayıcıya seviye kanalları (delta kodlama seviye bilgisini taşımıyordu).

**Etkisi**
- Rezervuar artık τ ile gerçekten ölçeklenen bir belleğe sahip → regülatörün τ'yu ayarlaması (Faz 5) anlamlı hale geldi.
- Faz 1 çıkış kriteri sağlandı; Faz 2'ye geçilebilir.

**Sonuç** (`runs/20261024-131643_memory_seeds0-1-2-3-4`, 5 seed, window hedefi, `[f,v]` okuma)
- Bellek ufku (R² ≥ 0.1): τ = 20 / 100 / 500 / 2000 ms → **100 / 200 / 500 / 1000** adım (5 seed'de aynı).
- τ=2000 ufku / τ=20 ufku = **10×** (kriter ≥5×). Kısa pencerede en iyi τ=20, uzun pencerede τ=2000 ✔.
- Ateşleme oranı her τ'da ≈ %3.7 (K7 aralığında).
- Yalnız `f` okuması çok daha zayıf (ör. τ=2000, d=100: R² 0.07 vs `[f,v]` 0.80) → belleğin çoğu `v`'de.
- İdeal doğrusal referansa yakın (τ=2000, d=1000: 0.38 vs 0.42).
