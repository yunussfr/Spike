# Proje Durumu (Günlük)

Her gün sonunda güncellenir. En yeni gün en üstte.

**Şu an:** Faz 2 tamam → Faz 3 (karar bloğu, primat motor) sırada; veri indirilmedi.
**Açık kararlar:** θ(τ) tablosu hâlâ "geçici"; `γ_rec` = 1.0 (1/3/10 arası fark gürültü düzeyinde).

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
