# Karar Kaydı

Projede alınan her mimari ya da deneysel karar burada tutulur. Kural değişikliği gerektiren kararlar ayrıca `ARCHITECTURE.md` §11'e işlenir.
Numara biçimi `KR-xxx` (Karar). `K1–K8` ise ARCHITECTURE.md'deki değişmez kurallardır; karıştırmamak için farklı önek kullanılır.

Her kayıt:
- **Bağlam:** Kararı ne gerektirdi?
- **Karar:** Ne yaptık?
- **Alternatifler:** Başka ne düşünüldü, neden seçilmedi?
- **Sonuç / Durum:** Ölçülen etki, hâlâ açık mı?

---

## Faz 1 — Projeksiyon bloğu (2026-10-24)

İlk bellek testi başarısız oldu. Aşağıdaki sekiz karar, teşhis tablolarıyla tek tek alındı.

### KR-001 — Girdi akımındaki `(1−α)` çarpanı kaldırıldı
- **Bağlam:** `V = α·V + (1−α)·I` ile τ büyüdükçe girdi gücü küçülüyordu; τ=100 ms'de ateşleme oranı %0,08'e düştü.
- **Karar:** `V_i = α_i·V_i + c_i·I_i`. τ artık girdiyi zayıflatmıyor, yalnızca birikimin ne kadar kalacağını belirliyor.
- **Sonuç:** Uzun τ'da nöronlar ateşleyebiliyor. Bunun yan etkisi KR-005'i gerektirdi.

### KR-002 — k-WTA: "tam k" yerine "eşiği geçenlerden en fazla k"
- **Bağlam:** "Her adımda tam k nöron" kuralı, "girdi yokken aktivite sönsün" kararlılık testiyle çelişiyordu.
- **Karar:** Yalnızca `V > θ` olan adaylar arasından en güçlü en fazla `k` nöron ateşler.

### KR-003 — Okuma vektörü `r = [f, v]`
- **Bağlam:** Uzun τ'da bellek spike'larda değil zar potansiyelinde duruyor; yalnızca `f` okuyan bir okuma onu göremiyordu (τ=2000, d=100: R² 0,07 vs 0,80).
- **Karar:** Tüm okuma blokları `[f, v]` (`[B, 2N]`) okur.

### KR-004 — τ aralığı 5–100 ms → 20–2000 ms
- **Bağlam:** Bir ortam adımı 20 ms. Eski aralıkta bellek en fazla 1–2 adım olabilirdi.

### KR-005 — Sabit eşik yerine θ(τ) tablosu (geçici)
- **Bağlam:** KR-001'den sonra büyük τ'da potansiyel çok birikiyor, ateşleme oranı τ ile değişiyordu (K7 ihlali).
- **Karar:** θ her τ için bir kez kalibre edilir ve sabit tablo olarak saklanır.
- **Durum:** Hâlâ "geçici" (ARCHITECTURE.md §3.2). Faz 3'te bu kalibrasyonun gerçek veriyle ilişkisi tekrar sorgulandı, bkz. KR-014.

### KR-006 — Nöron başına zaman sabiti `τ_i = τ·m_i` ve denge katsayısı `c_i`
- **Bağlam:** Tüm nöronlar aynı τ'yu kullanınca bellek her τ'da ~2 adımda bitiyordu; aynı filtreyi paylaşan nöronlardan gecikmeler ayrıştırılamıyor.
- **Karar:** `m_i` log-düzgün, `S = 10`; `c_i` sabit girdide tüm nöronların aynı seviyeye çıkmasını sağlar.

### KR-007 — Yumuşak sıfırlama `V ← V − θ`
- **Bağlam:** `V ← 0`, aktif nöronların belleğini her ~1,4 ortam adımında siliyordu.

### KR-008 — Rekürrent kazanç `γ_rec`
- **Bağlam:** Seyrek ikili spike'larla `W_rec` katkısı eşiğe göre çok küçüktü.
- **Durum:** Ayar seed'inde 1/3/10 arasında fark gürültü düzeyinde; config'te 1,0.

### KR-009 — Kodlayıcıya seviye (termometre) kanalları eklendi
- **Bağlam:** Delta kodlama yalnızca değişim yönünü taşıyor; ideal bir hafızayla bile pencere ortalaması geri çıkarılamıyordu (R² ≈ 0).
- **Karar:** `SpikeEncoder` = delta (2·D_in) + seviye (L·D_in) kanalları.

---

## Faz 2 — Beyincik (2026-10-28)

### KR-010 — Beyincik varsayılan kuralı: normalize çevrimiçi LMS
- **Bağlam:** §3.3'teki ham delta kuralı `ΔW = η·e·φᵀ`, `‖φ‖² ~ 1e6` olduğu için 5 seed'in hepsinde ıraksadı (NaN). Sorun veri setinde değil, `[f, v]` vektörünün ölçeğinde.
- **Karar:** Girdi her öznitelik için geçmişin üstel ortalaması/varyansıyla standartlaştırılır (nedensel), sabit bias `b = √F` eklenir, adım `‖φ̃‖²`'ye bölünür.
- **Alternatifler:** Çok küçük η (≈1e-7) — öğrenme fiilen durur. Global sabit ölçekleme — τ ve veri değişince bozulur.
- **Sonuç:** Mackey-Glass MSE = "tekrarla" tabanının %17,9'u, ridge'e göre fark %6,8. ARCHITECTURE.md §3.3'e işlendi.

---

## Faz 3 — Karar bloğu, primat motor görevi (2026-10-29)

### KR-011 — Karar bloğu da normalize LMS kullanır; ortak yardımcı
- **Bağlam:** `ΔW_a = η·δ·rᵀ` aynı ölçek sorunuyla karşılaşır.
- **Karar:** `src/utils/online_norm.py` (`OnlineStandardizer`, `nlms_step`); beyincik ve karar bloğu aynı kodu kullanır. Beyincik davranışı birebir korundu (testler yeşil).
- **Durum:** ARCHITECTURE.md §3.6'ya işlenecek (plan onayıyla).

### KR-012 — Veri: 20 ms örtüşmesiz kutular, NeuroBench bölmesi
- **Bağlam:** NeuroBench varsayılanı 28 ms pencere, 4 ms adım. Bizim bir ortam adımımız `T·dt = 20 ms`.
- **Karar:** `bin_width = stride = 0.02 s`. Her ortam adımı 20 ms'lik sayım vektörü (değerler 0–5). Bölme NeuroBench'inkiyle aynı (segmentlerin %80 train, %10 val, %10 test, zaman sırasıyla).
- **Alternatifler:** NeuroBench'in 4 ms adımı → rezervuar her 4 ms'de 20 iç adım koşardı, zaman ölçeği (K4, K6) bozulurdu.
- **Sonuç:** R², NeuroBench'in `R2` sınıfıyla bizim 20 ms ızgaramızda hesaplanır. NeuroBench taban çizgileri 4 ms ızgarada raporlanır; bu fark tabloda not edilir.

### KR-013 — Kanal başına seviye ölçeği (`range: train_per_channel`)
- **Bağlam:** Sinir kanallarının ateşleme oranları çok farklı. Tek global aralıkta sakin kanalların seviye bitleri hiç yanmaz, yoğun kanallar erken doyar.
- **Karar:** Her kanal, train bölmesindeki sayımlarının 99. yüzdeliğine bölünür; kodlayıcı aralığı `[0, 1]`. Ölçek yalnızca train'den.
- **Durum:** ARCHITECTURE.md §3.1'e işlenecek (plan onayıyla).

### KR-014 — θ(τ) kalibrasyon girdisi ve K7 (karar: kural aynı kalır)
- **Bağlam:** θ(τ), düzgün rastgele girdiyle `U(0, 1)` kalibre ediliyor. Gerçek sinir verisi çok daha seyrek (ortalama ölçekli değer ~0,1–0,2), bu yüzden `indy_20170131_02`'de ateşleme oranı %1,0–1,1 çıktı: K7 alt sınırı (ρ/2 = %2,5) ihlal.
- **Teşhis** (`indy_20170131_02`, ayar seed'i 100, ridge val R²):

  | Varyant | Val R² | Oran | K7 |
  |---|---|---|---|
  | Ham sayım x(t) | 0,111 | — | — |
  | Son 10 adım sayımı | 0,510 | — | — |
  | Düzgün kalibrasyon (§3.2) | 0,516 | %1,03 | ✗ |
  | Train ile kalibrasyon | 0,526 | %3,31 | ✓ |
  | + `c_in=10` / `30` | 0,518 / 0,530 | %3,3 | ✓ |
  | + `L=4` / `L=2` | 0,510 / 0,505 | %3,2 | ✓ |
  | + gecikme 2 adım | 0,544 | %3,3 | ✓ |
- **Karar (kullanıcı, 2026-10-29):** Kural aynı kalır — θ(τ) düzgün rastgele girdiyle kalibre edilir (§3.2 değişmez). K7 ihlali her sonuç tablosunda açıkça not edilir. `reservoir.calibration.source: train` seçeneği kodda duruyor ama kullanılmıyor.
- **Ek sonuç:** `c_in` ve `L` değişince fark gürültü düzeyinde → varsayılanlar (3, 8) korunur. Kanal sayısının (960–1920) rezervuarı zorladığı endişesi doğrulanmadı.

### KR-015 — Adım adım akış CPU'da, paralel süreçlerle
- **Bağlam:** B = 1, adım başına 20 iç adım. GPU'da çekirdek başlatma yükü baskın: 9,6 ms/adım; CPU tek iş parçacığı 4,1 ms/adım. Çok iş parçacığı yardımcı olmuyor (4,6–4,9 ms).
- **Karar:** `device: cpu`, her (oturum, seed) işi ayrı bir süreçte, 1 iş parçacığıyla (`--workers`).

### KR-016 — r(t) bir kez üretilir, A ve B aynı akışı okur
- **Bağlam:** Regülatörler kapalıyken (K5) g ve τ sabit, rezervuar sabit (K1): r(t) yalnızca x'e bağlı, öğrenmeye bağlı değil.
- **Karar:** Her (oturum, seed) için r bir kez hesaplanır; A ve B ajanları aynı r'yi okur. Bu, Faz 5'te regülatörler açılınca geçerliliğini yitirir (τ ve g öğrenmeye bağlı hale gelir).

### KR-018 — NeuroBench ile bölme farkı
- **Bağlam:** NeuroBench makalesi "ilk %75 train+val, son %25 test" diyor; NeuroBench kodunun `PrimateReaching` varsayılanı ise `train_ratio = 0.8` (segmentlerin %80 train, kalanın yarısı val, yarısı test).
- **Karar:** Kodun varsayılanı kullanılır (val bölmesi ayar için gerekli). Taban çizgisi karşılaştırması bu fark not edilerek yapılır; sayılar birebir aynı protokolde değildir.

### KR-020 — Faz 3 okuma ayarı (yalnızca val, ayar seed'i 100)
- **Bağlam:** İlk koşuda yerel kuralın test R²'si 0,334, ridge 0,493 idi. Ayar `indy_20170131_02` + `loco_20170215_02` val bölmelerinin ortalamasıyla yapıldı (`experiments/tune_nhp.py`).
- **Sonuç tablosu** (ortalama val R²):

  | Ayar | Val R² |
  |---|---|
  | 1 tur, `norm_rate` 0,01 (başlangıç) | 0,267 |
  | 3 tur, `norm_rate` 0,001 | 0,342 |
  | 12 tur, `norm_rate` 0,0003 | 0,381 |
  | 24 tur, `norm_rate` 0,0003 | 0,384 |
  | 48 tur, `norm_rate` 0,0003 | 0,378 |
  | η = 0,03 (en iyi ayar, 3 tur) | 0,355 |
  | η = 0,03 (24 tur) | 0,357 |
  | biyolojik gecikme 2 / 4 adım | +0,002 / −0,004 |
- **Karar:** `nhp.epochs: 12`, `decision.norm_rate: 0.0003`. η, K4 varsayılanı 0,01'de kalır (çok turda zaten en iyisi). Gecikme 0.
- **Not:** Ridge ile arada hâlâ ~0,14 fark var. NLMS, ilişkili özniteliklerde yavaş yakınsıyor; tur sayısı bunu kısmen kapatıyor.

### KR-017 — Değerlendirmede normalizasyon istatistikleri de donar
- **Bağlam:** Faz 2'de test sırasında ağırlıklar donuk ama standartlaştırma istatistikleri güncellenmeye devam ediyordu (etiketsiz uyum).
- **Karar:** `agent.freeze_norm_eval: true` — val/test'te hem ağırlıklar hem istatistikler donuk. Böylece "donuk okuma" tam anlamıyla donuktur. Çevrimiçi uyum Faz 6'da ayrıca açılır.

---

## Faz 4 — Duygu uzayı (2026-10-29)

### KR-019 — Duygu uzayının uygulama ayrıntıları (ARCHITECTURE.md §3.4)
- **Bağlam:** §3.4 yalnızca `z = tanh(W_z·r)`, üç baş ve "delta kuralı" diyor. q = −‖δ‖², u = ‖e‖² ve s'nin ölçekleri birbirinden çok farklı (u yüzlerce, s ±birkaç). Ham r ile delta kuralı Faz 2'deki gibi ıraksar.
- **Karar:**
  - r, karar ve beyincikle aynı nedensel standartlaştırmadan geçer.
  - Her hedef çevrimiçi standartlaştırılır; başlar standart hedefi tahmin eder.
  - Başlar ve W_z normalize LMS ile güncellenir. W_z'ye hata yalnızca W_h üzerinden gelir: `((W_hᵀ ε_h) ⊙ (1−z²))`.
  - `lr_z` ve `lr_head` config'ten gelir; η regülatörüne bağlı değildir (K4).
  - W_z kendi seed'li üreteciyle başlatılır. Böylece duygu uzayını açmak diğer blokların sonuçlarını bit düzeyinde değiştirmez (sızıntı testi).
- **Ek (isteğe bağlı, varsayılan kapalı):** `preact_l2` (z ön-aktivasyonuna L2 cezası) ve `head_weights` (başların W_z'ye dönen hatadaki ağırlığı). Sebep: ayar koşusunda z'nin %36–68'i ±1'e doymuştu (1 − z² ≈ 0 → W_z öğrenemiyor) ve Loco'da u başı öğrenmiyordu.
- **Sonuç:** Seçim KR-021'de.

### KR-021 — Duygu uzayı ayarı: `lr_z = 0,03` + ön-aktivasyon L2 `λ = 0,03`
- **Bağlam:** İlk ayarda (`lr_head` 0,1, 12 tur) `lr_z = 0,01` üç başı da öğretiyor ama AUC düşük (0,72 / 0,80); `lr_z = 0,1` Loco'da u başını bozuyor. Her ayarda z'nin %36–68'i doymuş (`1 − z² ≈ 0` → W_z'ye gradyan gitmiyor). Kullanıcı L2 cezası ve u ağırlığıyla bir doğrulama koşusu istedi; başarısız olursa `lr_z = 0,01`'e dönülecekti.
- **Doğrulama** (`scripts/tune_affect.py`, yalnızca val, seed 100, `indy_20170131_02` / `loco_20170215_02`):

  | `lr_z` | λ | u ağırlığı | AUC (Indy / Loco) | Doygun | Üç baş düşüyor |
  |---|---|---|---|---|---|
  | 0,01 | 0 | 1 | 0,72 / 0,80 | %53 / %58 | ✓ / ✓ |
  | 0,03 | 0 | 1 | 0,825 / 0,832 | %60 / %68 | ✓ / ✗ (u) |
  | 0,03 | 0 | 3 | 0,814 / 0,825 | %54 / %64 | ✓ / ✓ |
  | **0,03** | **0,03** | **1** | **0,807 / 0,814** | **%3 / %1** | **✓ / ✓** |
  | 0,03 | 0,03 | 3 | 0,799 / 0,799 | %1 / %2 | ✓ / ✓ |
  | 0,03 | 0,3 | 1 | 0,749 / 0,747 | %0 / %0 | ✓ / ✓ |
  | 0,1 | 0 | 1 | 0,833 / 0,819 | %36 / %62 | ✓ / ✗ (u) |
  | 0,1 | 0,03 | 1 | 0,739 / 0,747 | %2 / %1 | ✓ / ✓ |
  | 0,1 | 0,3 | 1 | 0,686 / 0,526 | %0 / %0 | ✓ / ✓ |
- **Karar:** `affect.lr_z: 0.03`, `affect.preact_l2: 0.03`, `head_weights: [1, 1, 1]`. Üç baş iki oturumda da öğreniyor, doygunluk ~%2, AUC 0,81 (`lr_z = 0,01` yedeğine göre +0,05). ARCHITECTURE.md §3.4 ve §11 güncellendi.
- **Açık:** Loco'da u başı zayıf kalıyor (MSE 0,81 → 0,80; ortalama tahmini ≈ 1). u hedefi, beyincik öğrendikçe kayıyor olabilir. u ağırlığı 3 bunu belirgin düzeltmedi. Faz 5'te belirsizlik regülatörü için not.
