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

---

## Faz 5 — Regülatörler (2026-10-31)

### KR-023 — Regülatörlerin uygulama ayrıntıları (ARCHITECTURE.md §3.5)
- **Bağlam:** §3.5 formülü ve perturbasyonun üç adımını veriyor. Ama başlangıç değerleri, "öncekiyle karşılaştır"ın neyle yapılacağı ve batch'te η/β'nın nasıl kullanılacağı açık değil.
- **Karar** (`src/affect.py` `Regulators`, `src/agent.py` adım 2 ve 10):
  - `W_r = 0`, `b_r = logit(varsayılanın konumu)` → başlangıçta çıktı tam olarak varsayılanlar.
  - Perturbasyon dönüşümlü pencerelerle yapılır: M adım referans (mevcut W_r) → `W_r + N(0, σ²)` ile M adım deneme → deneme penceresinin ortalama q'su referanstan büyükse tutulur, değilse geri alınır. Referans her denemeden hemen önce yeniden ölçülür. Sebep: veri kayarken eski bir referansla karşılaştırmak, zorlaşan veride her denemeyi reddeder, kolaylaşan veride her denemeyi kabul eder.
  - Yalnızca `W_r` perturbe edilir (§3.5'teki gibi), `b_r` edilmez. Gürültü kendi seed'li üretecinden gelir (`perturbation.seed_offset`).
  - g ve τ örnek başınadır (`[B]`). η ve β batch ortalamasıdır, çünkü güncellemeler zaten batch üzerinden ortalanıyor.
  - Konfigürasyon C = B + duygu uzayı + regülatörler. `regulators.enabled: false` iken C, B+Z ile birebir aynıdır (K5, test edildi).
  - Değerlendirmeye geçerken (`set_eval`) yarım kalan deneme geri alınır, regülatörler donar. Ancak z'yi okumaya devam ederler.
  - Elle eşleme (`mode: hand`, PHASES.md ara adımı): η satırı `+c·w_u`, τ satırı `−c·w_u`. Burada `w_u`, duygu uzayının u başının z ağırlıklarıdır. Yani "belirsizlik yüksekse η artar, τ kısalır". g ve β varsayılanda kalır.
  - `BatchedAgent` regülatörleri desteklemez. C'de r öğrenmeye bağlıdır, önceden üretilemez; her ajan kendi rezervuar kopyasıyla adım adım akar.
- **Durum:** Uygulandı. Testler: `tests/test_regulators.py` (12), `tests/test_agent.py` (Faz 5 bölümü), `tests/test_invariants.py` (K4, K5). ARCHITECTURE.md §3.5'e işlenmesi kullanıcı onayı bekliyor.

### KR-024 — İlk deneme koşusu: mekanizma çalışıyor, iki sorun var (açık)
- **Koşu:** `experiments/regulator_trial.py`, `indy_20170131_02` (20 000 adıma kısaltılmış), ayar seed'i 100, 1 tur, yalnızca val. Log: `runs/20261031-200842_nhp_motor_regtrial_seed100`.

  | Konfigürasyon | R² (val) | τ aralığı (ms) | η aralığı | g aralığı | Sınırda | Deneme / kabul |
  |---|---|---|---|---|---|---|
  | B+Z | 0,214 | 100 (sabit) | 0,01 (sabit) | 1 (sabit) | — | — |
  | C:hand (c = 1) | 0,166 | 35–452 | 0,0028–0,031 | 1 (sabit) | %0 | — |
  | C:perturb (σ = 0,01, M = 500) | 0,216 | 92–108 | 0,0095–0,0106 | 0,97–1,04 | %0 | 15 / 9 |
- **Sorun 1 — perturbasyon çok küçük adım atıyor:** Varsayılan σ = 0,01 ile değerler varsayılanın ±%5'inde kalıyor. Kabul oranı %60, yani rastgeleye yakın. 500 adımlık q ortalamasındaki veri gürültüsü, perturbasyonun etkisinden büyük olabilir. Çıkış kriteri 1 ("anlamlı biçimde değişiyor") bu ayarla sağlanmıyor.
- **Sorun 2 — değerler adım adım titriyor:** Hem elle eşlemede hem perturbasyonda g/τ/η her adımda büyük sıçrıyor, çünkü z(t) anlık bir durumdur. Elle eşlemede τ'nun hızlı değişmesi okumanın gördüğü r dağılımını sürekli kaydırıyor. R²'deki düşüş (0,214 → 0,166) muhtemelen bundan.
- **Sonraki adım (karar bekliyor):** (a) val/ayar seed'inde σ ∈ {0,01; 0,1; 0,3} × M ∈ {500; 2000} ızgarası; (b) regülatör girdisinin yavaşlatılması (ör. z'nin EMA'sı). (b) §3.5'i değiştirir, önce teşhis tablosu ve kullanıcı onayı gerekir.

### KR-025 — Perturbasyon ızgarası (σ × M): öneri σ = 0,1, M = 500 (onay bekliyor)
- **Koşu:** `experiments/regulator_trial.py`, `indy_20170131_02` (20 000 adım), ayar seed'i 100, **3 tur**, yalnızca val. 7 paralel süreç, her biri ~17–18 dk. Loglar: `runs/20261031-203910_grid_*`.

  | σ | M | R² (val) | g aralığı | τ aralığı (ms) | η aralığı | Sınırda (en kötü) | Deneme / kabul |
  |---|---|---|---|---|---|---|---|
  | — (B+Z) | — | 0,322 | 1 | 100 | 0,01 | — | — |
  | 0,01 | 500 | 0,325 | 0,93–1,07 | 90–115 | 0,0086–0,011 | %0 | 47 / 28 |
  | 0,01 | 2000 | 0,321 | 0,98–1,02 | 93–107 | 0,0094–0,011 | %0 | 11 / 5 |
  | **0,1** | **500** | **0,330** | **0,62–1,65** | **39–412** | **0,0016–0,038** | **%0** | **47 / 26** |
  | 0,1 | 2000 | 0,305 | 0,87–1,21 | 54–209 | 0,0051–0,018 | %0 | 11 / 5 |
  | 0,3 | 500 | 0,212 | 0,50–2,00 | 21–1753 | 0,0001–0,099 | %7 (g, η) | 47 / 29 |
  | 0,3 | 2000 | 0,273 | 0,68–1,60 | 28–764 | 0,0014–0,046 | %0 | 11 / 5 |
- **Okuma:**
  - σ = 0,01 her M'de değerleri varsayılanın ±%15'inde tutuyor → çıkış kriteri 1 sağlanmıyor.
  - σ = 0,3 aşırı: M = 500'de g ve η zamanın ~%7'sinde sınırda, R² 0,11 düşüyor.
  - σ = 0,1, M = 500 iki kriteri de karşılıyor: değerler geniş aralıkta oynuyor, sınıra yapışmıyor, R² B+Z'den kötü değil (+0,008).
  - **Dikkat:** Tek seed ve tek kısaltılmış oturum; ±0,01'lik R² farkları büyük olasılıkla gürültü düzeyinde. Kabul oranı her ayarda %45–60, yani perturbasyonun q üzerindeki etkisi pencere gürültüsünden pek ayrışmıyor. "Regülatörler iyileştiriyor" iddiası buradan çıkmaz. Yalnızca "zarar vermeden anlamlı değişiyor" denebilir.
  - M = 2000'de 3 turda yalnızca 11 deneme var; bu kısa oturumda adil bir karşılaştırma değil.
- **Öneri:** `regulators.perturbation.noise_std: 0.1`, `M: 500`. Doğrulama için ikinci bir oturumda (ör. `loco_20170215_02`, KR-021'deki gibi) aynı karşılaştırma.
- **Durum:** Config henüz değişmedi, kullanıcı onayı bekliyor. Adım adım titreme sorunu (KR-024, sorun 2) açık.

### KR-026 — E/M/R duygu dinamiği; regülatörler z yerine E'yi okur (ARCHITECTURE.md §3.5)
- **Bağlam:** KR-024'teki titreme sorunu: regülatörler anlık z'yi okuyor, g/τ/η her 20 ms'de sıçrıyor. Kullanıcı seyrek güncelleme, basit EMA ve hız sınırı yerine literatüre dayanan bir dinamik istedi: DynAffect (homeostatik geri çekilme), WASABI (duygu–duygudurum çift zaman ölçeği), CoMERG/Gross (karşıt süreç). Formüller ve varsayılan değerler kullanıcıdan geldi.
- **Karar:**
  - `src/affect.py` `AffectDynamics`: E (fazik duygu), M (tonik duygudurum), R (karşıt süreç). Formüller §3.5'te. Config `affect.dynamics` altında.
  - Ajan adım 9'dan sonra `E = dynamics(z)` hesaplar; regülatörler E_{t−1}'i okur. `dynamics.enabled: false` iken z okunur (eski davranış).
  - **Girdi merkezleme (kullanıcı onayıyla eklendi):** `z_in = z − μ_z`, `center_rate = 0,001` (~20 s).
  - İsim ayrımı: kullanıcının beta/theta/M adları projede başka anlamlarda kullanıldığından config'te `mood_rate`, `threshold` adları kullanıldı. Perturbasyon penceresi M, W_r'nin öğrenme protokolü olarak kalıyor; regülatör çıktıları her adım güncelleniyor.
- **Çevrimdışı tekrar** (`experiments/affect_dynamics_replay.py`): Faz 4 tam koşusunun 30 z izi (son train turu, döngü açık). Log: `runs/20261101-161333_affect_dynamics_replay_seeds0-1-2-3-4`.

  | Oturum | \|ΔE\|/\|Δz\| | Doygun (merkezsiz) | Doygun (merkezli) | En kötü boyut (merkezli) | R aktif |
  |---|---|---|---|---|---|
  | indy_20170131_02 | 0,21 | %73 | %9 | %23 | %81 |
  | indy_20160630_01 | 0,24 | %51 | %11 | %28 | %83 |
  | indy_20160622_01 | 0,25 | %68 | %4 | %12 | %87 |
  | loco_20170301_05 | 0,20 | %71 | %5 | %10 | %93 |
  | loco_20170215_02 | 0,20 | %69 | %2 | %4 | %95 |
  | loco_20170210_03 | 0,22 | %80 | %6 | %14 | %87 |

- **Okuma:**
  - Titreme her oturumda ~4–5 kat azalıyor.
  - Merkezleme şart: merkezlemesiz E zamanın %51–80'inde doygun.
  - Merkezlemeyle de ortalaması çok kayık boyutta (ör. Indy'de ortalama −0,76) E zamanın ~%23'ünde −1 civarında. Sebep `(1−|E|)` çarpanının asimetrisi: z çoğunlukla hafif negatif, arada büyük pozitif sıçramalar var; sıçramalar geldiğinde E zaten −1'e yakın ve etkisi kısılıyor.
  - **R zamanın %81–95'inde devrede.** θ_R = 0,15, E'nin yayılımına (std ~0,4) göre küçük. "Olay güdümlü" olması beklenen regülatör neredeyse sürekli çalışıyor. θ_R ayarı açık soru.
- **Durum:** Uygulandı, testler geçiyor (`tests/test_affect_dynamics.py`, ajan testleri). Kapalı döngü sonucu KR-027'de.

### KR-027 — Kapalı döngü E/M/R denemesi: C her iki sürümde de bir oturumda kötüleşiyor (açık)
- **Koşu:** `experiments/regulator_trial.py`, iki oturum (20 000 adım), ayar seed'i 100, 3 tur, yalnızca val, σ = 0,1, M = 500. 6 paralel süreç, ~16 dk. Loglar: `runs/20261101-161716_emr_*`.

  | Oturum | B+Z | C:perturb (z) | C:perturb+EMR (E) | E doygun | \|ΔE\| |
  |---|---|---|---|---|---|
  | indy_20170131_02 | 0,322 | **0,330** | 0,219 | %2,2 | 0,065 |
  | loco_20170215_02 | 0,139 | 0,035 | **0,123** | %4,3 | 0,060 |

  Regülatör aralıkları (EMR): g 0,62–1,57, τ 46–257 ms, η 0,0024–0,033. Sınırda %0.
- **Okuma:**
  - KR-025'teki σ = 0,1 seçimi ikinci oturumda **doğrulanmadı**: Loco'da C:perturb, B+Z'den 0,10 kötü.
  - E/M/R, Loco'daki kaybı büyük ölçüde kapatıyor (−0,016), ama Indy'de 0,10 kaybettiriyor. İki sürüm de çıkış kriteri 2'yi ("C, A ve B'den kötü değil") en az bir oturumda sağlamıyor.
  - Kabul oranı her yerde %53–62, yani rastgeleye yakın. Perturbasyon, W_r'yi yönlü öğrenmekten çok rastgele yürütüyor. Sonuç, yürüyüşün nereye vardığına bağlı. Tek seed'le bu, gürültüden ayırt edilemez.
  - Grafikte E'yi okuyan regülatörler ham z'ye göre daha sakin, ama hâlâ onlarca adımlık salınımlar var (E'nin kendi salınımı, KR-026 grafiği).
- **Sonuç:** Faz 5 tam koşusu (plan basamak 6) bu haliyle başlatılmamalı. Önce farkların seed gürültüsü mü yoksa sistematik mi olduğu ölçülmeli.
- **Seçenekler (karar bekliyor):**
  - (a) Aynı denemeyi 3 seed'le tekrarlamak (~25–30 dk, 16 çekirdek): farklar gürültü mü?
  - (b) Perturbasyon sinyalinin gürültüsünü azaltmak: deneme penceresinin q'su, aynı veriyi gören regülatörsüz bir gölge ajanın q'suna göre karşılaştırılır (eşli karşılaştırma). §3.5 değişikliği, onay gerekir.
  - (c) E/M/R ayarları (θ_R, α_up): R zamanın %81–95'inde devrede (KR-026).

### KR-028 — 3 ayar seed'i: z okuyan C Loco'da sistematik kötü, E/M/R zararı azaltıyor ama iyileştirmiyor
- **Koşu:** KR-027 ile aynı protokol, ayar seed'leri 100, 101, 102 (rapor seed'leri 0–4 değil). 12 ek süreç, ~25 dk. Loglar: `runs/20261101-*_emr_*_s101|s102_regtrial_seed10x`.

  R² (val), seed 100 / 101 / 102:

  | Oturum | B+Z | C:perturb (z) | C:perturb+EMR (E) |
  |---|---|---|---|
  | indy_20170131_02 | 0,322 / 0,305 / 0,277 | 0,330 / 0,309 / 0,266 | 0,219 / 0,294 / 0,294 |
  | loco_20170215_02 | 0,139 / −0,028 / 0,101 | 0,035 / −0,085 / 0,029 | 0,123 / −0,088 / 0,104 |

  Eşli fark, C − B+Z (aynı seed), ortalama ± std:

  | Oturum | C:perturb (z) | C:perturb+EMR (E) |
  |---|---|---|
  | indy_20170131_02 | +0,000 ± 0,008 (2 +, 1 −) | −0,032 ± 0,052 (1 +, 2 −; seed 100 −0,10) |
  | loco_20170215_02 | **−0,078 ± 0,020 (3/3 −)** | −0,024 ± 0,026 (1 +, 2 −) |

- **Okuma:**
  - z okuyan C, Loco'da üç seed'in üçünde B+Z'den 0,06–0,10 kötü. Bu gürültü değil, sistematik.
  - E/M/R bu zararı ~üçte birine indiriyor, ama hiçbir oturumda iyileştirme yok. Indy'deki ortalama tek bir kötü seed'den geliyor.
  - Muhtemel mekanizma: kabul oranı ~%50, yani perturbasyon W_r'yi rastgele yürütüyor. Okuma blokları (`norm_rate`, epochs) varsayılan τ = 100 ms, g = 1'de ayarlandı. Rastgele yürüyüş sistemi bu ayarlanmış çalışma noktasından uzaklaştırıyor ve ortalamada bedel ödetiyor. Loco'nun daha hassas olması bununla uyumlu.
  - Kısaltılmış Loco val'i çok gürültülü (B+Z seed'ler arası −0,03 ile 0,14). Mutlak sayılar değil, yalnızca eşli farklar anlamlı.
  - E doygunluğu bazı seed'lerde %17–18'e çıkıyor (seed 100'de %2–4).
- **Sonuç:** Faz 5 çıkış kriteri 2 ("C, A ve B'den kötü değil") hiçbir sürümde güvenle sağlanmıyor. Tam koşu bekletiliyor. Sorun dinamikten çok W_r'nin öğrenme sinyalinde: perturbasyon kararları gürültüye dayanıyor.
- **Önerilen sonraki adım:** Eşli perturbasyon karşılaştırması. Deneme ve referans pencerelerinin q'su, aynı veriyi aynı anda gören regülatörsüz bir gölge ajanın (B+Z) q'suyla farklanır: `Δq = q_C − q_gölge`. Böylece "bu pencere zaten zordu" etkisi çıkarılır. §3.5 değişikliği, kullanıcı onayı gerekir. Maliyet: C başına bir ek okuma akışı (~%40–60 daha yavaş).

### KR-029 — Eşli perturbasyon zararı azaltmadı; asıl sorun train → val rejim kayması (→ KR-030)
- **Koşu:** KR-028 ile aynı protokol, `perturbation.paired: true` (gölge A ajanı, ARCHITECTURE.md §3.5). Ayar seed'leri 100–102, 12 süreç. Loglar: `runs/20261101-18*_paired_*`. B+Z sonuçları KR-028'den (gölge yalnızca C'de var).

  C − B+Z eşli fark (seed 100 / 101 / 102):

  | Oturum | C:perturb (z), eşli | C:perturb+EMR (E), eşli | Kabul oranı |
  |---|---|---|---|
  | indy_20170131_02 | +0,009 / −0,021 / −0,069 | −0,041 / +0,005 / −0,063 | %36–49 |
  | loco_20170215_02 | −0,38 / **−4,68** / −0,13 | **−0,90** / +0,15 / −0,00 | %36–64 |

- **Okuma:**
  - Eşli karşılaştırma kabul oranını Indy'de %50'nin altına (%36–49) indirdi, yani kararlar artık tamamen rastgele değil. Ama sonuçlar iyileşmedi, Loco'da felaket koşular var.
  - **Mekanizma (izlerden doğrulandı):** Kötü koşularda regülatörler val'de train'de nadiren görülen bir rejime geçiyor:

    | Koşu | τ medyanı (train → val) | ΔR² |
    |---|---|---|
    | Loco perturb s101 | 74 → 314 ms | −4,68 |
    | Loco perturb s100 | 121 → 291 ms | −0,38 |
    | Loco EMR s100 | 130 → 347 ms | −0,90 |
    | Loco perturb s102 | 150 → 134 ms | −0,13 |
    | Loco EMR s101 | 152 → 63 ms | +0,15 |

  - Val'de karar bloğu ve normalizasyonu donuk (KR-017). Yeni veri z'yi kaydırıyor, z τ'yu kaydırıyor, τ da rezervuar çıktısını donuk okumanın hiç görmediği bir dağılıma taşıyor. **Regülatörler verideki kaymayı temsil kaymasına çevirip büyütüyor.** Araştırma sorusunun tersi yönde, önemli bir bulgu.
  - Uzun τ'ya kayma özellikle yıkıcı. Kısa τ'ya kayma bir koşuda zararsız, hatta yararlıydı.
- **Sonuç:** Faz 5 çıkış kriteri 2 sağlanmıyor. Sorun perturbasyon sinyalinin gürültüsünden çok, kapalı döngünün donuk okumayla birleşince ürettiği rejim kayması.
- **Seçenekler (karar bekliyor):**
  - (a) Val'de normalizasyonu donmuş tutmamak (`agent.freeze_norm_eval: false`): etiketsiz, nedensel istatistik takibi, ağırlıklar yine donuk. Rejim kaymasını doğrudan hedefler. KR-017 kararını C için gözden geçirir; adil olması için B+Z de aynı ayarla yeniden koşulur.
  - (b) Regülatör duyarlılığını sınırlamak: W_r'ye L2 / norm sınırı. Kaymayı büyütme kapasitesini azaltır ama regülatörü de zayıflatır.
  - (c) Faz 5'in "donuk değerlendirme" protokolünü bırakıp Faz 6'daki gibi val'de de öğrenmeyi açık tutmak. Regülatörün amacına (çevrimiçi uyum) en uygunu, ama Faz 5 çıkış kriterinin anlamını değiştirir.

### KR-030 — Nedensel intrinsik homeostatik ölçekleme + asimetrik τ: felaket kaymalar kalktı, C hâlâ B+Z'den iyi değil
- **Bağlam:** KR-029'daki temsil kayması. Kullanıcı seçimi: (a) uygulanacak, ama Faz 5'in donuk ağırlık protokolü korunacak. (b) gibi W_r köreltilmeyecek, (c)'ye (Faz 6) de geçilmeyecek. Ek olarak uzun τ asimetrik sınırlanacak ve τ'ya bağlı ölçek kayması dengelenecek.
- **Teşhis ölçümü:** Ayar seed'i 100, train'in ilk 3000 adımı, sabit τ. Değerler τ = 100'ün istatistikleriyle standartlaştırıldı.

  | τ (ms) | θ(τ) | \|V\| ort. | V std | v ortalama kayması (σ) | f'nin standart sapması (σ birimi) |
  |---|---|---|---|---|---|
  | 30 | 14,6 | 6,6 | 7,1 | 0,84 | 6,4 |
  | 100 | 18,2 | 17,5 | 14,3 | 0 | 0,7 |
  | 300 | 18,9 | 44,9 | 26,0 | 1,96 | 71 |
  | 500 | 19,0 | 71,1 | 34,4 | 3,80 | 87 |

  - Tablo Indy'ye ait. Loco'da tablo aynı, f sapması τ = 500'de 165σ.
  - θ(τ) neredeyse sabit, bu yüzden V/θ ölçeği düzeltmez.
  - Asıl yıkım f'de. τ = 100'de hiç ateşlemeyen nöronların varyansı ≈ 0. Uzun τ'da ateşlemeye başlayınca donuk normda onlarca σ'lık değerler üretiyorlar. KR-029'daki −4,68'in mekanizması bu.
- **Karar (ARCHITECTURE.md §3.2, §3.5, §8, §11):**
  1. **Sinaptik ağırlıklardan bağımsız nedensel intrinsik homeostatik ölçekleme** (`agent.freeze_norm_eval: false`, artık varsayılan):
     - Değerlendirmede tüm ağırlıklar donuk kalır: karar `W_a`, beyincik `W_c`, duygu `W_z`/`W_h` ve `W_r`. Hedef istatistikleri (q, u, s) de donuktur.
     - Yalnızca okuma bloklarının girdi standartlaştırıcıları güncellenir: karar, beyincik ve duygu uzayının r normu.
     - Güncelleme etiketsiz ve nedenseldir. μ_t ve σ²_t her adımda yalnızca r_1..r_t ile hesaplanır (üstel ortalama, ρ_n = 0,01). Batch istatistiği ve gelecek bilgisi kullanılmaz.
     - Literatür çerçevesi: Turrigiano (2008) homeostatik sinaptik ölçekleme (nöron, girdi dağılımını hedef aktivite aralığına çarpımsal olarak ölçekler) ve Carandini & Heeger (2012) kanonik bölücü normalizasyon. İkisi de öğrenilmiş ağırlık yapısını değiştirmeden girdi ölçeğini yeniden ayarlar.
     - Göz bebeği refleksi gibi etiketsiz duyusal adaptasyona benzer: ne söylendiğini değil, sinyalin ölçeğini ayarlar ve görev etiketine erişmez. Bu yüzden Faz 5'in "donuk ağırlık, etiketsiz değerlendirme" protokolüyle uyumludur. Regülatörün τ modülasyonu karar katmanını doygunluğa sokmaz.
     - Nedensellik `test_eval_norm_is_causal` ile doğrulandı: gelecekteki x değişse de geçmişteki ŷ'ler birebir aynı kalıyor. Ağırlıkların dondurulduğu `test_eval_norm_weights_frozen` ile doğrulandı.
  2. **Asimetrik τ eşlemesi** (`regulators.tau_ceiling: 1.2`):
     - a = (W_r·E)_τ.
     - Aşağı yön: log τ = log τ_0 + log(τ_0/τ_alt)·tanh(a). τ, 20 ms'lik alt sınıra kadar inebilir.
     - Yukarı yön: log τ = log τ_0 + log 1,2·tanh(a). Tavan 120 ms.
     - W_r'ye ceza ya da norm sınırı eklenmedi. g, η, β eşlemesi değişmedi. K4 aralığının içinde kalır.
     - Yu & Dayan (2005) esinli yorum: beklenmeyen belirsizlikte (dağılım kayması) eski bilgiyi hızlı unutmak (kısa τ) doğru yöndür, belleği uzatmak yanlış yöndür. KR-029'da da uzun τ'ya kayma yıkıcıydı (−4,68), kısa τ'ya kayma zararsızdı (+0,15).
     - Yukarıdaki tablodaki τ ↔ ölçek bağı da uzun τ'nun temsil ölçeğini şişirdiğini gösteriyor.
  3. **Varyans koruyucu kazanç** (`reservoir.v_gain_exp: 0.5`):
     - Okuma vektöründe v = (τ_ref/τ)^0,5 · V.
     - Yalnızca okumaya uygulanır: dinamik, spike'lar, θ(τ) ve K7 değişmez. Girdi terimini ölçeklemek V'yi ve kalibre θ'yı birlikte ölçeklerdi, spike'lar yine aynı kalırdı. Okuma açısından iki yol eşdeğerdir.
     - τ = 100'de κ = 1. Sabit τ'lu A/B/B+Z'nin r'si birebir aynı kalır.
     - p, ayar seed'inin train verisinde seçildi. p = 0,5 iken V std oranı τ ∈ [30, 500] aralığında 0,86–1,06'da kalıyor. p = 1 (tam (1−ᾱ) telafisi) ortalamayı korur ama varyansı 0,38'e indirir.
- **Testler:** 7 yeni test eklendi (asimetrik τ ×3, κ(τ) ×1, nedensel norm ×3). Toplam 121/121 geçiyor.
- **Koşu:** KR-029 ile aynı protokol: Indy + Loco, ayar seed'leri 100–102, 20 bin adım, 3 tur, σ = 0,1, M = 500, eşli perturbasyon, val'de donuk ağırlık. **B+Z de aynı nedensel normla yeniden koşuldu.** Loglar: `runs/20261102-0*_nhp_motor_regtrial_seed10*`.

  R² (val):

  | Oturum | Seed | B+Z | C:perturb (z) | C:perturb+EMR (E) |
  |---|---|---|---|---|
  | indy_20170131_02 | 100 | 0,325 | 0,325 | 0,334 |
  | | 101 | 0,306 | 0,314 | 0,322 |
  | | 102 | 0,281 | 0,272 | 0,266 |
  | loco_20170215_02 | 100 | 0,191 | 0,135 | 0,154 |
  | | 101 | 0,141 | 0,128 | 0,152 |
  | | 102 | 0,177 | 0,099 | 0,166 |

  C − B+Z eşli fark. KR-029 satırında seed başına değerler (donuk norm, simetrik τ), KR-030 satırında ortalama ± std (3 seed):

  | | Indy, z | Indy, E | Loco, z | Loco, E |
  |---|---|---|---|---|
  | KR-029 | +0,009 / −0,021 / −0,069 | −0,041 / +0,005 / −0,063 | −0,38 / −4,68 / −0,13 | −0,90 / +0,15 / −0,00 |
  | **KR-030** | −0,001 ± 0,007 | **+0,004 ± 0,013** | −0,049 ± 0,027 (3/3 kötü) | **−0,012 ± 0,020** |

  - τ medyanı (train → val): en kötü durum 74 → 314 ms idi. Şimdi val'de τ en çok 118 ms.
  - Val medyanları 47–107 ms arasında; sistem kısa τ tarafını kullanıyor (ör. Loco E s100: 100 → 60 ms).
  - Kabul oranı %34–53.
- **Okuma:**
  - Felaket koşular kalktı: en kötü fark −4,68'den −0,078'e indi. Temsil kayması kapandı.
  - Nedensel norm B+Z'yi de iyileştirdi (seed 100: Loco 0,139 → 0,191, Indy 0,322 → 0,325). Yani karşılaştırma çıtası da yükseldi.
  - E okuyan C (E/M/R), iki oturumda da B+Z'den ayırt edilemiyor: Indy +0,004 ± 0,013, Loco −0,012 ± 0,020.
  - z okuyan C, Loco'da hâlâ 3 seed'in 3'ünde kötü (−0,049). Bu kurulumda E/M/R yumuşatması gerekli görünüyor.
  - Faz 5 çıkış kriteri 2 ("C, A ve B'den kötü değil") val'de E sürümü için sağlanıyor görünüyor, z sürümü için sağlanmıyor. Hiçbir sürümde üstünlük yok.
- **Uyarılar:**
  - Üç değişiklik birlikte uygulandı ve katkıları ayrıştırılmadı. Gerekirse tek tek kapatarak ablation yapılabilir: `tau_ceiling: null`, `v_gain_exp: 0`.
  - Tavan (1,2) ve p (0,5), KR-029'daki val bulgusuna bakılarak seçildi. Test bölmesine bakılmadı. Rapor seed'lerinde (0–4) ve test bölmesinde doğrulanmadan sonuç iddia edilmez.
  - K7: tüm konfigürasyonlarda ateşleme oranı 0,011–0,020, yani ρ/2 = 0,025'in altında. Bu KR-014'teki bilinen ihlal; B+Z'de de aynı.
  - 4 EMR koşusu gece uykusu yüzünden duvar saatinde ~9 saat sürdü. Sonuçlar bundan etkilenmedi.
- **Faz 5 kapanışı (2026-11-02, kullanıcı kararı):**
  - **Çözülen:** τ modülasyonunun yarattığı temsil patlaması kapandı. Mekanizma: nedensel normalizasyon (Turrigiano 2008, Carandini & Heeger 2012) ve κ(τ) = (100/τ)^0,5 varyans dengelemesi. En kötü C − B+Z farkı −4,68'den −0,078'e indi. 3 ayar seed'inde E okuyan C − B+Z: Indy +0,004 ± 0,013, Loco −0,012 ± 0,020. z okuyan C: Indy −0,001 ± 0,007, Loco −0,049 ± 0,027.
  - **Çözülmeyen:** C, B+Z'den iyi değil. Protokol, val'de tüm okuma ağırlıklarını (W_a) donuk tutuyor.
  - **Hipotez (test edilmedi):** τ değişimi rezervuarın zamansal temsilini kaydırıyor (gecikme/faz dönmesi). Donuk okuma hiperdüzlemi bu kaymayı izleyemiyor. Bu durumda τ regülasyonu ancak okuma da çevrimiçi uyum sağlarsa avantaja dönüşebilir.
  - Bu hipotez Faz 5 verisiyle kanıtlanmış değil, yalnızca sonucun olası açıklaması. Faz 6'da (KR-031) donuk / sabit η / η_t okuma karşılaştırmasıyla sınanacak.
  - **Açık:** `experiments/faz5_full.py` (rapor seed'leri 0–4, test bölmesi) yazıldı ve smoke koşusu geçti, ama tam koşu yapılmadı. Faz 5 sonuçları ayar seed'leri ve val ile sınırlı.

---

## Faz 6 — Çevrimiçi okuma adaptasyonu (2026-11-02)

### KR-031 — Hızlı-yavaş okuma + prequential NLMS + η_t kapısı: uyum her konfigürasyona yardım ediyor; eşlenmiş η'da η_t ve C'nin katkısı yok
- **Bağlam:** KR-030 sonrası hipotez: donuk okuma, τ modülasyonunun yarattığı temsil kaymasını izleyemiyor. Kullanıcı spesifikasyonu (ADIM 1–4) uygulandı (ARCHITECTURE.md §3.2, §3.6, §8, §11):
  - W = W_base + ΔW: W_base train sonunda donar, ΔW_0 = 0.
  - Prequential akış: ŷ_t skorlanır, sonra e_t ile ΔW güncellenir.
  - Güncelleme: elastik çapalı NLMS. ΔW ← (1 − λ_pull)·ΔW + η_t·e·φ̃ᵀ/(‖φ̃‖² + ε).
  - Norm sınırı: ‖ΔW‖_F ≤ ρ‖W_base‖_F.
  - Üç faktörlü kapı: η_t = η_base + η_mod·min(1, |log(τ_t/τ_0)|/log 2).
  - Opsiyonel çapa alt-popülasyonu: %20 nöron τ = 100'de sabit.
- **Değerler** (önceden seçildi, val'de ayarlanmadı): η_base = 1e-3 (train η'sının 1/10'u), η_mod = 9e-3, λ_pull = 1e-3 (yarı ömür ~700 adım), ρ = 0,25, d = 1.
- **Testler** (`tests/test_faz6.py`, 14 test; toplam 139/139):
  - Sızıntı yok: y_t değişince ŷ_t ve ŷ_{t+d−1} birebir aynı, ilk değişen ŷ_{t+d} (d = 1, 2).
  - Aşırı girdi (1e6) ve hata (1e8) altında ‖ΔW‖ hiçbir adımda sınırı aşmıyor. Girdi kesilince ΔW → 0 ve W_base hiç değişmiyor.
  - η = 0 (fixed ve modulated): B+Z ve C'de tahminler, τ izi ve spike sayıları Faz 5 ile bit düzeyinde aynı.
  - Ek testler: ΔW'nin τ izini değiştirmediği, kapı değerleri ve çapa maskesinin K1 dizilimini bozmadığı.
- **Koşu:** `experiments/faz6_trial.py`, `runs/20261102-112732_nhp_motor_faz6_seeds100-101-102`.
  - KR-029/030 protokolü: 20 bin adım, 3 tur train, val. Test bölmesine bakılmadı.
  - Her (oturum, seed) için B+Z, C ve C+çapa bir kez eğitildi. Val başında varyantlara kopyalandı. Bu geçerli, çünkü val'de ΔW τ izini değiştirmiyor (testli).
  - Kontrol: donuk varyantlar KR-030 değerlerini tekrar üretiyor (B+Z 0,325 / 0,306 / 0,281; C Loco 0,154 / 0,152 / 0,166).

  R² (val), ortalama ± std (3 seed):

  | Oturum | B+Z F5 donuk | B+Z F6 sabit η | C F5 donuk | C F6 sabit η | C F6 η_t | C+çapa F6 η_t |
  |---|---|---|---|---|---|---|
  | indy_20170131_02 | 0,304 ± 0,018 | 0,308 ± 0,015 | 0,307 ± 0,030 | 0,313 ± 0,026 | 0,317 ± 0,026 | 0,318 ± 0,024 |
  | loco_20170215_02 | 0,169 ± 0,021 | 0,192 ± 0,014 | 0,157 ± 0,006 | 0,172 ± 0,010 | 0,189 ± 0,013 | 0,206 ± 0,007 |

  Eşli farklar (kötü seed / 3):

  | Karşılaştırma | Indy | Loco |
  |---|---|---|
  | B+Z: sabit η − donuk | +0,004 ± 0,003 (0) | +0,023 ± 0,007 (0) |
  | C: sabit η − donuk | +0,005 ± 0,004 (0) | +0,015 ± 0,004 (0) |
  | C η_t − C sabit η | +0,005 ± 0,001 (0) | +0,017 ± 0,005 (0) |
  | C+çapa η_t − C η_t | +0,001 ± 0,004 (1) | +0,016 ± 0,013 (0) |
  | **C η_t − B+Z F5 donuk** (istenen ana karşılaştırma) | +0,014 ± 0,009 (0) | +0,020 ± 0,016 (1) |
  | **C sabit η − B+Z sabit η** (aynı η) | +0,005 ± 0,013 (1) | −0,020 ± 0,014 (3) |
  | **C η_t − B+Z sabit η** | +0,010 ± 0,012 (1) | −0,003 ± 0,012 (1) |
  | **C+çapa η_t − B+Z sabit η** | +0,011 ± 0,010 (1) | +0,013 ± 0,019 (1) |

- **Adaptasyon istatistikleri:**
  - η̄: sabit 1e-3; η_t 3,5e-3 (Indy) ve 4,0e-3 (Loco).
  - ‖ΔW‖/‖W_base‖ son değeri 0,004–0,022. ρ = 0,25 sınırına hiç dayanmadı (%0); fiilen λ_pull sınırlıyor.
  - Val τ: C'de medyan 85–90 ms, aralık 41–112. Çapa ile Indy'de aralık daralıyor (61–109).
  - K7: oran 0,009–0,020, tümü ✗ (KR-014'teki bilinen ihlal, değişmedi).
- **Okuma:**
  1. **Okuma uyumu her konfigürasyonda yardım ediyor:** 12/12 eşli karşılaştırmada pozitif. ΔW, W_base'in yalnızca %0,4–2'si kadar.
  2. **KR-030 hipotezi, "C özellikle okuma uyumuna muhtaç", desteklenmiyor:** Aynı sabit η'da uyum B+Z'ye C'den fazla yardım ediyor (Loco +0,023 vs +0,015). C sabit η, Loco'da B+Z sabit η'dan 3/3 seed kötü (−0,020).
  3. **η_t, sabit η'dan 6/6 seed iyi.** Ancak η_t'nin ortalaması sabit η'nın ~3,5–4 katı. Kazanç kapının zamanlamasından değil, yalnızca daha büyük η'dan geliyor olabilir. Eşlenmiş η kontrolü (sabit η ≈ 4e-3) yapılmadı. Bu yapılmadan "nöromodüle plastisite" etkisi iddia edilemez.
  4. **İstenen ana karşılaştırma pozitif ama adil değil:** C η_t, donuk B+Z'den Indy +0,014, Loco +0,020. Ancak uyum B+Z'ye de açıldığında fark Indy +0,010 ± 0,012, Loco −0,003 ± 0,012 oluyor. n = 3'te gürültüden ayırt edilemez.
  5. **Çapa umut verici:** Loco'da 3/3 seed iyileştirdi (+0,016), B+Z sabit η'ya karşı +0,013 ± 0,019. Tek oturum, n = 3; ön bulgu.
- **Uyarılar:**
  - Yalnızca ayar seed'leri (100–102), val ve iki kısaltılmış oturum var. Rapor seed'leri ve test bölmesi yok.
  - PHASES.md'deki asıl Faz 6 sorusu (oturumlar arası kayma, Wilcoxon) henüz koşulmadı. Bu adım 6a (oturum içi).
  - Koşu duvar saatinde ~9 saat sürdü, beklenen ~1 saatti (büyük olasılıkla uyku/pil). Sonuçlar etkilenmez.
- **Sıradaki öneri:** Eşlenmiş η kontrolü: B+Z ve C için sabit η = 4e-3. Ayrıca B+Z için de η_t'ye eşdeğer bir zaman profili denenebilir.
- **Ek — eşlenmiş η kontrolü (2026-11-03):** `runs/20261103-104631_nhp_motor_etamatch_faz6_seeds100-101-102`.
  - B+Z ve C için sabit η = 4e-3 (C η_t'nin η̄'sı: Indy 3,5e-3, Loco 4,0e-3). Eski varyantlar önceki koşuyu birebir tekrar etti.
  - R² (val), ortalama ± std:

    | | B+Z donuk | B+Z sabit 1e-3 | B+Z eşlenmiş 4e-3 | C sabit 1e-3 | C eşlenmiş 4e-3 | C η_t |
    |---|---|---|---|---|---|---|
    | Indy | 0,304 ± 0,018 | 0,308 ± 0,015 | 0,313 ± 0,013 | 0,313 ± 0,026 | 0,320 ± 0,023 | 0,317 ± 0,026 |
    | Loco | 0,169 ± 0,021 | 0,192 ± 0,014 | 0,216 ± 0,008 | 0,172 ± 0,010 | 0,190 ± 0,013 | 0,189 ± 0,013 |
  - Eşli farklar:
    - C η_t − C eşlenmiş: Indy −0,002, Loco −0,001. **η_t'nin kazancı tamamen daha büyük ortalama η'dan geliyor; τ kapısının zamanlaması bir şey katmıyor.** (Loco'da η̄ birebir eşleşik; Indy'de eşlenmiş η biraz yüksek.)
    - C eşlenmiş − B+Z eşlenmiş: Indy +0,006 ± 0,012 (1/3 kötü), Loco −0,025 ± 0,013 (3/3 kötü). Aynı η'da C, B+Z'yi geçmiyor; Loco'da tutarlı biçimde geride.
    - B+Z eşlenmiş − B+Z donuk: Indy +0,010 (3/3), Loco +0,046 (3/3). Şimdiye kadarki en iyi Loco sonucu (C+çapa η_t 0,206'yı da geçiyor).
  - Ek gözlemler:
    - ‖ΔW‖/‖W_base‖ ≤ 0,022, sınıra hiç dayanmıyor; η ayarlanmamış ve daha büyük η hâlâ yardım ediyor olabilir (yalnızca ayar seed'leri/val'de denenecek).
    - Loco'da R² val'in ilk yarısından ikinci yarısına düşüyor (B+Z donuk 0,207 → 0,136); Indy'de artıyor (0,220 → 0,372). Loco val'inde gerçek bir oturum içi kayma var; uyumun Loco'da daha çok işe yaraması bununla tutarlı.
  - **Sonuç:** Faz 6a'da "nöromodüle plastisite" (τ kapılı η_t) ve regülatör (C) hipotezleri desteklenmedi. Yardım eden şey sınırlı çevrimiçi okuma uyumunun kendisi; B+Z ile en iyi çalışıyor.
  - Çapa varyantı bu kontrolde koşulmadı; çapanın Loco kazancının da η'dan bağımsız olup olmadığı açık.


### KR-032 — Sürpriz / duygu kapılı η (rezervuar sabit τ): eşlenmiş η'ya göre katkı yok; asıl kaldıraç η'nın kendisi
- **Bağlam:** KR-031 ekinden sonra kullanıcının önerisi: rezervuar B+Z'deki gibi sabit kalsın (τ = 100), z yalnızca okumanın öğrenme hızına bağlansın; hata beklenmedik biçimde artınca hızlı, her şey yolundaysa yavaş öğren.
- **Uygulama (deneysel, ARCHITECTURE.md değişmedi — onay bekliyor):** `decision.fast.eta_mode` iki yeni değer aldı, varsayılan kapalı:
  - `surprise`: ‖δ‖²'nin kısa (0,1) / uzun (0,002) EMA oranı ρ_t; kapı = clip((ρ_t − 1)/(3 − 1), 0, 1).
  - `affect`: duygu uzayının standart ölçekli `s` başı (kötüleşme eğilimi); kapı = clip(ŝ/2, 0, 1).
  - η_t = η_base + η_mod·kapı = 1e-3 + 1,5e-2·kapı (K4 sınırında). Kapı ŷ_t skorlandıktan sonra hesaplanır.
  - Testler: sıfır sızıntı, kapı [0, 1], hata adımında açılıp kapanma, η = 0 → Faz 5 bit eşitliği.
- **Koşu:** `runs/20261103-121154_nhp_motor_surprise_faz6_seeds100-101-102` (B+Z eğitimi, Indy + Loco, seed 100–102, val). Sabit η eğrisi 2e-3 … 1,6e-2.
  - R² (val), ortalama ± std:

    | | donuk | η 2e-3 | η 4e-3 | η 8e-3 | η 1,6e-2 | sürpriz η | affect η |
    |---|---|---|---|---|---|---|---|
    | Indy | 0,304 ± 0,018 | 0,310 ± 0,014 | 0,313 ± 0,013 | 0,320 ± 0,012 | 0,335 ± 0,011 | 0,309 ± 0,011 | 0,312 ± 0,015 |
    | Loco | 0,169 ± 0,021 | 0,203 ± 0,011 | 0,216 ± 0,008 | 0,232 ± 0,007 | 0,260 ± 0,006 | 0,209 ± 0,008 | 0,209 ± 0,009 |
  - Eşlenmiş η karşılaştırması (seed başına, sabit η eğrisinin log η'da kapının η̄'sına enterpolasyonu; η̄ ≈ 2,8–4,1e-3):
    - sürpriz η − eşlenmiş: Indy −0,005 ± 0,002 (3/3 kötü), Loco −0,002 ± 0,002 (2/3 kötü).
    - affect η − eşlenmiş: Indy −0,001 ± 0,003 (1/3 kötü), Loco −0,002 ± 0,003 (2/3 kötü).
  - Kapı istatistikleri: sürpriz kapısı adımların ~%25–30'unda açık, ~%10'unda tam açık; affect kapısı %40–80 açık, hiç tam açılmıyor.
- **Sonuç:**
  - Hipotez desteklenmedi: η'yı zamana göre dağıtmak (sürpriz ya da z ile) aynı ortalama η'lı sabit kuraldan iyi değil; sürpriz kapısı Indy'de 3/3 hafif kötü.
  - Asıl bulgu: R², sabit η ile tekdüze artıyor ve 1,6e-2'de hâlâ tepe yapmadı (Loco donuğa göre +0,091, Indy +0,031; 3/3). ‖ΔW‖/‖W_base‖ ≤ 0,06, ρ = 0,25 sınırından uzak. Okuma uyumu ciddi biçimde az ayarlı; önce η (ve λ_pull) taranmalı, kapılar bundan sonra yeniden değerlendirilmeli (daha yüksek η'da "yavaş öğren, eskiyi koru" tarafının değeri artabilir).
  - Loco'da η 1,6e-2 ile bile R² val'in ilk yarısından ikinciye düşüyor (0,325 → 0,208); kayma tam izlenemiyor.
- **Uyarılar:** 3 ayar seed'i, yalnızca val, oturum içi. η taraması da val'de bir ayar; sonuçlar rapor seed'leri ve test bölmesinde bir kez doğrulanmalı.

### KR-033 — K2 kaldırıldı; tavan ve kayıp ölçümü: kayıp okumada değil, doğrusal temsilde
- **Bağlam:** Kullanıcı K2'yi (global geri yayılım yok) CLAUDE.md'den çıkardı; ARCHITECTURE.md'de hâlâ yazılıydı. Ayrıca "iyileştirmeler körlemesine olmasın" diye önce kaybın yerinin ölçülmesini istedi.
- **Karar (K2):** ARCHITECTURE.md'de K2 "kaldırıldı" olarak işaretlendi, numara boş bırakıldı (K3–K8 adları değişmez). Okuma blokları yerel kurallarla öğrenmeye devam eder (tasarım seçimi, değişmez değil). Tavan modelleri otomatik türev kullanabilir. `test_k2_no_autograd_in_src` kaldırıldı; §3.3, §10, §11, Explain.md ve yorumlardaki K2 atıfları temizlendi.
- **Ölçüm:** `experiments/ceiling.py`, `configs/ceiling.yaml`. Koşu `runs/20261103-151255_ceiling_seeds0-1-2-3-4`. Aynı 20 ms ızgara, 80/10/10 bölme, test R²; tüm seçimler val'de.
  - (a) Wiener: son H kutunun ham sayımları üzerinde ridge, H ∈ {1, 5, 10, 20}.
  - (b) GRU: 1 katman, gizli 128/256 (val'de), TBPTT (16 akış × 50 adım), AdamW, val'de erken durdurma; değerlendirmede tüm dosya tek nedensel akış. 5 seed.
  - (c) ridge-r ve NLMS (A, 12 tur): Faz 3 tam koşusundan (`runs/20261030-101907_nhp_motor_full_seeds0-1-2-3-4`).

  | Oturum | (a) Wiener H=10 | Wiener H=20 | (b) GRU | (c) ridge-r | NLMS |
  |---|---|---|---|---|---|
  | indy_20170131_02 | 0,508 | 0,585 | 0,779 ± 0,014 | 0,500 ± 0,014 | 0,446 ± 0,018 |
  | indy_20160630_01 | 0,351 | 0,352 | 0,623 ± 0,007 | 0,368 ± 0,007 | 0,352 ± 0,009 |
  | indy_20160622_01 | 0,510 | 0,601 | 0,806 ± 0,008 | 0,553 ± 0,013 | 0,529 ± 0,010 |
  | loco_20170301_05 | 0,393 | 0,457 | 0,689 ± 0,021 | 0,355 ± 0,011 | 0,399 ± 0,014 |
  | loco_20170215_02 | 0,391 | 0,403 | 0,662 ± 0,020 | 0,378 ± 0,023 | 0,344 ± 0,009 |
  | loco_20170210_03 | 0,336 | 0,386 | 0,705 ± 0,021 | 0,319 ± 0,007 | 0,352 ± 0,007 |
  | **Indy ort.** | 0,457 | 0,513 | 0,736 | 0,474 | 0,442 |
  | **Loco ort.** | 0,373 | 0,415 | 0,685 | 0,351 | 0,365 |
- **Okuma:**
  1. **(c) − NLMS küçük:** Indy +0,032, Loco −0,014 (iki Loco oturumunda NLMS ridge'den iyi). Okuma kuralı darboğaz değil.
  2. **(a) ≈ (c), ama (a) H = 20 > (c):** Rezervuar ~200 ms doğrusal geçmiş kadar bilgi taşıyor; 400 ms geçmişin doğrusal okumasına göre Indy −0,04, Loco −0,06 kaybediyor. Rezervuar bellek ufku kısa ya da bilgiyi doğrusal okunur biçimde tutmuyor.
  3. **(b) ≫ hepsi:** GRU her oturumda +0,25–0,35 önde. GRU NeuroBench taban çizgilerini de geçiyor (0,736 / 0,685 vs SNN 0,593 / 0,568) ve AEGRU makalesindeki vanilla GRU ile tutarlı (6 oturum ort. 0,67, 4 ms adım, 50/25/25). Asıl kayıp doğrusal okunan temsilin kendisinde: hiçbir doğrusal okuma (ham geçmiş dahil) ~0,5'i geçmiyor.
- **Sonuç:** Okuma kuralını (η, kapılar) iyileştirmek en fazla ~0,03 kazandırabilir. Büyük fark (~0,3) doğrusal olmayan zamansal bütünleştirmede; mimari kararı gerekiyor (ör. rezervuarın bellek ufku/doğrusal olmayanlığı, okumada doğrusal olmayan katman ya da eğitilebilir rekürrent blok).
- **Uyarılar:** Wiener'de Loco λ ızgara kenarında (1e5) ve H = 20 en uzun denenen; daha uzun H / büyük λ Wiener'i biraz daha yükseltebilir. GRU gizli boyutu da çoğunlukla ızgaranın üst ucunda (256): GRU sayısı tavanın alt sınırı. Protokol NeuroBench ile birebir değil (20 ms adım, 80/10/10, KR-018).

---

## Spike v2 (2026-11-03)

### KR-034 — Spike v2 hattına geçiş: eğitilebilir omurga + nöromodüle çevrimiçi yerel uyum
- **Bağlam:** KR-033: kayıp okuma kuralında değil, doğrusal okunan temsilde (GRU tavanı v1'den +0,25–0,35 R²). Kullanıcı `docs/spike_v2_mimari_spesifikasyon.md`'yi yükledi ve bu doğrultuda yeni özellik eklemek istedi.
- **Karar:**
  - ARCHITECTURE.md'ye §12 (Spike v2) ve v2 kuralları **KV1–KV8** eklendi. v1 kuralları (§2–§10) v1 koduna uygulanmaya devam ediyor; v2 kodu yalnızca §12'ye bağlı.
  - Temel değişiklikler (v1 → v2):
    - Omurga çevrimdışı eğitilir, çevrimiçi donuktur (K1 → KV1).
    - τ çevrimiçi modüle edilmez; regülatörler yalnızca g_fast/g_slow ve η üretir (K4 → KV4).
    - k-WTA yerine homeostatik kayıp + eşik adaptasyonu (K7 → KV7).
    - Kodlayıcı Anscombe + kanal z-skoru + yumuşak kırpma.
    - Δt = 4 ms, iç adım yok.
  - Fazlar: `docs/PhasesV2.md` (V0–V6). v1 Faz 6 kayma deneyi askıya alındı.
  - Değerlendirme koşulları `v2-A` … `v2-E` (v1 A/B/C ile karışmasın diye).
- **Alternatifler:**
  - v1'i yerinde değiştirmek: v1 sonuçlarının tekrarlanabilirliği ve 139 test bozulurdu. Seçilmedi; v2 ayrı modüllerde (öneri `src/v2/`, V0'da onaylanacak).
  - v1 rezervuarını büyütmek / doğrusal olmayan okuma eklemek: KR-033'te seçenek olarak duruyordu; kullanıcı spesifikasyonu eğitilebilir omurgayı seçti.
- **Spesifikasyon notları:**
  - Spesifikasyonun v1 teşhisinin bir kısmı depodaki v1 ile uyuşmuyor: kodlayıcı Min-Max + Poisson değil, delta + termometre; τ tekil değil, τ·m_i yayılımlı; efference copy ve elastik çapalı NLMS v1'de zaten var. Ateşleme oranı teşhisi (%1–2) doğru. Bunlar v2 kararlarını değiştirmez.
  - Açık noktalar ARCHITECTURE.md §12.10'da: bölme protokolü, kod yerleşimi, plif sıfırlama gecikmesi, beyincik girdisi döngüsü, bias güncellemesi, hesap bütçesi, primat başına ön-eğitim.
- **Risk:** v1'de sürpriz/duygu kapılı η eşlenmiş sabit η'yı geçmedi (KR-031, KR-032). v2-D > v2-B iddiası bu yüzden önceden kanıtlı sayılmaz; v2-B (eşlenmiş η̄) ve v2-C (karıştırılmış η) kontrolleri zorunlu.
- **Durum:** Yalnızca dokümanlar güncellendi; kod yok. Sıradaki: Faz V0.

### KR-035 — v2 bölme protokolü, kod yerleşimi ve günler arası protokol (Faz V0)
- **Bağlam:** ARCHITECTURE.md §12.10 açık noktaları 1, 2 ve 7. Bölme seçimi tüm v2 fazlarında sabit kalmalı; sonradan değişirse tavan ve sonuçlar karşılaştırılamaz, test seti kayabilir.
- **Karar:**
  - **Bölme:** NeuroBench segment bölmesi, v1 ile aynı: hedef segmentlerinin ilk %80'i train, kalan yarı yarıya val / test (80/10/10). `data.train_ratio: 0.8`. Tüm v2 fazlarında sabit.
  - **Kod yerleşimi:** v2 modülleri `src/v2/`, testleri `tests/v2/`, config'leri `configs/v2_*.yaml`, betikleri `experiments/v2_*.py`. v1 dosyaları değişmez; v2 yalnızca altyapıyı (`src/data/primate.py`, `src/utils/`, `experiments/common.py`, `experiments/ceiling.py`) import eder.
  - **Δt ızgarası:** `time.dt_ms = data.bin_width = data.stride` (4 ms); uyuşmazlık hata verir (`check_time`).
  - **Günler arası protokol:** primat başına oturumlar tarih sırasıyla, ilki Day-0 (Indy: 20160622 → 20160630 → 20170131; Loco: 20170210 → 20170215 → 20170301). Kanal sayıları farklı olduğundan (Indy 96, Loco 192) ön-eğitim primat başınadır; kanal eşleme gerekmez.
  - **Pencereler:** 2000 ms uzunluk, 1000 ms kayma, yalnızca train diliminden.
- **Alternatifler:**
  - Spesifikasyonun 65/10/25'i: test daha büyük, ama v1 sonuçları ve KR-033 tavanı ile doğrudan karşılaştırma kaybolur. Seçilmedi.
- **Not:** Bölme v1 ile aynı olduğu için v1 sonuçları (KR-033, Faz 6a) v2 ile aynı val/test diliminde karşılaştırılabilir. 80/10/10, NeuroBench resmî yükleyicisinin varsayılanıdır (`train_ratio=0.8`, segment sırasıyla).
- **Literatür karşılaştırması (2026-11-04 taraması; kullanıcı 80/10/10'da kalmayı seçti):**
  - v2 spesifikasyonu Zenke Lab RSNN protokolüyle (arXiv 2409.01762) neredeyse aynıdır: 4 ms, 2 s pencere, maymun başına ön-eğitim + oturum başına ince ayar, RMSE, 5 seed. Fark: onlar 65/10/25 kullanır. "R² ≥ 0,65–0,70" hedefi oradaki tinyRSNN (0,660) / bigRSNN (0,698) ortalamalarıdır.
  - bigRSNN oturum başına: I1 0,770, I2 0,585, I3 0,772, L1 0,698, L2 0,629, L3 0,734. NeuroBench SNN tabanı ort. 0,581, SNN_Flat 0,633.
  - Bölmeler farklı olduğundan bu sayılarla kıyas **yaklaşıktır** (dış referans). Ana referans 4 ms iç tavanıdır. Birebir kıyas gerekirse sonuç fazında 65/10/25 ile ayrı bir değerlendirme koşusu yapılır; ana bölme değişmez.
  - Günler arası etiketsiz SNN uyumu için en yakın iş: Membrane Potential Alignment (arXiv 2606.14866; Indy, 50/25/25, 10 hedef gün, uyumsuz 0,09 → uyumlu 0,33; LoRA + KL, gradyanlı). V6'da bağlam olarak raporlanır.
- **Durum:** Kodda uygulandı (`src/v2/data.py`, `configs/v2_base.yaml`), testler `tests/v2/test_v0_data.py`.

### KR-036 — Hesap bütçesi ölçümü Faz V2'ye ertelendi
- **Bağlam:** V0 çıkış kriterlerinden biri epoch süresi ölçümüydü (`experiments/v2_bench.py`, §12.10 nokta 6). Omurga henüz yazılmadığından betik gerçek `TemporalBackbone`'u değil, prototip `plif` / `gru_loop` adımlarını ölçüyor.
- **Karar (kullanıcı, 2026-11-04):** Bütçe ölçümü Faz V2'ye taşındı; gerçek omurga yazıldığında aynı betikle (prototip yerine `src/v2` omurgası) ölçülür ve epoch sayıları gerekiyorsa o zaman KR ile güncellenir. V0 çıkış kriterinden çıkarıldı.
- **Etkisi:** V3 ön-eğitimi başlamadan önce ölçüm yine yapılmış olur; prototip sayısına dayanarak erken karar verilmez.
- **Durum:** `experiments/v2_bench.py` ve `bench` config bloğu depoda kalır; V2'de güncellenecek.

### KR-037 — 4 ms tavanı: GRU Indy 0,708 / Loco 0,616; 20 ms tavanının altında (Faz V0 kapanışı)
- **Koşu:** `experiments/v2_ceiling.py`, `configs/v2_ceiling.yaml`, `runs/20261103-214626_v2_ceiling_seeds0-1-2-3-4`.
  - Δt = 4 ms, 80/10/10 (KR-035), tahmin ve skor her 4 ms adımında, test R².
  - Wiener: 20 ms'lik nedensel kutu toplamları, 100/200/400 ms geçmiş; λ ve geçmiş val'de.
  - GRU: KR-033 ile aynı model ve eğitim (gizli 128/256 val'de, TBPTT 16 akış × 1 s, erken durdurma). 5 seed.
  - Koşu v2 dosyaları commit edilmeden başladı (`meta.json`: 33d5036, dirty); kod sonra değişmeden `a0ad343`'te commit edildi.

  | Oturum | Wiener 200 ms | Wiener 400 ms (val'de seçilen) | GRU 4 ms | GRU 20 ms (KR-033) | Fark |
  |---|---|---|---|---|---|
  | indy_20170131_02 | 0,513 | 0,580 | 0,752 ± 0,019 | 0,779 | −0,027 |
  | indy_20160630_01 | 0,356 | 0,359 | 0,597 ± 0,014 | 0,623 | −0,026 |
  | indy_20160622_01 | 0,508 | 0,599 | 0,775 ± 0,013 | 0,806 | −0,031 |
  | loco_20170301_05 | 0,472 | 0,518 | 0,632 ± 0,008 | 0,689 | −0,057 |
  | loco_20170215_02 | 0,396 | 0,430 | 0,616 ± 0,020 | 0,662 | −0,046 |
  | loco_20170210_03 | 0,294 | 0,348 | 0,600 ± 0,029 | 0,705 | −0,105 |
  | **Indy ort.** | 0,459 | 0,513 | **0,708** | 0,736 | −0,028 |
  | **Loco ort.** | 0,387 | 0,432 | **0,616** | 0,685 | −0,069 |
- **Okuma:**
  1. Wiener 4 ms'de 20 ms ile aynı düzeyde (KR-033 H = 20: Indy 0,513, Loco 0,415). Ön işleme doğru: 4 ms ızgara bilgi kaybettirmiyor.
  2. GRU 4 ms'de her oturumda 20 ms'den düşük; Loco'da belirgin (−0,07). Sebep büyük olasılıkla eğitim: hiperparametreler 20 ms için seçildi. Loco'da en iyi epoch çoğunlukla 3–6 (erken tepe, sonra val düşüyor). Gizli boyut çoğunlukla ızgaranın üst ucunda (256).
  3. Bu yüzden 4 ms GRU sayısı gerçek tavan değil, **alt sınırdır**. Dış referans bigRSNN (KR-035, farklı bölme) Loco'da 0,629–0,734; bizim 4 ms GRU'muz bunun altında.
  4. `indy_20160630_01`'de 4 ms GRU 0,597; v2'nin ≥ 0,65 hedefi bu oturumda tavana göre değerlendirilmeli (bigRSNN de 0,585).
  5. `loco_20170210_03`'te Wiener λ ızgaranın üst kenarında (1e6); Wiener değeri hafif eksik olabilir.
- **Sonuç:** V0 çıkış kriterleri sağlandı (KR-036 ile bütçe V2'ye taşındı). v2 için iç referans: 4 ms GRU (Indy 0,708 / Loco 0,616), 20 ms GRU (0,736 / 0,685) ve bigRSNN birlikte raporlanır.
- **Açık:** Daha sıkı bir 4 ms tavanı istenirse GRU yalnızca val'de yeniden ayarlanabilir (lr, gizli 512, BPTT uzunluğu). V2'deki `tau_gru` karşılaştırması bu sayıya dayanacağı için karar V2 öncesinde verilmeli.

---

## Altyapı

### KR-022 — Batch'li okuma blokları; tam koşu iki aşamalı, oturum başına CPU süreci
- **Bağlam:** Tam koşu (6 oturum × 5 seed × {A, B, B+Z}, 12 tur) tekil ajanlarla CPU'da ~6–9 saat sürecekti. Asıl yük rezervuar değil, okuma bloklarıydı: tekil B+Z ajanında adım başına ~4,6 ms, çoğu Python/çekirdek çağrısı maliyeti. Kullanıcı "önce GPU batch" seçeneğini seçti.
- **Karar:**
  - `src/batched.py` (`BatchedAgent`): M bağımsız ajanın (5 seed) okuma blokları tek tensörde `[M, …]` yürütülür. Matematik tekil `Agent` ile aynıdır; tekil ajanlardan kurulur, böylece başlangıç ağırlıkları birebir kopyalanır. `tests/test_batched.py` A, B, B+Z için eşdeğerliği doğrular.
  - `experiments/nhp_full.py` iki aşamalıdır:
    - Aşama 1 (CPU, paralel): r üretilir ve diske yazılır (float32 `.npy`, ~17 GB); oran ve ridge hesaplanır.
    - Aşama 2: oturum başına bir süreç, r bellek eşlemesiyle parça parça okunur.
- **Ölçüm** (5 ajan birlikte, adım başına):

  | | CPU 1 iş parçacığı | GPU (RTX 4060 Laptop) |
  |---|---|---|
  | A | 0,21 ms | 0,60 ms |
  | B+Z | 3,71 ms | 2,91 ms |
  | B+Z, 15 ajan | 11,6 ms | 3,05 ms |
- **Sonuç:** Batch'leme ajan başına maliyeti ~6× düşürdü (4,6 → 0,74 ms). GPU, Windows'ta çekirdek başlatma yükü yüzünden adım başına ~3 ms'nin altına inmiyor; tüm işi tek GPU sürecine yığmak ~6–7 saat sürerdi. Oturum başına paralel CPU süreçleri en hızlısı (`nhp.stage2_device: cpu`, değiştirilebilir). GPU'nun kazanması için M'nin çok büyük olması gerekir (ör. Faz 6'da çok sayıda ablation/seed tek batch'te) ya da CUDA Graphs.
- **Doğrulama:** Duman koşusu (`indy_20170131_02`, 2 seed, 1 tur): `leak_free = True`, `K3 (A = B) = True`; aşama 2 1,3 dakika.
- **Gerçekleşen (2026-10-31):** Tam koşu sorunsuz bitti (30/30). Ama duvar saati beklenenden uzun sürdü: aşama 2 ~19–20 saat. Sebepler: ~7,5 saat gece uykusu, pilde çalışırken yavaşlama, 6 süreç × 2 iş parçacığının çekirdekleri paylaşması. Ders: uzun koşularda fişe takılı + uyku kapalı. Faz 5–6 öncesi 30 ajanı tek GPU batch'inde (veya Linux/Colab'da) koşmak ölçülmeli.
