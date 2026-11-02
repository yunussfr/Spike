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
