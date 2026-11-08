# Mimari Kuralları: Projeksiyon–Beyincik–Duygu SNN

Bu dosya kod tabanının **değişmez kurallarını** tanımlar. Her yeni kod, her refaktör ve her deney bu kurallara uymak zorundadır. Bir kural değişecekse önce bu dosya güncellenir, sonra kod.

> **Sürümler (2026-11-03, KR-034).** Depoda iki mimari hattı vardır:
> - **Spike v1** (§1–§10): sabit LIF rezervuarı + yerel okumalar. Kuralları K1–K8. v1 kodu (`LIFReservoir`, A/B/C, Faz 0–6) bu kurallara uymaya devam eder.
> - **Spike v2** (§12): çevrimdışı eğitilen zamansal omurga + çevrimiçi nöromodüle yerel uyum. Kuralları **KV1–KV8**. Kaynak: `docs/spike_v2_mimari_spesifikasyon.md`, fazlar: `docs/PhasesV2.md`.
> - v2 kodu yalnızca §12'ye bağlıdır; v1 kuralları (ör. K1 sabit rezervuar, K7 k-WTA) v2 modüllerini bağlamaz. Aktif geliştirme hattı v2'dir.

---

## 1. Amaç

Sabit (eğitilmeyen) bir spiking projeksiyon bloğu üzerine, yalnızca yerel kurallarla öğrenen üç okuma bloğu kurmak:

- **Beyincik bloğu:** Bir sonraki girdiyi tahmin eder, tahmin hatasıyla öğrenir.
- **Duygu uzayı:** Sistemin o anki durumunu düşük boyutlu bir vektöre (z) sıkıştırır ve bu vektörden dört regülatör üretir.
- **Karar bloğu:** Görev çıktısını üretir, sonuç hatasıyla öğrenir.

**Araştırma sorusu:** Öğrenilen duygu uzayı ve regülatörler, veri zamanla değiştiğinde (kayma) sistemin uyum sağlamasına yardım ediyor mu?

---

## 2. Değişmez kurallar

### K1 — Projeksiyon bloğu asla eğitilmez
- Projeksiyon matrisi `W_in` ve rekürrent matris `W_rec` başlangıçta **sabit bir seed** ile rastgele üretilir.
- Bu tensörler `requires_grad=False` olarak tutulur ve hiçbir güncelleme fonksiyonu onlara dokunmaz.
- Nöron başına zaman sabiti çarpanları `m_i` (Bölüm 3.2) da aynı seed ile bir kez üretilir ve aynı şekilde sabittir.
- Matrisler checkpoint'e kaydedilir. Aynı seed ve aynı config her zaman aynı matrisleri üretmelidir.
- Bu blokta değişebilen tek şey, regülatörlerden gelen **g** (girdi kazancı) ve **τ** (sızıntı zaman sabiti) değerleridir.

### K2 — (kaldırıldı, 2026-11-03)
- Eski kural ("global geri yayılım yok, tüm güncellemeler yerel") kullanıcı kararıyla değişmez kurallardan çıkarıldı (KR-033). Numaralar korunur; K3–K8'in adları değişmez.
- Mevcut okuma blokları (§3.3, §3.4, §3.6) açık formüllü yerel kurallarla öğrenmeye devam eder; bu artık bir değişmez değil, bir tasarım seçimidir.
- Tavan/referans modelleri (ör. BPTT ile eğitilen GRU, §8) otomatik türev kullanabilir. Bunlar ajanın parçası değildir.

### K3 — İki öğrenme sinyali birbirine karışmaz
| Sinyal | Tanım | Kimi eğitir | Kimi eğitmez |
|---|---|---|---|
| **δ** (sonuç hatası) | hedef − karar çıktısı | Yalnızca karar bloğu | Beyincik, projeksiyon |
| **e** (tahmin hatası) | x(t+1) − x̂(t+1) | Yalnızca beyincik bloğu | Karar, projeksiyon |

- Duygu uzayı δ ve e'yi **okur** (kendi hedeflerini bunlardan hesaplar) ama bu sinyalleri **değiştirmez**.

### K4 — Regülatörler yalnızca dört parametreyi değiştirir
| Regülatör | Etkilediği yer | Alt sınır | Üst sınır | Varsayılan |
|---|---|---|---|---|
| **g** (kazanç) | Projeksiyona giren akım | 0.5 | 2.0 | 1.0 |
| **τ** (sızıntı, ms) | Rezervuar zar sızıntısı | 20 | 2000 | 100 |
| **η** (öğrenme hızı) | Karar ve beyincik güncellemeleri | 1e-4 | 1e-1 | 1e-2 |
| **β** (karar sıcaklığı) | Karar bloğunun seçim keskinliği (yalnızca ayrık görevlerde) | 0.1 | 10 | 1.0 |

- Regülatör çıktıları sigmoid ile [0, 1] aralığına alınır, sonra bu sınırlara doğrusal olarak ölçeklenir.
- τ log-ölçekte ölçeklenir (20 ile 2000 arası çok geniş bir aralık olduğu için).
- τ eşlemesi asimetrik olabilir (`regulators.tau_ceiling`, §3.5): aşağı yönde alt sınıra (20) kadar iner, yukarı yönde `tau_ceiling · τ_varsayılan` tavanında yumuşakça doyar. Tavan K4 aralığının içindedir; K4 ihlal edilmez (KR-030).
- Regülatörün ürettiği τ, rezervuar nöronlarının zaman sabitlerinin **geometrik ortalamasıdır**. Her nöronun kendi zaman sabiti `τ_i = τ · m_i`'dir (Bölüm 3.2); regülatör yalnızca ortak τ'yu değiştirir, `m_i`'lere dokunamaz.
- τ aralığı zaman ölçeğine göre seçilmiştir: bir ortam adımı `T · dt = 20 ms` sürer. τ = 20 ms ile bellek yaklaşık 1 ortam adımı, τ = 2000 ms ile yaklaşık 100 ortam adımı mertebesindedir.
- Regülatörler başka hiçbir hiperparametreye, ağırlığa ya da modül yapısına dokunamaz.

### K5 — Regülatörsüz mod her zaman çalışmalı
- Config'te `regulators.enabled: false` olduğunda dört regülatör de varsayılan değerlerinde sabit kalır.
- Bu mod, karşılaştırma (ablation) için zorunludur ve her sürümde test edilmelidir.

### K6 — Zaman ve bellek kuralları
- Ortamın (ya da veri setinin) bir adımı = SNN'nin **T** iç adımı. Varsayılan `dt = 1 ms`, `T = 20`.
- Rezervuar durumu (zar potansiyelleri) bir dizi boyunca **sıfırlanmaz**. Sıfırlama yalnızca yeni bir dizi ya da bölüm (episode) başladığında yapılır.
- **Gelecekten bilgi sızması yasaktır:** Beyincik x̂(t+1)'i üretirken x(t+1)'i göremez. Hata e, x(t+1) geldikten sonra hesaplanır.

### K7 — Seyreklik
- k-WTA, her iç adımda **eşiği geçen** adaylar (`V > θ`) arasından en güçlü **en fazla** `k = ⌈ρ · N⌉` nöronun ateşlenmesine izin verir. Varsayılan `ρ = 0.05`.
- Ateşleyen nöron sayısı hiçbir iç adımda `k`'yı aşmaz. Girdi yeterince güçlüyse `k`'ya eşittir, girdi yoksa sıfıra düşebilir (aktivite sönebilir). "Her adımda tam k nöron ateşler" diye bir kural **yoktur**.
- Ortalama ateşleme oranı her deneyde ve **izin verilen her τ değeri için** loglanır. `ρ/2` ile `2ρ` aralığının dışına çıkarsa bu bir hata sayılır. Yani τ yalnızca belleği değiştirmeli, rezervuarı sessizleştirmemeli ya da aşırı aktifleştirmemelidir.

### K8 — Tekrarlanabilirlik
- Her deney tek bir YAML config dosyasından çalışır. Kod içinde sabit (hard-coded) hiperparametre olmaz.
- Her deney en az **5 farklı seed** ile koşulur. Sonuçlar ortalama ± standart sapma olarak raporlanır.
- Config, seed, git commit hash'i ve metrikler her koşuda birlikte kaydedilir.

---

## 3. Denklemler

### 3.1 Spike kodlayıcı (`SpikeEncoder`)
Kodlayıcı iki kanal grubunu yan yana üretir: **delta** (ne değişti) ve **seviye** (şu anki değer ne).

**Delta kanalları:** girdi bir eşikten fazla değiştiğinde spike üretilir.
- Pozitif kanal: `x_t − x_{t−1} > θ_enc` ise spike.
- Negatif kanal: `x_{t−1} − x_t > θ_enc` ise spike.
- Boyut: `2 · D_in`. Dizinin ilk adımında önceki değer olmadığı için spike üretilmez.

**Seviye kanalları (termometre kodu):** her girdi boyutu için `L` kanal, eşikleri `[x_alt, x_üst]` aralığına eşit aralıklı yerleştirilmiş:
```
b_j      = x_alt + j · (x_üst − x_alt) / (L + 1),   j = 1..L
seviye_j = 1  eğer x_t > b_j,  yoksa 0
```
- Boyut: `L · D_in`. Deterministiktir (gürültü yok); x arttıkça soldan sağa daha çok kanal yanar.
- `[x_alt, x_üst]` config'ten gelir. Gerçek veri setlerinde bu aralık **yalnızca eğitim verisinden** belirlenir; test verisine bakılmaz.
- **Kanal başına ölçek** (`range: train_per_channel`): girdi boyutları çok farklı ölçekteyse (ör. sinir kanallarının ateşleme oranları), her boyut eğitim verisindeki değerlerinin yüksek bir yüzdeliğine (config `data.scale_percentile`) bölünür ve aralık `[0, 1]` alınır. Ölçek yine yalnızca eğitim verisinden hesaplanır.

**Toplam çıktı:** `u = [delta, seviye]`, şekil `[B, T, C]`, `C = (2 + L) · D_in`. Her iki grup da ortam adımının T iç adımı boyunca sabit tutulur. `encoder.level.enabled: false` iken yalnızca delta kanalları üretilir (ablation için).

**Sebep:** Delta kodlama x'in yalnızca değişim yönünü iletir, seviyesini iletmez. Faz 1 teşhisinde, delta spike'larının geçmişini *mükemmel* hatırlayan bir hafıza bile son 10 adımın ortalamasını geri çıkaramadı (R² ≈ 0). Aynı doğrusal filtre bankası ham x ile 100 adımlık ortalamayı R² ≈ 0.94 ile çıkardı. Beyincik (Bölüm 3.3) x(t+1)'in değerini tahmin edeceği için seviye bilgisi zorunludur.

### 3.2 Rezervuar (LIF)
Her iç adımda (i: nöron indisi):
```
τ_i      = τ · m_i                              # τ regülatörden gelir; m_i sabit nöron çarpanı (aşağıya bak)
α_i      = exp(−dt / τ_i)                       # nöron başına sızıntı
ᾱ        = exp(−dt / τ)                         # ortak (ortalama) sızıntı
c_i      = (1 − α_i) / (1 − ᾱ)                  # göreli denge katsayısı, m_i = 1 ise c_i = 1
I(t)     = g · W_in · u(t) + γ_rec · W_rec · s(t−1)   # g regülatörden gelir, γ_rec sabit rekürrent kazanç
V_i(t)   = α_i · V_i(t−1) + c_i · I_i(t)        # ortak (1−ᾱ) çarpanı YOK: τ girdi gücünü küçültmez, yalnızca birikimin ne kadar süre kalacağını belirler
aday     = V(t) − θ(τ)                          # θ τ'ya bağlı sabit bir tablodur, aşağıya bak
s(t)     = kWTA(aday, k)                        # yalnızca aday > 0 olanlar arasından en büyük en fazla k tanesi → 1, diğerleri 0
V(t)     = V(t) − θ(τ) · s(t)                   # yumuşak sıfırlama: ateşleyen nöronun V'sinden θ çıkarılır, sıfırlanmaz
```

**Nöron başına zaman sabitleri (`m_i`).**
- `m_i` log-düzgün dağılımdan çekilir: `log m_i ~ U(−log S, log S)`, sonra geometrik ortalaması tam 1 olacak şekilde normalize edilir. Varsayılan `S = 10`: τ = 100 ms iken nöronların zaman sabitleri yaklaşık 10–1000 ms arasına yayılır.
- Sebep: Tüm nöronlar aynı τ'yu kullanırsa her nöron geçmişi aynı üstel filtreyle karıştırır. Bu durumda doğrusal bir okuma farklı gecikmeleri (ör. 5 ve 6 adım önce) birbirinden ayıramaz. Farklı zaman sabitleri farklı filtreler, yani ayrıştırılabilir bir geçmiş temsili üretir.
- `c_i` göreli denge katsayısıdır: sabit bir girdiye her nöron aynı kararlı seviyeye (`I / (1 − ᾱ)`) ulaşır, yalnızca hızları farklıdır. Bu olmadan uzun `τ_i`'li nöronların V'si çok büyür ve k-WTA hep aynı nöronları seçer. `c_i` ortak bir `(1−α)` çarpanı değildir: ortalama nöronda 1'e eşittir, girdi gücünü τ ile küçültmez.

**Yumuşak sıfırlama.**
- Ateşleyen nöronun V'sinden θ çıkarılır. Eşiği aşan fazlalık ve nöronun geçmiş birikimi korunur.
- Sebep: Sıfıra çekme, aktif nöronların belleğini her ateşlemede siler. Ateşleme oranı ~%4 iken bu, τ'dan bağımsız olarak ~1–2 ortam adımlık bir bellek sınırı yaratıyordu.

**Rekürrent kazanç (`γ_rec`).**
- `W_rec` önce spektral yarıçapı `ρ_spec` olacak şekilde ölçeklenir, sonra sabit bir `γ_rec` ile çarpılır.
- Sebep: Spike'lar ikili ve seyrek, eşik ise kalibrasyon sonrası θ ≈ 7 mertebesinde. `γ_rec = 1` iken bir spike'ın komşusuna katkısı (~0.1) eşiğe göre çok küçük kalıyor, nöronlar arası karışım neredeyse hiç olmuyordu.
- `γ_rec`, Faz 1'de raporlanan seed'lerden **ayrı** bir ayar seed'inde, bellek testinin doğrulama bölümüne göre seçilir. Kararlılık testi (sıfır girdide aktivite sönmeli) her zaman geçmelidir; kendi kendini sürdüren aktivite üreten bir `γ_rec` seçilemez.
**Okuma vektörü r(t).** Ortam adımı sonunda okuma bloklarına (beyincik, duygu uzayı, karar) giden temsil iki parçanın birleşimidir:
```
f(t) = Σ_{i=1..T} s(i) / T          # T iç adım boyunca ortalama ateşleme oranı   [B, N]
v(t) = κ(τ) · V(T)                   # ortam adımı sonundaki zar potansiyelleri     [B, N]
κ(τ) = (τ_ref / τ)^p                 # varyans koruyucu kazanç; τ_ref = τ varsayılanı, p = reservoir.v_gain_exp
r(t) = [f(t), v(t)]                  # okuma vektörü                                [B, 2N]
```
Sebep: Uzun τ'da bellek spike'lardan çok zar potansiyelinde saklanır. Okuma yalnızca spike'lara bakarsa bu bilgi görünmez. Bu dosyada `r(t)` yazan her yer bu birleşik `[B, 2N]` vektörü ifade eder.

**Varyans koruyucu kazanç κ(τ).** `(1−ᾱ)` çarpanı olmadığı için V'nin durağan ölçeği τ ile büyür. Ölçüm (ayar seed'i, train): τ 100 → 300'de |V| ~2,5 kat, std ~1,7 kat artar; θ(τ) neredeyse sabittir, bu yüzden V/θ bunu düzeltmez.
- κ yalnızca okuma vektörüne uygulanır. Dinamik, spike'lar, θ(τ) tablosu ve K7 değişmez. Girdi terimini κ ile ölçeklemek, V'yi ve kalibre θ'yı birlikte ölçekler; spike'lar aynı kalır. Bu yüzden iki yol okuma açısından eşdeğerdir.
- τ = τ_ref'te κ = 1'dir; sabit τ'lu A ve B'nin r'si değişmez.

**Çapa alt-popülasyonu (opsiyonel, Faz 6, `reservoir.anchor_frac`).**
- Sabit seed'le (`anchor_seed_offset`, W_in/W_rec diziliminden ayrı üreteç) seçilen ⌊frac·N⌉ nöronun ortak τ'su regülatörden bağımsız olarak τ_ref'te kalır: τ_i = τ_ref·m_i. Bu nöronlarda θ ve κ da τ_ref'ten okunur.
- θ(τ) tablosu, k-WTA ve K7 değişmez. Tablo nöron başına okunur.
- Amaç: tüm nöronların zaman temsili aynı anda kaydığında okuma için sabit bir referans bırakmak.
- `anchor_frac: 0` (varsayılan) → kapalı, kod yolu aynı.
- p = 0,5, std oranını τ ∈ [30, 500] aralığında 0,86–1,06'da tutar ("varyans koruyucu"). p = 1 (`(1−ᾱ)/(1−ᾱ_ref)`) ortalamayı korur ama varyansı yarıya indirir. p = 0 → kapalı.

**Ateşleme eşiği θ(τ):**
- θ tek bir sayı değil, τ'nun fonksiyonudur. Başlangıçta bir kez, rastgele bir kalibrasyon girdisiyle, her τ değerinde ortalama ateşleme oranı `ρ`'ya yakın kalacak şekilde hesaplanır. Sonra sabit bir tablo olarak saklanır ve τ aralığında log-doğrusal interpolasyonla okunur.
- θ(τ) öğrenilmez ve regülatörler tarafından doğrudan değiştirilmez. Yalnızca τ'nun deterministik bir fonksiyonudur, bu yüzden K1 ve K4'ü ihlal etmez.
- Sebep: `(1−α)` çarpanı kalkınca büyük τ'da zar potansiyeli çok daha fazla birikir. Eşik sabit kalırsa ateşleme oranı τ ile çok değişir (K7 ihlali).
- **Durum: geçici.** Bu bölüm Faz 1 teşhis tablosuyla doğrulanır. Tablo başka bir çözümün daha iyi olduğunu gösterirse önce bu dosya güncellenir.

**Başlatma kuralları:**
- `W_in`: her nöron rastgele `c_in` girdiye bağlı (varsayılan `c_in = 3`), ağırlıklar `N(0, 1)`.
- `W_rec`: seyreklik `p_rec = 0.05`. Spektral yarıçapı `ρ_spec = 0.9` olacak şekilde ölçeklenir (sistemin kararlı kalması için).

### 3.3 Beyincik bloğu
```
x̂(t+1) = W_c · [r(t), a(t)]          # a(t): karar bloğunun çıktısının kopyası
e(t+1) = x(t+1) − x̂(t+1)
ΔW_c   = η · e(t+1) · [r(t), a(t)]ᵀ
```

**Normalize edilmiş çevrimiçi kural (varsayılan).** Ham delta kuralı (yukarıdaki) `φ = [r, a]` için `‖φ‖² ~ 1e6` olduğundan, `η < ~1e-7` dışında ıraksar (Faz 2). Varsayılan kural, aynı yerel yapıyı korur ama girişi nedensel olarak standartlaştırır ve adımı normalize eder:
```
φ̃     = [(φ − μ) / √(σ² + ε), b]          # μ, σ²: her öznitelik için geçmişin üstel ortalaması/varyansı (yalnızca o ana kadarki veriden)
ΔW_c   = η · e(t+1) · φ̃ᵀ / (ε + ‖φ̃‖²)     # normalize LMS
```
- `b = √F` sabit bias girişidir (`bias_scale: auto`); `F`, öznitelik sayısıdır.
- Hâlâ yereldir: bir ağırlığın değişimi yalnızca kendi girdisine ve e'ye bağlıdır; `μ`, `σ²` gelecekten bilgi taşımaz (K6). Yalnızca `W_c` değişir.
- Aynı kural karar bloğu için de geçerlidir (Bölüm 3.6), ortak yardımcı kullanılır.
- Ham kural config'te `cerebellum.rule: plain` ile hâlâ seçilebilir (karşılaştırma için).

### 3.4 Duygu uzayı
```
z(t) = tanh(W_z · r(t))                     # şekil: [B, k], varsayılan k = 3
```
z'nin üç tahmin başı (head) vardır. Her biri z'den bir değer tahmin eder:

| Baş | Hedef | Anlamı |
|---|---|---|
| `q̂` | `q = −‖δ(t)‖²` | Anlık performans |
| `û` | `u = ‖e(t)‖²` | Belirsizlik (beyincik ne kadar yanılıyor) |
| `ŝ` | `s = EMA_kısa(‖e‖²) − EMA_uzun(‖e‖²)` | Eğilim (iyileşme mi, kötüleşme mi) |

`W_z` ve başların ağırlıkları, bu üç tahminin toplam hatasıyla delta kuralıyla güncellenir. Başlardan W_z'ye hata iletimi yalnızca tek bir doğrusal katman üzerinden yapılır (yerel kalır).

**Uygulama ayrıntıları** (`src/affect.py`):
- `r` girdisi beyincik ve kararla aynı nedensel standartlaştırmadan geçer: `z = tanh(W_z · φ̃(r))`.
- Başlar doğrusaldır: `[q̂, û, ŝ] = W_h · [z, 1]`. q, u, s'nin ölçekleri çok farklı olduğu için her hedef çevrimiçi (nedensel) standartlaştırılır; başlar standart hedefi tahmin eder.
- `ε_h` = standart hedef − baş çıktısı. Güncellemeler normalize LMS biçimindedir:
  ```
  ΔW_h = η_h · ε_h · [z, 1]ᵀ / (ε + ‖[z, 1]‖²)
  ΔW_z = η_z · (((W_hᵀ ε_h)[:k] ⊙ (1 − z²)) − λ · a) · φ̃ᵀ / (ε + ‖φ̃‖²),   a = W_z · φ̃
  ```
- `λ · a` terimi, ön-aktivasyon üzerindeki `λ/2 · ‖a‖²` cezasının yerel türevidir (`affect.preact_l2`, varsayılan 0,03). Sebep: cezasız z'nin %36–68'i ±1'e doyuyor, `1 − z² ≈ 0` olunca W_z öğrenemiyordu. Ceza doygunluğu %1–3'e indirir (KR-021).
- `η_z`, `η_h` config'ten gelir (`affect.lr_z`, `affect.lr_head`); η regülatörü yalnızca karar ve beyinciği etkiler (K4).
- `W_z` kendi seed'li üreteciyle başlatılır; global RNG'ye dokunmaz. Böylece duygu uzayını açmak diğer blokların sonuçlarını değiştirmez.

### 3.5 Regülatörler

**Duygu dinamiği (E, M, R).** Regülatörler z(t)'yi doğrudan okumaz. z anlık bir durumdur ve her adımda sıçrar (KR-024). Araya üç katmanlı, sürekli zamanlı bir dinamik girer. Esin kaynakları: DynAffect (homeostatik geri çekilme), WASABI (duygu–duygudurum çift zaman ölçeği), CoMERG/Gross (kapalı döngü regülasyon, karşıt süreç).
- `E` (fazik duygu): hızlı, kısa vadeli durum. Regülatörlerin girdisidir.
- `M` (tonik duygudurum): yavaş birikim; E için dinamik denge noktası.
- `R` (karşıt süreç): E ile M arasındaki sapmayı gecikmeli olarak bastırır.

Her adımda, hepsi `[B, k]`, eleman bazında:
```
z_in   = z − μ_z;   μ_z ← μ_z + c_μ · (z − μ_z)            # girdi merkezleme (nedensel kayan ortalama)
error  = E − M
drive  = max(0, |error| − θ_R) · sign(error)                 # eşik altı sapmaya tepki yok
R      ← (1 − γ_decay) · R + γ_up · (1 − |R|) · drive
λ_eff  = λ_max · (1 − κ · |M|)                               # kronik duygudurum yükü regülatörü zayıflatır
E      ← E + α_up · (1 − |E|) · z_in − α_down · (E − M) − λ_eff · R
M      ← (1 − β_M) · M + β_M · E
```
- E ve R yalnızca sonda sayısal güvenlik için `[−1, 1]`'e kırpılır. Adım başına değişim hızı sınırlanmaz. Seyrek güncelleme yoktur; dinamik ve regülatör çıktıları her adım güncellenir.
- Kısıtlar: `α_up > α_down` (duygu, söndüğünden hızlı yükselir); `γ_up < α_up` (önce duygu tepki verir, sonra regülatör dizginler).
- Dinamik öğrenmez (sabit kural, `update` yok). δ ve e'yi görmez, yalnızca z'yi okur (K3). Durum yeni dizide sıfırlanır (K6).
- Girdi merkezleme şart: z'nin ortalaması sıfır değildir. Merkezlemesiz E, ortalamayı biriktirir ve M ile birlikte ±1'e kayar. Faz 4 izlerinde E zamanın %54–98'inde doygundu; merkezlemeyle %0–28 (KR-026).
- Ad ayrımı: θ_R, β_M, α_up/α_down, γ_up/γ_decay, λ_max ve κ yalnızca bu dinamiğe aittir. θ(τ) eşiği, β karar sıcaklığı, α_i sızıntısı, γ_rec, preact λ ve perturbasyon periyodu M ile karıştırılmamalı. Config'te `affect.dynamics` altında tutulur.
- `affect.dynamics.enabled: false` iken regülatörler z'yi doğrudan okur (eski davranış, ablation için).

**Regülatör eşlemesi:**
```
ham     = σ(W_r · E(t) + b_r)                # şekil: [4]; dinamik kapalıyken E yerine z
g, η, β = alt + ham · (üst − alt)
τ       = exp(log τ_alt + ham_τ · (log τ_üst − log τ_alt))
```
**Asimetrik τ eşlemesi (`regulators.tau_ceiling: c`, null → yukarıdaki simetrik sigmoid).**
```
a       = (W_r · E(t))_τ                                  # bias yok: a = 0 → τ = τ_varsayılan
log τ   = log τ_0 + Δ_alt · tanh(a)     (a < 0),   Δ_alt = log(τ_0 / τ_alt)
log τ   = log τ_0 + Δ_üst · tanh(a)     (a ≥ 0),   Δ_üst = log(min(c · τ_0, τ_üst) / τ_0)
```
- Kısa τ'ya (hızlı unutma) alt sınıra kadar izin verilir; uzun τ `c · τ_0` tavanında doyar (varsayılan c = 1,2 → 120 ms).
- Sebep: KR-029'da uzun τ'ya kayma val'de yıkıcıydı (Loco −4,68), kısa τ'ya kayma zararsızdı. Uzun τ, V ölçeğini büyütür ve train'de hiç ateşlemeyen nöronları ateşletir. Yu & Dayan (2005) esinli yorum: beklenmeyen belirsizlikte (dağılım kayması) eski bilgiyi daha hızlı unutmak doğru yöndür, belleği uzatmak değil.
- W_r'ye ceza ya da norm sınırı yoktur; yalnızca τ'nun çıkış eşlemesi asimetriktir. g, η, β eşlemesi değişmez.

**W_r eğitimi (perturbasyon yöntemi):**
1. Her `M` adımda bir `W_r`'ye küçük bir rastgele gürültü eklenir.
2. Sonraki `M` adımın ortalama `q`'su öncekiyle karşılaştırılır.
3. İyileşme varsa değişiklik tutulur, yoksa geri alınır.

**Eşli karşılaştırma (varsayılan, `perturbation.paired: true`).** Ham q'nun pencereden pencereye değişimi büyük ölçüde verinin zorluğundan gelir (KR-028: kabul oranı ~%50, W_r rastgele yürüyor). Bu yüzden C'nin yanında regülatörsüz bir **gölge ajan** koşar: aynı sabit rezervuarın ayrı bir kopyası (varsayılan g, τ) ve kendi karar bloğu (konfigürasyon A). Aynı x(t), y(t)'yi aynı anda görür ve kendi δ'sıyla öğrenir. Perturbasyon kararı ham q yerine farkla verilir:
```
Δq(t) = q_C(t) − q_gölge(t),   q = −‖δ‖²
tut ⇔ ortalama Δq (deneme) > ortalama Δq (referans)
```
- Gölge yalnızca karşılaştırma sinyali üretir. C'nin hiçbir bloğuna girdi vermez, C'nin δ ve e'sini değiştirmez (K3). Rezervuar kopyası K1'e uyar.
- Gölge, A ile aynı kuruluma sahiptir (beyincik yok). K3'e göre beyincik karar çıktısını etkilemediği için q açısından B ile aynıdır.
- `paired: false` iken eski davranış (ham q) kullanılır.

Perturbasyon penceresi M yalnızca W_r'nin öğrenme protokolüdür. Regülatör çıktıları her adım güncellenir, M adımda bir değil.

### 3.6 Karar bloğu
- **Regresyon görevleri** (örneğin primat motor tahmini): `ŷ(t) = W_a · r(t)`. β kullanılmaz.
- **Ayrık görevler** (örneğin SSC, Crafter): `p(a) = softmax(β · W_a · r(t))`.
```
δ(t)  = y(t) − ŷ(t)
ΔW_a  = η · δ(t) · r(t)ᵀ
```
- Varsayılan kural, beyincikle aynı **normalize çevrimiçi LMS**'dir (Bölüm 3.3): `ŷ = W_a · φ̃(r)`, `ΔW_a = η · δ · φ̃ᵀ / (ε + ‖φ̃‖²)`. Ham kural config'te `decision.rule: plain` ile seçilebilir. Her iki blok aynı yardımcıyı kullanır (`src/utils/online_norm.py`).

**Hızlı-yavaş okuma (Faz 6, `decision.fast`).** Çerçeve: kararlılık-esneklik ikilemi (Hinton & Plaut 1987, Zenke vd. 2015). Değerlendirmede karar ağırlığı iki bileşene ayrılır:
```
W_t        = W_base + ΔW_t          W_base: train sonunda W_a, değerlendirmede hiç değişmez;  ΔW_0 = 0
ŷ_t        = W_t · φ̃_t              (skor bu tahminden; y_t henüz görülmedi)
e_t        = y_t − ŷ_t
ΔW_{t+1}   = (1 − λ_pull) · ΔW_t + η_t · e_{t−d+1} · φ̃_{t−d+1}ᵀ / (‖φ̃_{t−d+1}‖² + ε)      (d ≥ 1)
‖ΔW‖_F > ρ · ‖W_base‖_F  ⇒  ΔW ← ΔW · ρ‖W_base‖_F / ‖ΔW‖_F
η_t        = min(η_base + η_mod · gate_t, η_üst),   gate_t = min(1, |log(τ_t/τ_0)| / log(gate_scale))
```
- φ̃_t, nedensel normalize edilmiş ve κ(τ) uygulanmış okuma vektörüdür (§3.2, §8).
- η_t üç faktörlü kuraldaki nöromodülatör kapısıdır (Frémaux & Gerstner 2016). τ_t regülatörün E_{t−1}'den ürettiği τ olduğu için kapı duygu durumuna bağlıdır. `eta_mode: fixed` → η_t = η_base.
- K3 korunur: ΔW yalnızca karar bloğunun δ'sı ile öğrenir. Beyincik, duygu uzayı ve regülatör ağırlıkları değerlendirmede donuk kalır.
- `enabled: false` (varsayılan) → Faz 5 protokolü. η_t ≡ 0 iken çıktı Faz 5 ile bit düzeyinde aynıdır (`tests/test_faz6.py`).

---

## 4. Modül yapısı

```
snn_affect/
├── configs/
│   ├── base.yaml
│   ├── chaotic.yaml
│   └── nhp_motor.yaml
├── src/
│   ├── encoder.py        # SpikeEncoder (delta + seviye)
│   ├── reservoir.py      # LIFReservoir (eğitilmez)
│   ├── cerebellum.py     # Cerebellum
│   ├── affect.py         # AffectSpace + Regulators
│   ├── decision.py       # DecisionReadout
│   ├── agent.py          # Tüm blokları bağlayan ana döngü
│   └── utils/
│       ├── seeding.py
│       ├── logging.py
│       └── metrics.py
├── experiments/
│   ├── run.py            # python experiments/run.py --config configs/xxx.yaml
│   └── ablation.py       # A / B / C konfigürasyonlarını sırayla koşar
├── tests/
│   ├── test_reservoir.py
│   ├── test_cerebellum.py
│   ├── test_affect.py
│   ├── test_decision.py
│   └── test_invariants.py   # K1–K7 kurallarını otomatik kontrol eder
├── ARCHITECTURE.md
└── PHASES.md
```

## 5. Arayüzler

Her modül aynı üç metodu sağlar:
- `forward(...)`: hesaplama yapar.
- `update(...)`: kendi ağırlıklarını günceller (yerel kural).
- `reset_state()`: diziler arası durumu sıfırlar.

| Modül | `forward` girdisi | `forward` çıktısı | `update` girdisi |
|---|---|---|---|
| `SpikeEncoder` | `x: [B, D_in]` | `u: [B, T, C]`, `C = (2+L)·D_in` | — |
| `LIFReservoir` | `u: [B, T, C]`, `g`, `τ` | `r: [B, 2N]` | — (yasak, K1) |
| `Cerebellum` | `r: [B, 2N]`, `a: [B, D_out]` | `x̂: [B, D_in]` | `e`, `η` |
| `AffectSpace` | `r: [B, 2N]` | `z: [B, k]`, başlar | `q`, `u`, `s` |
| `Regulators` | `z: [B, k]` | `{g, τ, η, β}` | perturbasyon sonucu |
| `DecisionReadout` | `r: [B, 2N]`, `β` | `ŷ: [B, D_out]` | `δ`, `η` |

## 6. Bir adımın akışı (`agent.step`)

1. `u = encoder(x_t)`
2. `g, τ, η, β = regulators(E_{t−1})` (ilk adımda varsayılanlar kullanılır)
3. `r_t = reservoir(u, g, τ)`
4. `ŷ_t = decision(r_t, β)` → çıktı
5. `x̂_{t+1} = cerebellum(r_t, ŷ_t)`
6. Hedef `y_t` ve sonraki girdi `x_{t+1}` gelir.
7. `δ_t = y_t − ŷ_t` → `decision.update(δ_t, η)`
8. `e_{t+1} = x_{t+1} − x̂_{t+1}` → `cerebellum.update(e_{t+1}, η)`
9. `z_t = affect(r_t)` → `affect.update(q, u, s)`; `E_t = dynamics(z_t)` (§3.5)
10. Her `M` adımda: `regulators.update(...)` (perturbasyon kontrolü)

## 7. Varsayılan hiperparametreler

| Parametre | Değer |
|---|---|
| `N` (rezervuar boyutu) | 1000 |
| `dt` | 1 ms |
| `T` (iç adım sayısı) | 20 |
| `ρ` (k-WTA oranı) | 0.05 |
| `θ(τ)` (ateşleme eşiği) | Kalibrasyon tablosu (Bölüm 3.2); başlangıç tahmini 1.0 |
| `c_in` | 3 |
| `p_rec` | 0.05 |
| `ρ_spec` | 0.9 |
| `θ_enc` (delta eşiği) | 0.1 |
| `L` (girdi boyutu başına seviye kanalı) | 8 |
| `[x_alt, x_üst]` (seviye aralığı) | Veri setine göre; bellek testinde [0, 1] |
| `S` (nöron τ yayılımı, `m_i ∈ [1/S, S]`) | 10 |
| `γ_rec` (rekürrent kazanç) | Faz 1 ayar seed'inde seçilir (Bölüm 3.2) |
| Sıfırlama | Yumuşak: `V ← V − θ` |
| `k` (duygu uzayı boyutu) | 3 |
| `M` (perturbasyon periyodu) | 500 adım |
| Perturbasyon gürültüsü | `N(0, 0.01)` |

## 8. Değerlendirme kuralları

Her deney üç konfigürasyonla koşulur:

| Kod | İçerik |
|---|---|
| **A** | Rezervuar + karar bloğu (temel çizgi) |
| **B** | A + beyincik bloğu |
| **C** | B + duygu uzayı + regülatörler (tam sistem) |

**Raporlanacak metrikler:**
- Görev metriği (regresyonda R², sınıflandırmada doğruluk).
- Kayma testi: eğitim oturumlarından **sonraki** oturumlardaki performans düşüşü.
- Ortalama ateşleme oranı.
- Regülatörlerin zaman içindeki değerleri (g, τ, η, β grafikleri).
- z uzayının görselleştirmesi (k = 3 ise doğrudan 3B çizim).

**Değerlendirmede normalizasyon (`agent.freeze_norm_eval`).** Değerlendirmede ağırlıklar her zaman donuktur. `freeze_norm_eval: false` iken okuma bloklarının girdi standartlaştırıcıları (karar, beyincik, duygu uzayının r normu) etiketsiz ve nedensel olarak güncellenmeye devam eder:
```
μ_t = (1−ρ_n)·μ_{t−1} + ρ_n·r_t,   σ²_t = (1−ρ_n)·σ²_{t−1} + ρ_n·(r_t − μ_t)²,   r̃_t = (r_t − μ_t)/√(σ²_t + ε)
```
- Yalnızca r_1..r_t kullanılır (gelecek yok, batch istatistiği yok). Hedef (y, q, u, s) istatistikleri ve tüm ağırlıklar donuk kalır.
- Bu, sinaptik ağırlıklardan bağımsız intrinsik homeostatik ölçeklemedir: Turrigiano (2008) sinaptik ölçekleme ve Carandini & Heeger (2012) bölücü normalizasyon esinli. Etiketsiz duyusal adaptasyona (ör. göz bebeği refleksi) benzer (KR-030).
- Karşılaştırmalar aynı ayarla yapılır: A, B, B+Z ve C ya hepsi donuk ya hepsi nedensel normalizasyonla.

**Prequential değerlendirme (Faz 6).** `decision.fast.enabled: true` iken değerlendirme "önce tahmin et, sonra uyum sağla" sırasıyla akar:
1. ŷ_t üretilir ve metriğe yazılır.
2. Ancak bundan sonra e_t hesaplanıp kuyruğa girer. d adım sonra ΔW'ye uygulanır.

ŷ_t, y_t'ye ve sonrasına bağlı olamaz (`test_zero_leakage`). Yalnızca ΔW öğrenir (§3.6). Test bölmesinde hiperparametre seçilmez.

**Ana iddianın kanıtı:** C'nin, kayma testinde A ve B'den istatistiksel olarak anlamlı biçimde iyi olması (5 seed, t-testi ya da Wilcoxon testi).

## 9. Kodlama kuralları

- Python 3.11+, PyTorch. snnTorch yalnızca yardımcı fonksiyonlar için kullanılabilir. Rezervuar kendi kodumuzla yazılır, çünkü K1 ve k-WTA üzerinde tam kontrol gerekir.
- Tüm fonksiyonlarda tip ipuçları (type hints) ve tensör şekli yorumları zorunludur: `# [B, N]`.
- Yeni bir modül ancak testi yazıldıktan sonra `agent.py`'ye bağlanır.
- `tests/test_invariants.py` her commit öncesi geçmelidir.

## 10. Yasaklar

- Projeksiyon bloğu ağırlıklarını güncellemek (K1).
- δ'yı beyinciğe ya da e'yi karar bloğuna vermek (K3).
- Regülatörlerin dört parametre dışında bir şeyi değiştirmesi (K4).
- Beyinciğin tahmin anında gelecekteki girdiyi görmesi (K6).
- Kod içine sabit hiperparametre yazmak (K8).
- Test seti üzerinde regülatör ya da hiperparametre ayarı yapmak.
- Okuma bloklarına yalnızca spike oranını (`f`) vermek. Okuma vektörü her zaman `[f, v]`'dir (Bölüm 3.2).

---

## 11. Değişiklik kaydı

| Tarih | Değişiklik | Sebep |
|---|---|---|
| 2026-11-07 | §12.8 Aşama 3: çevrimiçi etiket 500 ms gecikmeli; günler arası model yalnızca Day-0'da eğitilir | KR-041: anında etiketle y_{t−1} kopyası R² 0,999 (uyum kazancı sızıntıya yakın). KR-040: ön-eğitim Day-k train dağılımını görüyordu |
| 2026-11-03 | Spike v2 hattı eklendi (§12): çevrimdışı eğitilen hızlı/yavaş omurga, Anscombe kodlayıcı, yumuşatılmış hedefli beyincik, sürpriz modülasyonlu η + alt-popülasyon kapılama, 5 başlıklı çapalı okuma; v2 kuralları KV1–KV8. v1 kuralları v1 koduna uygulanmaya devam eder | Kullanıcı spesifikasyonu (`docs/spike_v2_mimari_spesifikasyon.md`, KR-034). KR-033: kayıp doğrusal okunan temsilde (GRU tavanı +0,3 R²); sabit rezervuar bu farkı kapatamıyor |
| 2026-11-03 | K2 (global geri yayılım yok) kaldırıldı; numara boş bırakıldı | Kullanıcı kararı (KR-033). Tavan ölçümü için BPTT ile eğitilen GRU referansı gerekiyor; okuma blokları yerel kalır |
| 2026-11-02 | Faz 6: değerlendirmede hızlı-yavaş okuma W_base + ΔW (elastik çapa, NLMS, norm sınırı), η_t nöromodülatör kapısı (§3.6), prequential protokol (§8), opsiyonel çapa alt-popülasyonu (§3.2) | Kullanıcı spesifikasyonu (KR-031). Faz 5'te donuk okuma ile C ≈ B+Z; hipotez: τ'nun yarattığı temsil kayması okuma uyumu gerektiriyor |
| 2026-11-02 | Değerlendirmede nedensel homeostatik normalizasyon (`freeze_norm_eval: false`, §8); asimetrik τ eşlemesi (`tau_ceiling`, §3.5); okuma vektöründe varyans koruyucu κ(τ) (§3.2) | KR-029: regülatörlerin τ'yu kaydırması donuk okuma için temsil kaymasına dönüşüyordu (KR-030) |
| 2026-11-01 | Perturbasyon eşli karşılaştırmayla: Δq = q_C − q_gölge (§3.5) | Ham q ile kabul oranı ~%50; W_r rastgele yürüyor, z okuyan C Loco'da 3/3 seed kötü (KR-028) |
| 2026-11-01 | Regülatörler z yerine E/M/R duygu dinamiğinin E'sini okur (§3.5); z girdisi merkezlenir | z her adımda sıçradığı için g/τ/η her 20 ms'de titriyordu (KR-024). E/M/R titremeyi 10–30 kat azaltır; merkezlemesiz E doygun kalıyordu (KR-026) |
| 2026-10-24 | `(1−α)` çarpanı kaldırıldı | τ büyüdükçe girdi gücü küçülüyor, nöronlar ateşlemiyordu (τ=100'de ateşleme oranı %0.08) |
| 2026-10-24 | k-WTA "tam k" yerine "eşiği geçenlerden en fazla k" oldu | "Tam k" kuralı ile "girdi yokken aktivite sönsün" testi birbiriyle çelişiyordu |
| 2026-10-24 | Okuma vektörü `[f, v]` (`[B, 2N]`) oldu | Bellek zar potansiyelinde duruyor, yalnızca spike'lara bakan okuma onu göremiyordu |
| 2026-10-24 | τ aralığı 5–100 ms'den 20–2000 ms'ye, varsayılan 100 ms oldu | Bir ortam adımı 20 ms sürüyor, eski aralıkta bellek en fazla ~1–2 adım olabilirdi |
| 2026-10-24 | θ sabitten θ(τ) tablosuna geçti (geçici) | `(1−α)` kalkınca ateşleme oranının τ ile değişmesini engellemek için; Faz 1 teşhisiyle doğrulanacak |
| 2026-10-24 | Nöron başına zaman sabitleri `τ_i = τ · m_i` ve göreli denge katsayısı `c_i` eklendi | Bellek testi: tüm nöronlar aynı τ'yu kullanınca bellek her τ'da ~2 adımda bitiyordu; aynı filtreyi paylaşan nöronlardan gecikmeler ayrıştırılamıyor |
| 2026-10-24 | Sıfırlama `V ← 0`'dan `V ← V − θ`'ya (yumuşak) geçti | ~%4 ateşleme oranında sıfıra çekme, aktif nöronların belleğini her ~1.4 ortam adımında siliyordu |
| 2026-10-24 | Rekürrent kazanç `γ_rec` eklendi | Seyrek ikili spike'lar ve θ ≈ 7 ile `W_rec` katkısı eşiğe göre çok küçüktü, nöronlar arası karışım yoktu |
| 2026-10-30 | W_z güncellemesine ön-aktivasyon L2 terimi `− λ·a` eklendi (§3.4, λ = 0,03) | z'nin %36–68'i doymuştu; Loco'da u başı öğrenmiyordu. Ceza ile doygunluk %1–3, üç baş da her iki ayar oturumunda öğreniyor (KR-021) |
| 2026-10-29 | Duygu uzayı uygulama ayrıntıları (§3.4): standartlaştırılmış girdi ve hedefler, normalize LMS, `W_z` güncelleme formülü | Hedeflerin ölçekleri çok farklı; ham delta kuralı `r` ölçeğinde ıraksar (KR-019) |
| 2026-10-29 | Karar bloğu da normalize LMS kullanır (§3.6) | Ham kural `‖r‖² ~ 1e6` ile ıraksıyor (Faz 2, KR-011) |
| 2026-10-29 | Seviye aralığı kanal başına olabilir (§3.1, `train_per_channel`) | Sinir kanallarının ateşleme oranları çok farklı; tek aralıkta sakin kanallar hiç seviye biti yakmıyor (KR-013) |
| 2026-10-28 | Beyincik varsayılan kuralı normalize edilmiş çevrimiçi LMS oldu (§3.3) | Faz 2: ham delta kuralı `‖φ‖² ~ 1e6` ile ıraksadı (5 seed'de NaN); normalize kural MSE'yi "önceki değeri tekrarla"nın %17.9'una indirdi, ridge'e göre fark %6.8 |
| 2026-10-24 | Kodlayıcı `DeltaEncoder` → `SpikeEncoder`: delta kanallarına seviye (termometre) kanalları eklendi | Delta kodlama x'in seviyesini iletmiyordu; ideal bir hafızayla bile pencere ortalaması geri çıkarılamıyordu (Bölüm 3.1) |

---

## 12. Spike v2 — Ön-eğitimli omurga + nöromodüle çevrimiçi yerel uyum

Kaynak: `docs/spike_v2_mimari_spesifikasyon.md` (kullanıcı spesifikasyonu, 2026-11-03). Fazlar: `docs/PhasesV2.md`. Karar: KR-034.
Bu bölüm spesifikasyonun **kural ve denklem** özetidir; referans kod spesifikasyon dosyasındadır. Spesifikasyondaki sayılar başlangıç değerleridir ve ölçümle (KR kaydıyla) değişebilir. §12.10'daki açık noktalar ilgili fazda kapatılır.

### 12.1 Amaç ve araştırma sorusu
- Çevrimdışı (surrogate gradyan / BPTT) eğitilen, hızlı/yavaş alt-popülasyonlu bir zamansal omurga; üstünde yalnızca yerel kurallarla çevrimiçi uyum sağlayan okuma blokları.
- Hedef: NeuroBench Primate Reaching oturum içi test R² ≥ 0,65–0,70. Referans: KR-033 GRU tavanı (20 ms'de Indy 0,736 / Loco 0,685).
- **Araştırma sorusu (v2):** Beklenmedik sürprizle modüle edilen öğrenme hızı η(t) ve hızlı/yavaş alt-popülasyon kapılaması, günler arası kaymada ve elektrot kaybında, aynı ortalama η'lı sabit kuraldan (v2-B) ve zamanlaması karıştırılmış η'dan (v2-C) daha iyi uyum sağlıyor mu?

### 12.2 v2 değişmez kuralları (KV1–KV8)

**KV1 — Omurga çevrimdışı eğitilir, çevrimiçi donuktur.**
- Omurga ağırlıkları (`plif`: W_in, W_rec; GRU modları: kapı matrisleri) ve zaman sabitleri τ_mem,i, τ_syn,i yalnızca Aşama 1–2'de (§12.8) gradyanla öğrenilir. Başlatma sabit seed'lidir; ağırlıklar checkpoint'e kaydedilir.
- Aşama 3'te (çevrimiçi akış) omurganın tüm parametreleri `requires_grad=False`'tur ve hiçbir güncelleme onlara dokunmaz. τ_i çevrimiçi **değişmez**: v1'deki τ regülatörü v2'de yoktur.
- Çevrimiçi değişebilen tek omurga durumu homeostatik eşiktir: `ϑ_i ← clip(ϑ_i + η_homeo·(r̄_i − ρ), ϑ_min, ϑ_max)`. Etiketsiz, nöron başına yerel ve nedenseldir (v1 KR-030 nedensel homeostatik normalizasyonunun karşılığı).

**KV2 — Çevrimiçi öğrenme yereldir.** Aşama 3'te gradyan, BPTT ya da optimizer yoktur. Yalnızca açık formüllü kurallar çalışır: beyincik NLMS'i (§12.5), çapalı okuma NLMS'i (§12.7), homeostatik eşik (KV1). Çevrimdışı aşamalarda gradyan serbesttir (K2, KR-033'te kaldırıldı).

**KV3 — Öğrenme sinyalleri karışmaz.**
| Sinyal | Tanım | Kimi eğitir |
|---|---|---|
| **e_v** (sonuç hatası, v1'deki δ) | v(t) − v̂(t) | Çevrimiçi: yalnızca okuma başlıkları. Çevrimdışı: omurga + başlıklar (hız kaybı) |
| **e_cer** (tahmin hatası, v1'deki e) | x̄(t+1) − x̂_smooth(t+1) | Yalnızca W_cer |
- Beyincik kaybı omurgaya gradyan vermez. Duygu uzayı e_cer'i yalnızca **okur**, değiştirmez.

**KV4 — Regülatörler yalnızca okuma kapılama kazançlarını ve η'yı üretir.**
| Regülatör | Etkilediği yer | Aralık | Varsayılan |
|---|---|---|---|
| g_fast, g_slow | Okuma izlerinin hızlı / yavaş yarısı (§12.7) | [1 − α_gate, 1 + α_gate] ⊂ [0,5; 2] | 1 |
| η(t) | Okuma başlıklarının çevrimiçi NLMS adımı | [η_0, η_0·(1 + α_η)] ⊂ [1e-4; 1e-1] | η_0 |
- τ, ϑ, omurga ağırlıkları, beyincik adımı ve başka hiçbir hiperparametre regülatörden etkilenmez. β (karar sıcaklığı) regresyonda kullanılmaz.

**KV5 — Regülatörsüz mod her zaman çalışır.** `regulators.enabled: false` → g_fast = g_slow = 1 ve η sabit (v2-A'da 0, v2-B'de η̄). Ablation için zorunludur ve testlidir.

**KV6 — Zaman ve sızıntı.**
- Bir veri adımı = bir omurga adımı; Δt config'ten gelir (varsayılan 4 ms). v1'deki T iç adım yoktur.
- Akış değerlendirmesinde durum sıfırlanmaz. Çevrimdışı eğitimde her pencere yeni bir dizidir.
- Beyincik x̂_smooth(t+1)'i üretirken x(t+1)'i ve x̄(t+1)'i göremez.
- Prequential sıra: v̂(t) üretilir ve skorlanır; güncelleme ancak sonra e_v(t) ile yapılır.
- Kodlayıcı istatistikleri (μ_ch, σ_ch) ve tüm ölçekler yalnızca train diliminden hesaplanır.

**KV7 — Ateşleme rejimi (k-WTA yerine homeostaz).**
- v2 omurgasında k-WTA yoktur. Hedef ortalama oran ρ = 0,05 (config `target_rate`).
- Çevrimdışı: homeostatik kayıp oranı [f_min, f_max] = [0,02; 0,08] bandına iter. Çevrimiçi: KV1'deki eşik adaptasyonu.
- Ortalama ateşleme oranı her koşuda **alt-popülasyon başına** (hızlı / yavaş) loglanır. Değerlendirmede [f_min, f_max] dışı oran hata olarak raporlanır.
- `tau_gru` modu (sürekli aktivasyon, üst sınır kontrolü) bu kuraldan muaftır.

**KV8 — Tekrarlanabilirlik.** K8 aynen geçerlidir: hiperparametreler YAML'dan, ≥5 seed, ortalama ± std, config + seed + git hash + metrik. Ek olarak:
- Ön-eğitim, Day-0 ve W_0 checkpoint'lerinin yolu ve hash'i koşu kaydına yazılır.
- Spesifikasyon kodundaki sabitler config'e taşınır: Anscombe sabiti 3/8, kırpma ölçeği 3, τ_syn aralığı 4–100 ms, surrogate ölçeği 10, EMA katsayıları, ϑ sınırları 0,3–2,5.
- Test bölmesinde hiperparametre/regülatör ayarı yapılmaz.

### 12.3 Kodlayıcı v2 (`SpikeEncoderV2`)
```
x_ans(t)  = √(max(0, x_raw(t)) + 3/8)                 # Anscombe: Poisson varyansını sabitler    [B, C]
z(t)      = (x_ans(t) − μ_ch) / (σ_ch + ε)             # μ_ch, σ_ch yalnızca train'den             [B, C]
x_norm(t) = c · tanh(z(t) / c),  c = 3                 # yumuşak kırpma                             [B, C]
x̄(t)     = α_sm · x̄(t−1) + (1 − α_sm) · x_norm(t)    # α_sm = exp(−Δt / τ_smooth), τ_smooth = 48 ms; beyincik hedefi
```
- Çıktı sürekli değerlidir; v1'in delta + termometre kanalları (§3.1) v2'de kullanılmaz. C ham kanal sayısıdır (Indy 96, Loco 192).
- Sebep: M1/S1 kanallarının taban oranı ve patlama varyansı farkı, mimariyi bölmeden veri katmanında dengelenir.

### 12.4 Zamansal omurga (`TemporalBackbone`)
- N = 1000 nöron. **Hızlı** yarı (i < N/2): τ_mem ∈ [4, 40] ms. **Yavaş** yarı (i ≥ N/2): τ_mem ∈ [40, 400] ms. `τ_mem = τ_min + (τ_max − τ_min)·σ(λ_i)`, λ_i öğrenilir. τ_syn ∈ [4, 100] ms aynı biçimde.
- Modlar (config `backbone.mode`):
  1. `plif`:
     ```
     I(t) = α_syn ⊙ I(t−1) + W_in·x(t) + W_rec·s(t−1)
     V(t) = β_mem ⊙ V(t−1) + (1 − β_mem) ⊙ I(t) − ϑ ⊙ s(t−1)      # yumuşak sıfırlama
     s(t) = H(V(t) − ϑ)                                            # geri yönde hızlı-sigmoid surrogate
     ```
     β_mem = exp(−Δt/τ_mem), α_syn = exp(−Δt/τ_syn). Başlatma: W_in ~ N(0, 1/C), W_rec ortogonal (kazanç 0,9); dalgalanma güdümlü başlatma (Rossbroich vd. 2022).
  2. `spiking_gru`: GRU kapıları iç durumu günceller; dışarıya ve rekürrent bağlantıya yalnızca ikili s(t) gider. Güncelleme kapısı bias'ı τ_mem dağılımından chrono başlatmayla gelir.
  3. `tau_gru`: eşiksiz, sürekli aktivasyonlu kontrol modu (üst sınır).
- Eğitimde girdi ve s(t−1) üzerinde dropout (p = 0,3).
- Homeostatik kayıp: `L_homeo = mean_i [relu(f_min − r̄_i)² + relu(r̄_i − f_max)²]`; r̄_i pencere ve batch üzerinde ortalama orandır.
- Omurga çıktısı v1'deki `r = [f, v]` değil, s(t) ∈ {0,1}^N'dir (`tau_gru`'da sürekli). Yoğun okuma izleri §12.7'deki sızıntılı entegratörlerle elde edilir.

### 12.5 Beyincik v2 (`CerebellumV2`)
```
φ(t)            = [r̄(t), v̂(t−1)]                  # r̄: okuma izlerinin ortalaması [B, N]; v̂(t−1): efference copy [B, 2]
x̂_smooth(t+1)  = W_cer · φ(t)                      # [B, C]
e_cer(t+1)      = x̄(t+1) − x̂_smooth(t+1)
ΔW_cer          = η_cer · e_cer(t+1) · φ(t)ᵀ / (‖φ(t)‖² + ε)      # yerel NLMS; η_cer sabit (KV4)
```
- Sebep: ham x(t+1) büyük ölçüde Poisson gürültüsüdür; yumuşatılmış hedef, sürpriz sinyalini gürültüden ayırır. Efference copy, v1'deki a(t) = ŷ(t)'nin karşılığıdır.

### 12.6 Duygu uzayı v2 ve regülatörler (`AffectSpaceV2`)
Yu & Dayan (2005) beklenmedik belirsizlik (noradrenalin) esinli:
```
ε²(t)      = mean ‖e_cer(t)‖²
E_fast    ← (1 − a_f)·E_fast + a_f·ε²(t)                 # a_f = 0,05
E_slow    ← (1 − a_s)·E_slow + a_s·ε²(t)                 # a_s = 0,001
a(t)       = tanh((E_fast − E_slow) / (E_slow + ε))      # arousal: beklenmedik sürpriz ∈ (−1, 1)
valence    = −a(t)
dominance ← 0,95·dominance + 0,05·tanh(10·(E_fast(t−1) − E_fast(t)) / (E_slow + ε))
g_fast     = 1 + α_gate·a(t),   g_slow = 1 − α_gate·a(t)          # α_gate = 0,25
η(t)       = η_0·(1 + α_η·max(0, a(t)))                            # η_0 = 5e-4, α_η = 2
```
- Duygu uzayı öğrenmez: sabit kuraldır, yalnızca durumunu ilerletir. Durum yeni akışta sıfırlanır.
- v1'in z / üç baş / E-M-R / perturbasyonla öğrenilen W_r yapısı (§3.4–§3.5) v2'de yoktur. Gerekirse sonraki bir v2 fazında ayrı kararla geri eklenir.
- valence ve dominance yalnızca loglanır; regülatörlere girmez.

### 12.7 Karar bloğu v2 (`MultiHeadDecisionBlock`)
```
r^(k)(t)   = κ_k·r^(k)(t−1) + (1 − κ_k)·s(t),   κ_k = exp(−Δt/τ_read,k),  τ_read ∈ {16, 32, 64, 100, 160} ms     # [K, B, N]
r̃^(k)(t)  = r^(k)(t) ⊙ [g_fast·1_{N/2}, g_slow·1_{N/2}]
v̂(t)      = (1/K) Σ_k (W^(k)·r̃^(k)(t) + b^(k))                                                                     # [B, 2]
```
Çevrimiçi çapalı NLMS (W_0^(k): Day-0 ince ayarı sonunda dondurulan çapa):
```
W^(k) ← (1 − α_anchor)·W^(k) + α_anchor·W_0^(k) + η(t)·e_v(t)·r̃^(k)(t)ᵀ / (‖r̃^(k)(t)‖² + ε),   α_anchor = 1e-4
```
- v1 Faz 6'daki W_base + ΔW elastik çapasıyla (§3.6) aynı ilkedir. Fark: çapa doğrudan W_0'a çeker ve normalizasyon başlık başınadır.
- Çevrimdışı aşamalarda başlıklar omurgayla birlikte gradyanla eğitilir.

### 12.8 Eğitim protokolü
1. **Aşama 1 — Çoklu oturum ön-eğitimi** (primat başına): her oturumun train dilimi 2 s pencerelere bölünür (Δt = 4 ms → 500 adım, stride 250) ve birleştirilir. Batch 32, AdamW (lr 1e-3, wd 1e-4), CosineAnnealing, 100 epoch.
   `L = √(mean_t ‖v(t) − v̂(t)‖² + 1e-6) + λ_homeo·L_homeo`, λ_homeo = 10.
2. **Aşama 2 — Day-0 ince ayarı:** hedef oturumun (günler arası protokolde Day-0 kaynak oturumunun) train diliminde 150 epoch. Val R²'si en yüksek epoch seçilir, ardından `save_anchor_weights()` → W_0.
3. **Aşama 3 — Akış:** test dilimi (oturum içi ya da Day-k) pencerelenmeden adım adım akar (`eval`, gradyan kapalı, `online_homeostasis` config'ten). Geçerli kurallar: KV1, KV2, KV6.
   - **Etiket gecikmesi (KR-041):** y_t okumaya t + d'de ulaşır (`online.label_delay_ms`, varsayılan 500 ms). Güncelleme e = y_{t−d} − v̂_{t−d} ve r_{t−d} ile yapılır. Sebep: 4 ms'de anında etiketle y_{t−1} kopyası bile R² ≈ 0,999 verir, uyum kazancı anlamsızlaşır. Tablolarda gecikmeli kopyalama tabanı (v̂_t = y_{t−d}) raporlanır.
   - **Günler arası model (KR-040):** yalnızca Day-0'da eğitilir (ön-eğitim Day-k train dilimlerini gördüğü için kullanılmaz); Day-k, Day-0 istatistikleriyle kodlanır.

### 12.9 Değerlendirme koşulları (v2)
v1'in A/B/C'siyle karıştırmamak için tablolarda `v2-A` … `v2-E` yazılır.

| Kod | İçerik | Ölçtüğü |
|---|---|---|
| **v2-A** | Donuk model, η = 0 | Ön-eğitim + ince ayarın kendisi |
| **v2-B** | Çapalı NLMS, η = v2-D'nin η̄'sı (eşlenmiş), g = 1 | Temel çevrimiçi uyum (birincil adil kontrol) |
| **v2-C** | v2-D'nin η(t) dizisi zamanda karıştırılmış | Kazanç zamanlamadan mı, varyanstan mı geliyor |
| **v2-D** | Tam sistem: sürpriz modülasyonlu η(t) + g_fast/g_slow + çapalı NLMS + homeostaz | Önerilen yöntem |
| **v2-E** | Akış ortasında rastgele 20 giriş kanalı sıfırlanır; A, B, D üzerinde | Toparlanma süresi |

- Senaryolar: oturum içi test ve Day-0 → Day-k (günler arası).
- Metrikler: R² (NeuroBench), toparlanma süresi (E), alt-popülasyon ateşleme oranı (KV7), a(t)/g/η izleri.
- **Ana iddianın kanıtı:** v2-D'nin v2-B ve v2-C'den istatistiksel olarak anlamlı biçimde iyi olması (5 seed, Wilcoxon). v1'de kapılı η eşlenmiş η'yı geçemedi (KR-031, KR-032); bu yüzden B ve C kontrolleri zorunludur.

### 12.10 Spesifikasyon notları ve açık noktalar
Spesifikasyonun v1 teşhisi ile depodaki gerçek v1 arasındaki farklar (kayıt için; v2 kararlarını değiştirmez):
- v1 kodlayıcısı Min-Max + Poisson değil, delta + termometre kanallarıdır (kanal başına yüzdelik ölçek, §3.1).
- v1'de iç adım 1 ms, ortam adımı 20 ms (T = 20). τ tekil değil, τ·m_i ile ~10–1000 ms'ye yayılır (§3.2).
- v1 beyinciği zaten efference copy (a(t) = ŷ(t)) alır. v1 Faz 6 okuması zaten elastik çapalı NLMS'tir.
- v1'in gerçek veride ateşleme oranı 0,009–0,020'dir (KR-014, KR-031); spesifikasyondaki "%1–2" teşhisi doğru.

İlgili fazda kapatılacak açık noktalar:
1. **Bölme protokolü** (V0): spesifikasyon 65/10/25 (Zenke Lab RSNN ile aynı), v1 ve NeuroBench resmî yükleyicisi 80/10/10. → **Kapandı (KR-035): 80/10/10; literatürle kıyas yaklaşık, ayrıntı KR-035'te.**
2. **Kod yerleşimi** (V0): öneri `src/v2/` altında ayrı modüller; v1 sınıfları ve testleri korunur. → **Kapandı (KR-035): `src/v2/`, `tests/v2/`, `configs/v2_*.yaml`.**
3. **Sıfırlama gecikmesi** (V2): referans `plif` kodunda yumuşak sıfırlama s(t−1) ile uygulanıyor (bir adım gecikmeli). Bilinçli mi, düzeltilecek mi? → **Kapandı (KR-038): snnTorch `Leaky` ile aynı standart biçim, değiştirilmedi.**
4. **Beyincik girdisi döngüsü** (V5): r̄ kapılanmış izlerin ortalaması; g a(t)'ye, a(t) beyincik hatasına bağlı. Beyinciğe kapılanmamış iz mi verilmeli?
5. **Bias güncellemesi** (V4): referans çevrimiçi NLMS yalnızca W'yi günceller, b^(k) donuk kalır. → **Kapandı (KR-040): spesifikasyondaki gibi donuk.**
6. **Hesap bütçesi** (V2'ye ertelendi, KR-036): 100 + 150 epoch × N = 1000 × T = 500 adımlık döngü RTX 4060'ta ölçülmeli; gerekirse epoch sayısı azaltılır ya da Δt = 8 ms kullanılır. → **Kapandı (KR-038): CUDA Graph ile 4 ms'de seed başına tam protokol plif ~0,6–0,8 saat; Δt ve epoch sayıları değişmedi.**
7. **Ön-eğitim kapsamı** (V0/V3): Indy (96 kanal) ve Loco (192 kanal) ayrı primatlardır; ön-eğitim primat başına yapılır. → Günler arası sıra KR-035'te (tarih sırası, ilk oturum Day-0).

### 12.11 Varsayılan hiperparametreler (v2, başlangıç)
| Parametre | Değer |
|---|---|
| Δt | 4 ms (hızlı mod 8 ms) |
| N (hızlı + yavaş) | 1000 (500 + 500) |
| τ_mem hızlı / yavaş | [4, 40] / [40, 400] ms |
| τ_syn | [4, 100] ms |
| `backbone.mode` | `plif` |
| dropout | 0,3 |
| ρ (`target_rate`), [f_min, f_max] | 0,05, [0,02; 0,08] |
| λ_homeo | 10 |
| η_homeo, [ϑ_min, ϑ_max] | 1e-3, [0,3; 2,5] |
| τ_smooth | 48 ms |
| τ_read | {16, 32, 64, 100, 160} ms |
| α_anchor | 1e-4 |
| a_f, a_s | 0,05, 0,001 |
| α_gate, α_η, η_0 | 0,25, 2, 5e-4 |
| Pencere / stride | 500 / 250 adım |
| Ön-eğitim / ince ayar | 100 / 150 epoch, batch 32, AdamW 1e-3, wd 1e-4 |

### 12.12 Modül arayüzleri (v2)
| Modül | `forward` girdisi | `forward` çıktısı | Çevrimiçi `update` |
|---|---|---|---|
| `SpikeEncoderV2` | `x_raw: [B, C]` | `x_norm: [B, C]`, `x̄: [B, C]` | — (`fit` yalnızca train) |
| `TemporalBackbone` | `x_norm: [B, C]` | `s: [B, N]` | — (yalnızca ϑ homeostazı, KV1) |
| `MultiHeadDecisionBlock` | `s: [B, N]`, `g_fast`, `g_slow` | `v̂: [B, 2]`, `r̄: [B, N]` | `e_v`, `η(t)` |
| `CerebellumV2` | `r̄: [B, N]`, `v̂(t−1): [B, 2]` | `x̂_smooth: [B, C]` | `e_cer` |
| `AffectSpaceV2` | `e_cer: [B, C]` | `a`, valence, dominance, `g_fast`, `g_slow`, `η` | — (öğrenmez) |

Bir akış adımı:
1. `x_norm, x̄ = encoder(x_t)`
2. `s = backbone(x_norm)`
3. `v̂, r̄ = decision(s, g)` → skorlanır
4. `e_cer = x̄_t − x̂_smooth(t)` → `cerebellum.update(e_cer)`
5. `a, g, η = affect(e_cer)`
6. `decision.update(e_v, η)`
7. `x̂_smooth(t+1) = cerebellum(r̄, v̂)`
