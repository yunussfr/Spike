# Mimari Kuralları: Projeksiyon–Beyincik–Duygu SNN

Bu dosya kod tabanının **değişmez kurallarını** tanımlar. Her yeni kod, her refaktör ve her deney bu kurallara uymak zorundadır. Bir kural değişecekse önce bu dosya güncellenir, sonra kod.

> **Sürümler (2026-11-03, KR-034).** Depoda iki mimari hattı vardır:
> - **Spike v1** (§1–§10): sabit LIF rezervuarı + yerel okumalar. Kuralları K1–K8. v1 kodu (`LIFReservoir`, A/B/C, Faz 0–6) bu kurallara uymaya devam eder.
> - **Spike v2** (§12): çevrimdışı eğitilen zamansal omurga + çevrimiçi nöromodüle yerel uyum. Kuralları **KV1–KV8**. Kaynak: `docs/spike_v2_mimari_spesifikasyon.md`, fazlar: `docs/PhasesV2.md`.
> - v2 kodu yalnızca §12'ye bağlıdır; v1 kuralları (ör. K1 sabit rezervuar, K7 k-WTA) v2 modüllerini bağlamaz. Aktif geliştirme hattı v2'dir.
> - **2026-11-09 (KR-049):** v1 kodu master'dan kaldırıldı; `v1-final` git tag'inde durur. §1–§10 kayıt olarak kalır.

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
| 2026-11-10 | §12.16 BOCPD: tek boyutlu skor (kanal başına z²'nin ortalamasının logaritması) ve Day-0 rejimine dönüş (p_0); kaynak seçimi x / ar1 / cer_mon. Yeni §12.20: tonik / fazik iş bölümü — tonik arousal BOCPD (→ çapa gevşetme, λ_norm kapısı), fazik arousal plastik beyincik hatasının kopya tahminine oranı (→ η gazı, g). KV4 notu, §12.9 V7 koşulları ve ikincil (beyincik) aile, §12.11 | KR-050: teşhiste bugünkü BOCPD Day-0 ayrılmış dilimde her tavlamada %90–97 yanlış alarm verdi (dönüşsüz Day-0 rejimi + 96–192 kanalın bağımsız çarpımı); CR her kaynakta kalibre oldu. Beyincik sürprizi ‖e‖²/ε²_0 Indy'de ateşleme düşünce yeni günde azalıyordu; kopya oranı bunu düzeltti. Kullanıcı onayı (2026-11-10) |
| 2026-11-09 | §12.1, KV3, KV4, KV6, §12.8, §12.9, §12.11; yeni §12.15–§12.19 (V7): sönümlü ReFIT niyet etiketi, BOCPD sensörü (arousal = P(rejim ≠ Day-0)), ölü kanal dedektörü, Degenhart hizalaması, kanal bozulması augmentation'ı; ayar oturumları test oturumlarından ayrı; istatistik birimi oturum, Holm tüm ailede, toparlanma Kaplan–Meier; v1 kodu `v1-final` tag'ine taşındı | KR-049: hakem raporu (sahte tekrar, ayar = test oturumu, D ≈ C, R5 elektrot kaybında üstün, literatür tabanları eksik) ve KR-046/048'deki 31 s tespit gecikmesi. Kullanıcı onayı (2026-11-09) |
| 2026-11-08 | KV2, KV3, KV4, §12.6, §12.7, §12.9, §12.14: kalıcı sürpriz sensörü (yeniden ölçekleme kapısı); dinamik duygu: arousal (iç hata + başlık uyuşmazlığı, sabit Day-0 tabanı), valens (e_v eğilimi), dominans (e_v düzeyi) → gaz, fren, çapa gevşetme, akış içi çapa konsolidasyonu; yeni regülatörler α_anchor(t), c(t); yeni kontrol v2-P, v2-B ᾱ ile eşlenir, v2-C η, α, c dizilerini karıştırır; v2-E senaryosu tanımlandı | KR-046: kapalı döngü sönüyordu; kalıcı sensör ilk günü korudu. KR-047: Pf ≈ sabit yüksek η (duygu zamanlaması katkısız); valence = −a iken kurallar tek sinyale çöküyordu. Kullanıcı önerisi ve onayı (valens / dominans e_v'den) |
| 2026-11-08 | KV4, KV6, §12.6, §12.9, §12.14: duygu-kapılı etiketsiz kanal yeniden ölçeklemesi ana sisteme girdi; yeni regülatör λ_norm(t); koşul v2-R. W_in kuralı (§12.13) varsayılan kapalı | KR-045: etiketsiz yeniden ölçekleme günler arası val R² ≈ 0 → ~0,19 ama Day-0'da bedelli; W_in kuralı okuma uyumunu geçmedi (karar kapısı 2). Kullanıcı seçimi |
| 2026-11-08 | KV1–KV7, §12.1, §12.6, §12.9, §12.13: W_in çevrimiçi, sürpriz-kapılı (η_in = η_in,0·max(0, a)), çapalı ve norm topu sınırlı üç faktörlü yerel kuralla değişebilir; W_rec ve τ donuk. Yeni koşullar v2-W, v2-L; homeostaz hedefi `day0` seçeneği, v2-L'de akan kanal z-skoru | KR-042: günler arası Day-0 modeli ≈ 0, okuma NLMS'i kurtarmıyor (kayma girişte). Kullanıcı önerisi ve onayı (KR-044) |
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
- **Araştırma sorusu (v2):** Beklenmedik sürprizle modüle edilen öğrenme hızı η(t), sürprizle açılan giriş plastisitesi η_in(t) (KR-044) ve hızlı/yavaş alt-popülasyon kapılaması, günler arası kaymada ve elektrot kaybında, aynı ortalama η'lı sabit kuraldan (v2-B) ve zamanlaması karıştırılmış η'dan (v2-C) daha iyi uyum sağlıyor mu?
- **Araştırma sorusu (V7, KR-049):** Değişim noktası tespitiyle (BOCPD, §12.16) kapılanan girdi uyumu (yeniden ölçekleme, ölü kanal dedektörü, Degenhart hizalaması) ve öğrenme hızı, niyet etiketiyle (sönümlü ReFIT, §12.15), sürekli yeniden ölçeklemeden (R5), eşlenmiş sabit kontrollerden ve zamanlaması karıştırılmış kontrolden daha iyi uyum sağlıyor mu? "Duygu" adları bu bölümde tanımlanan istatistiksel niceliklerin etiketidir: arousal = P(şu anki rejim Day-0 değil); yorum tartışmaya aittir, iddia istatistiksel nicelik üzerinedir.

### 12.2 v2 değişmez kuralları (KV1–KV8)

**KV1 — Omurga çevrimdışı eğitilir; çevrimiçi yalnızca W_in sınırlı biçimde değişebilir (KR-044).**
- Omurga ağırlıkları (`plif`: W_in, W_rec; GRU modları: kapı matrisleri) ve zaman sabitleri τ_mem,i, τ_syn,i yalnızca Aşama 1–2'de (§12.8) gradyanla öğrenilir. Başlatma sabit seed'lidir; ağırlıklar checkpoint'e kaydedilir.
- Aşama 3'te (çevrimiçi akış) omurganın tüm parametreleri `requires_grad=False`'tur. W_rec, τ_mem,i, τ_syn,i, bias'lar ve GRU modlarının tüm kapı matrisleri **değişmez** (v1'deki τ regülatörü v2'de yoktur).
- **İstisna — W_in (yalnızca `plif`):** `plasticity.enabled: true` ise W_in §12.13'teki duygu-kapılı yerel üç faktörlü kuralla değişebilir. Kural Day-0 değerine çapalıdır ve ‖W_in − W_in,0‖_F ≤ δ_max·‖W_in,0‖_F norm topuna izdüşürülür. Akış bitince W_in,0 geri yüklenir. `plasticity.enabled: false` → W_in tamamen donuk (V4 davranışı). Gerekçe: kayma girişte (elektrot / kanal ölçeği); okuma uyumu tek başına telafi edemiyor (KR-042). W_rec donuk: öğrenilen hareket dinamiği günler arası kararlı varsayılır.
- Çevrimiçi değişebilen tek omurga durumu homeostatik eşiktir: `ϑ_i ← clip(ϑ_i + η_homeo·(r̄_i − ρ), ϑ_min, ϑ_max)`. Etiketsiz, nöron başına yerel ve nedenseldir (v1 KR-030 nedensel homeostatik normalizasyonunun karşılığı).

**KV2 — Çevrimiçi öğrenme yereldir.** Aşama 3'te gradyan, BPTT ya da optimizer yoktur. Yalnızca açık formüllü kurallar çalışır: beyincik NLMS'i (§12.5), çapalı okuma NLMS'i ve akış içi çapa konsolidasyonu (§12.7, KR-047), homeostatik eşik (KV1), W_in üç faktörlü kuralı (§12.13: ön-sinaptik iz × nöronun surrogate türevi ψ × okuma ağırlıklarıyla yayınlanan gecikmeli hata; e-prop ailesi yaklaşık kural). Çevrimdışı aşamalarda gradyan serbesttir (K2, KR-033'te kaldırıldı).

**KV3 — Öğrenme sinyalleri karışmaz.**
| Sinyal | Tanım | Kimi eğitir |
|---|---|---|
| **e_v** (sonuç hatası, v1'deki δ) | v(t) − v̂(t) | Çevrimiçi: okuma başlıkları ve (plastisite açıksa) W_in, sabit geri besleme B = okuma ağırlıkları üzerinden (§12.13). Çevrimdışı: omurga + başlıklar (hız kaybı) |
| **e_cer** (tahmin hatası, v1'deki e) | x̄(t+1) − x̂_smooth(t+1) | Yalnızca W_cer |
| **e_int** (niyet hatası, V7, §12.15) | y_int(t) − v̂(t); y_int sönümlü ReFIT etiketi (imleç ve hedef konumundan) | Çevrimiçi okuma başlıkları (`online.label_source: refit`'te e_v'nin yerine) |
- Beyincik kaybı omurgaya gradyan vermez; e_cer W_in'e ulaşmaz. Duygu uzayı e_cer'i yalnızca **okur**, değiştirmez; W_in'e yalnızca η_in(t) üzerinden, adım boyu olarak etki eder.
- Dinamik duygu (KR-047, §12.6): duygu uzayı e_cer'i (iç yol ve izleme yolu), girdi sapmasını (§12.14), okuma başlıklarının çıktılarını (uyuşmazlık) ve gecikmeli e_v'yi **okur**; hiçbirini eğitmez. Okumaya yalnızca η(t), α_anchor(t), c(t) ve g üzerinden, adım boyu / kazanç olarak etki eder.

**KV4 — Regülatörler yalnızca okuma kapılama kazançlarını, η'yı, η_in'i, λ_norm'u, α_anchor'u ve konsolidasyon adımını üretir.**
| Regülatör | Etkilediği yer | Aralık | Varsayılan |
|---|---|---|---|
| g_fast, g_slow | Okuma izlerinin hızlı / yavaş yarısı (§12.7) | [1 − α_gate, 1 + α_gate] ⊂ [0,5; 2] | 1 |
| η(t) | Okuma başlıklarının çevrimiçi NLMS adımı | [η_0, η_0·(1 + α_η)] ⊂ [1e-4; 1e-1]; dinamik duyguda [0, η_0·(1 + α_η)] (fren η'yı 0'a çekebilir) | η_0 |
| η_in(t) | W_in üç faktörlü kuralın adımı (§12.13) | [0, η_in,0] | 0 (sakinken W_in yalnızca çapaya döner) |
| λ_norm(t) | Etiketsiz kanal yeniden ölçeklemesinin EMA adımı (§12.14) | [λ_calm, λ_surp] | λ_calm (sakinken neredeyse donuk) |
| α_anchor(t) | Okuma NLMS'inin çapaya çekme adımı (§12.7, KR-047) | [0, α_0 + α_b] | α_0 |
| c(t) | Akış içi çapanın konsolidasyon adımı: W_anc ← W_anc + λ_c·c·(W − W_anc) (§12.7, KR-047) | [0, 1] | 0 (çapa W_0) |
- τ, ϑ, W_rec, beyincik adımı ve başka hiçbir hiperparametre regülatörden etkilenmez. β (karar sıcaklığı) regresyonda kullanılmaz.
- V7'de (KR-049) regülatörleri süren arousal BOCPD'nin a(t) = 1 − P(rejim = Day-0) olasılığıdır (§12.16); aralıklar değişmez. Tonik / fazik iş bölümünde (§12.20, KR-050) α_anchor(t) ve λ_norm(t) tonik arousal'dan (BOCPD), η(t) ve g fazik arousal'dan (plastik beyincik) gelir; aralıklar yine değişmez, η_cer sabittir (regülatör üretmez). Ölü kanal dedektörü (§12.17) ve Degenhart hizalaması (§12.18) regülatör değil, etiketsiz girdi dönüşümüdür; hizalama yalnız a(t) ≥ a_on iken açılır.

**KV5 — Regülatörsüz mod her zaman çalışır.** `regulators.enabled: false` → g_fast = g_slow = 1, η ve η_in sabit (v2-A'da 0, v2-B'de η̄ ve η_in = 0, v2-W'de η̄ ve η̄_in). Ablation için zorunludur ve testlidir.

**KV6 — Zaman ve sızıntı.**
- Bir veri adımı = bir omurga adımı; Δt config'ten gelir (varsayılan 4 ms). v1'deki T iç adım yoktur.
- Akış değerlendirmesinde durum sıfırlanmaz. Çevrimdışı eğitimde her pencere yeni bir dizidir.
- Beyincik x̂_smooth(t+1)'i üretirken x(t+1)'i ve x̄(t+1)'i göremez.
- Prequential sıra: v̂(t) üretilir ve skorlanır; güncelleme ancak sonra e_v(t) ile yapılır.
- Kodlayıcı istatistikleri (μ_ch, σ_ch) ve tüm ölçekler yalnızca train diliminden hesaplanır. İstisna: §12.14'teki etiketsiz kanal yeniden ölçeklemesi (KR-045) — x_norm'un kanal ortalaması / varyansı akışta nedensel EMA ile izlenir ve girdi Day-0 train ölçeğine döndürülür. Yalnızca geçmiş girdiyi kullanır, etiket kullanmaz; referans (m_0, v_0) Day-0 train'den.
- V7 istisnaları (KR-049), hepsi etiketsiz, nedensel ve referansları yalnızca Day-0 train'den: BOCPD sensörü (§12.16), ölü kanal dedektörü (§12.17), Degenhart hizalaması (§12.18; yeni günün FA'sı yalnızca akışın geçmiş girdisinden). Sensörün kalibrasyonu (tavlama, hazard) Day-0 train'in ayrılmış son diliminde ve ayar oturumlarında yapılır.
- Niyet etiketi (§12.15) imleç ve hedef konumunu kullanır: gerçek BCI'da ekranda bilinen görev durumudur. Hız etiketi (y) kullanmaz; hız büyüklüğü dekoderin kendi çıktısından gelir.
- **Ayar oturumları test oturumlarından ayrıdır (KR-049).** `tune_primates`'teki oturumlar `primates`'te (test) bulunamaz; çakışma hata verir. Day-0 modeli ayar ve testte aynıdır (yalnızca Day-0 train'de eğitilir).

**KV7 — Ateşleme rejimi (k-WTA yerine homeostaz).**
- v2 omurgasında k-WTA yoktur. Hedef ortalama oran ρ = 0,05 (config `target_rate`).
- Çevrimdışı: homeostatik kayıp oranı [f_min, f_max] = [0,02; 0,08] bandına iter. Çevrimiçi: KV1'deki eşik adaptasyonu.
- Çevrimiçi homeostaz hedefi `controls.homeo_target`: `fixed` (ρ) ya da `day0` (nöronun Day-0 akışında ölçülen kendi oranı; KR-044 kontrolü).
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
η_in(t)    = η_in,0·max(0, a(t))                                    # W_in adımı (§12.13); sakinken 0
λ_norm(t)  = λ_calm + (λ_surp − λ_calm)·max(0, a(t))                # yeniden ölçekleme adımı (§12.14, KR-045)
```
- Duygu uzayı öğrenmez: sabit kuraldır, yalnızca durumunu ilerletir. Durum yeni akışta sıfırlanır.
- v1'in z / üç baş / E-M-R / perturbasyonla öğrenilen W_r yapısı (§3.4–§3.5) v2'de yoktur. Gerekirse sonraki bir v2 fazında ayrı kararla geri eklenir.
- Yukarıdaki biçimde valence = −a'dır ve dominance yalnızca loglanır. Dinamik duyguda (aşağıda) üçü ayrı sinyallerdir ve regülatörlere girer.

**Dinamik duygu (KR-047, `AffectDynamics`).** Yukarıdaki biçimde valence = −a olduğu için "hata düşüyor" ile "sürpriz düşük" aynı sinyaldi. Dinamik duyguda arousal bir *düzey* (sabit Day-0 tabanına göre), valens bir *eğilim* (iki uyarlanan EMA'nın farkı), dominans okuma hatasının *düzeyi*dir:
```
s_int = mean e_cer,iç² / ε²_0                       # iç yol: koşulun kendi izleri, hedef yeniden ölçeklenmiş x̄' (§12.14)
u     = Var_k(v̂^(k)) / u_0                          # 5 başlığın uyuşmazlığı; u_0: Day-0 train sonundaki ayrılmış dilim
E_a  ← E_a + λ_a·((1 − w_u)·s_int + w_u·u − E_a);   a = tanh(E_a − 1)            # Day-0 tabanı 1, sabit
q     = mean(e_v²) / q_0                            # gecikmeli okuma hatası; q_0: aynı dilimde donuk modelin hatası
Q_f, Q_m, Q_d ← EMA_{λ_f, λ_m, λ_d}(q)  (λ_m < λ_f);  v = tanh((Q_m − Q_f)/(Q_m + ε));  d = −tanh(ln Q_d)
m     = clamp(‖W − W_anc‖_F / (δ_m·‖W_0‖_F), 0, 1)  # son konsolidasyondan beri öğrenme
b = v⁻·m (fren)   ℓ = a⁺·d⁻ (gevşetme)   c = v⁺·m (konsolidasyon)
η(t)  = η_0·(1 + α_η·a⁺)·(1 − ρ_v·v⁺)·(1 − b)     # gaz · rahatlamada kısma · fren
α(t)  = α_0·(1 − ℓ) + α_b·b                        # yeni günde gevşet, frende geri çek
g_fast = 1 + α_gate·a
```
- Döngü: yeni gün (a yüksek, d düşük) → çapa gevşek, η açık → rahatlama (v > 0) → W_anc öğrenilen ağırlığa taşınır, η kısılır → ΔW hatayı artırırsa (v < 0, m > 0) son iyi ağırlığa (W_anc) geri çekilir. "Hata artıyor → fren" yalnız m > 0 iken çalışır; yeni günün başında (W ≈ W_anc) fren yoktur.
- Valens ve dominans e_v(t) ile güncellenir ve bir sonraki adıma uygulanır (prequential, KV6). Yeniden ölçekleme kapısı bu arousal'dan değil, §12.14'teki sensörden gelir.
- Kontrolü süren arousal seçilebilir (`DynSpec.a_mode`): iç (yukarıdaki a), sensör (§12.14'teki kalıcı sürprizin a_s'si) ya da ikisinin maksimumu. u_0 ve q_0 Day-0 val diliminden (donuk model; train sonu fazla iyimser, KR-047).
- **Val seçimi (KR-047, v2-D):** arousal = sensör a_s (iç a günü ayırmıyor: başlık uyuşmazlığı kaymada düşüyor); gaz α_η = 2; gevşetme açık; g açık; fren, konsolidasyon ve rahatlamada kısma kapalı (ölçülü gazda zararlı ya da nötr; fren yalnız 10·η_0 gazda koruyor). Bu seçimde v2-D: η = η_0(1 + 2·a_s⁺), α = α_0(1 − a_s⁺·d⁻), g_fast = 1 + α_gate·a_s.

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
- **Dinamik duyguda (KR-047):** çapa akış içi W_anc^(k)'dır ve adımı α(t)'dir: W^(k) ← W^(k) − α(t)·(W^(k) − W_anc^(k)) + η(t)·e_v·r̃ᵀ/(‖r̃‖² + ε); ardından W_anc^(k) ← W_anc^(k) + λ_c·c(t)·(W^(k) − W_anc^(k)). W_anc akış başında W_0'dır; akış sonunda modelin W_0'ı değişmemiş olur (akış durumu).

### 12.8 Eğitim protokolü
1. **Aşama 1 — Çoklu oturum ön-eğitimi** (primat başına): her oturumun train dilimi 2 s pencerelere bölünür (Δt = 4 ms → 500 adım, stride 250) ve birleştirilir. Batch 32, AdamW (lr 1e-3, wd 1e-4), CosineAnnealing, 100 epoch.
   `L = √(mean_t ‖v(t) − v̂(t)‖² + 1e-6) + λ_homeo·L_homeo`, λ_homeo = 10.
2. **Aşama 2 — Day-0 ince ayarı:** hedef oturumun (günler arası protokolde Day-0 kaynak oturumunun) train diliminde 150 epoch. Val R²'si en yüksek epoch seçilir, ardından `save_anchor_weights()` → W_0.
3. **Aşama 3 — Akış:** test dilimi (oturum içi ya da Day-k) pencerelenmeden adım adım akar (`eval`, gradyan kapalı, `online_homeostasis` config'ten). Geçerli kurallar: KV1, KV2, KV6.
   - **Etiket gecikmesi (KR-041):** y_t okumaya t + d'de ulaşır (`online.label_delay_ms`, varsayılan 500 ms). Güncelleme e = y_{t−d} − v̂_{t−d} ve r_{t−d} ile yapılır. Sebep: 4 ms'de anında etiketle y_{t−1} kopyası bile R² ≈ 0,999 verir, uyum kazancı anlamsızlaşır. Tablolarda gecikmeli kopyalama tabanı (v̂_t = y_{t−d}) raporlanır.
   - **Günler arası model (KR-040):** yalnızca Day-0'da eğitilir (ön-eğitim Day-k train dilimlerini gördüğü için kullanılmaz); Day-k, Day-0 istatistikleriyle kodlanır.
   - **Etiket kaynağı (V7, KR-049):** `online.label_source: delayed` (KR-041, 500 ms gecikmeli hız) ya da `refit` (§12.15, gecikmesiz sönümlü niyet etiketi). V7'nin varsayılanı `refit`; `delayed` ablasyon olarak raporlanır.
4. **Kanal bozulması augmentation'ı (V7, KR-049, §12.19):** Aşama 1 ve 2'deki her eğitim penceresine, x_norm alanında rastgele kanal düşürme (kanal "sıfır sayım" değerine çekilir) ve kanal başına kazanç / kayma bozulması uygulanır. Yalnızca eğitimde; val seçimi ve akış bozulmasız veriyle yapılır.

### 12.9 Değerlendirme koşulları (v2)
v1'in A/B/C'siyle karıştırmamak için tablolarda `v2-A` … `v2-E` yazılır.

| Kod | İçerik | Ölçtüğü |
|---|---|---|
| **v2-A** | Donuk model, η = 0 | Ön-eğitim + ince ayarın kendisi |
| **v2-B** | Çapalı NLMS, η = v2-D'nin η̄'sı ve α = ᾱ'sı (eşlenmiş), g = 1 | Temel çevrimiçi uyum (birincil adil kontrol) |
| **v2-P** | v2-D ile aynı sensörlü yeniden ölçekleme (§12.14) + eşlenmiş sabit η̄, ᾱ; duygu okumaya girmez (KR-047) | Duygunun okuma zamanlamasının katkısı (kilit kontrol) |
| **v2-C** | v2-D'nin η(t), α(t) ve c(t) dizileri zamanda karıştırılmış; yeniden ölçekleme D ile aynı sensörden (KR-047). Önceki biçim: η(t), η_in(t), λ_norm(t) | Kazanç zamanlamadan mı, varyanstan mı geliyor |
| **v2-W** | v2-B + W_in kuralı, sabit η_in (v2-D'nin η̄_in'i, eşlenmiş), duygu yok | W_in plastisitesinin kendisi; duygu kapısının kontrolü |
| **v2-D** | Tam sistem: sensörlü yeniden ölçekleme (§12.14, KR-046) + dinamik duygu (§12.6, KR-047): η(t), α(t), konsolidasyon c(t), g_fast/g_slow + çapalı NLMS (+ ops. W_in kuralı, homeostaz) | Önerilen yöntem |
| **v2-R** | v2-B + sabit hızlı yeniden ölçekleme (λ̄_norm = v2-D'nin ortalaması, eşlenmiş; ayrıca sabit τ = 5 / 30 s) | Duygu kapısının yeniden ölçeklemeye katkısı |
| **v2-L** | Etiketsiz kontrol: Day-0 hedefli homeostaz ve/veya akan kanal z-skoru, η = 0 | Kaymanın ne kadarı yalnızca ölçek? |
| **v2-E** | Akış ortasında rastgele 20 giriş kanalı sıfırlanır; A, B, D üzerinde | Toparlanma süresi |

- Senaryolar: oturum içi test, Day-0 → Day-k (günler arası) ve elektrot kaybı (v2-E: oturum içi model, değerlendirme diliminin ortasında seed'e bağlı 20 kanalın ham sayımı 0).
- Metrikler: R² (NeuroBench), kayıp sonrası R² ve toparlanma süresi (E: kayıptan sonraki ilk 5 s'lik pencerenin R²'si ≥ kayıp öncesi pencere medyanı − 0,05), alt-popülasyon ateşleme oranı (KV7), a(t)/v(t)/d(t)/g/η/α izleri.
- **Ana iddianın kanıtı:** v2-D'nin v2-B, v2-P ve v2-C'den istatistiksel olarak anlamlı biçimde iyi olması (5 seed × oturum, kesin Wilcoxon, Holm). v1'de kapılı η eşlenmiş η'yı geçemedi (KR-031, KR-032); bu yüzden B ve C kontrolleri zorunludur. v2-W V6 dışında: W_in kuralı kapı 2'de okuma uyumunu geçmedi (KR-045).

**V7 koşulları (KR-049).** Hepsi aynı (augmentation'lı) checkpoint'le, `label_source: refit`:

| Kod | İçerik | Ölçtüğü |
|---|---|---|
| **A** | Donuk model, η = 0 | Ön-eğitim + augmentation |
| **B7** | Çapalı NLMS, ReFIT etiketi, eşlenmiş sabit η̄, ᾱ | Yalnız okuma uyumu |
| **R5** | B7 + sürekli yeniden ölçekleme (τ 5 s) | Basit z-skor tabanı (hakem) |
| **G7** | B7 + Degenhart hizalaması akış başından açık, BOCPD yok | Literatür hizalama tabanı |
| **P7** | BOCPD-kapılı yeniden ölçekleme + ölü kanal dedektörü + Degenhart; okumada eşlenmiş sabit η̄, ᾱ | Okuma zamanlamasının katkısı |
| **C7** | D7'nin η(t), α(t) dizileri zamanda karıştırılmış; girdi tarafı P7 ile aynı | Zamanlama mı, varyans mı |
| **D7** | P7 + BOCPD arousal'ı → η gazı, çapa gevşetme, g (§12.6 v2-D kuralları, a = a_BOCPD) | Önerilen sistem |
| D7−dead, D7−align, D7−delay | D7'den ölü kanal dedektörü / hizalama çıkarılmış; ReFIT yerine 500 ms gecikmeli etiket | Bileşen ablasyonları |
| D7@v6 | D7, V6 checkpoint'iyle (augmentation'sız) | Augmentation'ın katkısı |
| D7_split | D7 + tonik / fazik iş bölümü (§12.20): BOCPD → α, λ_norm; plastik beyincik (kopya oranı) → η, g | İş bölümünün katkısı |
| D7_ph0, D7_N0 | D7_split'ten fazik kol çıkarılmış (a_ph = 0) / fazik kol eski normalizasyonla (‖e‖²/ε²_0) | Plastik beyinciğin ve normalizasyonun katkısı |
| D7@kaynak, D7_split@kaynak | BOCPD öteki kaynakla (ar1 ↔ cer_mon, §12.16) | Beyincik artığının sensöre katkısı |

- **İstatistik (V7, KR-049; hakem raporu):** birim **oturumdur**. Seed'ler önce oturum içinde ortalanır; oturum düzeyinde kesin Wilcoxon (n < 6'da ulaşılabilir en küçük p raporlanır) ve oturum → seed sıralı küme bootstrap'lı %95 güven aralığı. Seed × oturum birimli Wilcoxon yalnızca ek bilgi olarak verilir. Holm iki düzeyde: önceden kayıtlı birincil aile (D7 − P7, D7 − R5, D7 − C7, D7 − B7) ve tüm karşılaştırma ailesi. İkincil (beyincik) aile ayrı Holm'la ve keşifsel olarak raporlanır (KR-050): D7_split − D7_ph0, D7_split − D7_N0, D7@cer_mon − D7@ar1, D7_split − D7. Toparlanma süresi sansürlü veridir: Kaplan–Meier medyanı ve sansürlü birim sayısı raporlanır, ortalama ± std raporlanmaz.

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
| α_gate, α_η, η_0 | 0,25, 2, 5e-4 (V5/V6: η_0 = 1,5e-3, KR-046) |
| Kalıcı sürpriz sensörü (§12.14) | τ_d 10 s, v_f 1e-2, δ_0 = Day-0 train maks.; τ_fast 5 s; kapı [0,3; 0,6]; τ_calm ∞, τ_surp 5 s; τ_ret 30 s |
| Dinamik duygu (§12.6, KR-047) | a_mode = sensör; τ_a 5 s, w_u 0,5; valens τ 2 s / 20 s, dominans τ 20 s; ρ_v 0; τ_brake 2 s, τ_cons 5 s, δ_m 0,2 (fren ve konsolidasyon v2-D'de kapalı) |
| Elektrot kaybı (v2-E) | 20 kanal, değerlendirme diliminin ortası; toparlanma penceresi 5 s, adım 0,5 s, tol 0,05 |
| Sönümlü ReFIT (§12.15, başlangıç) | r_d 20 mm, r_on 5 mm, tepki süresi 200 ms |
| BOCPD (§12.16, başlangıç) | blok 200 ms (graph_chunk'ın böleni), hazard 600 s, R_max 512 blok, tek boyutlu skor ε_y 1e-3, Day-0'a dönüş p_0 0,5, kaynak ar1 (karşılaştırma cer_mon), tavlama ızgarası {0,002 … 1} (Day-0 ayrılmış dilimde yanlış alarm ≤ %1), yeni rejim önseli κ 1 / α 1, kalibrasyon payı Day-0 train'in son %20'si, kapı [0,5; 0,9] |
| Fazik kol (§12.20, başlangıç) | kopya oranı (N1), τ_a 5 s, w_u 0, η_cer ayar ızgarası {1e-4, 3e-4, 1e-3} (başlangıç 3e-4) |
| Ölü kanal (§12.17) | α_dead 1e-6 |
| Degenhart (§12.18, başlangıç) | q 10 faktör, FA kutusu 20 ms, T_cal 30 s, yeniden uydurma 30 s, pencere ≤ 300 s, kararlı oran 0,6, 5 iterasyon, a_on 0,5 |
| Augmentation (§12.19, başlangıç) | p_drop 0,5, k_max = ⌈0,2·C⌉, p_gain 0,5, σ_g 0,2, σ_b 0,2 |
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

### 12.13 Duygu-kapılı W_in plastisitesi (KR-044, yalnızca `plif`)
Her akış adımında, okuma güncellemesiyle aynı gecikmeli hatayla (§12.8, d adım):
```
ψ_i(t)   = 1 / (1 + k·|V_i(t) − ϑ_i|)²                     # eğitimdeki surrogate türevi (k = surrogate_scale)   [N]
ψ̄(t)    = κ_ψ·ψ̄(t−1) + (1 − κ_ψ)·ψ(t)                     # post-sinaptik iz, κ_ψ = exp(−Δt/τ_ψ)                [N]
x̃(t)    = κ_x·x̃(t−1) + (1 − κ_x)·x_norm(t)                # ön-sinaptik iz, κ_x = exp(−Δt/τ_x)                  [C]
L(t)     = W̄ᵀ·e_v(t),   W̄ = (1/K)·Σ_k W^(k)               # okuma ağırlıklarıyla yayınlanan hata (simetrik e-prop) [N]
W_in    ← W_in − α_in·(W_in − W_in,0)
          + η_in(t)·(L(t) ⊙ ψ̄(t−d))·x̃(t−d)ᵀ / (‖x̃(t−d)‖² + ε)
W_in    ← W_in,0 + Π_δ(W_in − W_in,0)                       # ‖·‖_F ≤ δ_max·‖W_in,0‖_F izdüşümü
```
- e_v(t) = y(t−d) − v̂(t−d); ψ̄ ve x̃ halka tamponda saklanır (rank-1 güncelleme). Burn-in ve dolgu adımlarında güncelleme yok.
- Sakinken (a ≤ 0) η_in = 0 ve W_in yalnızca W_in,0'a geri çekilir: öğrenilen ağırlıklara sadakat. Sürpriz yükselince W_in açılır.
- Beyincik W_cer Day-0 train diliminde ridge ile başlatılır (Aşama 2b) ve E_slow Day-0 düzeyinden başlar; çevrimiçi NLMS'i W_in'den hızlı uyum sağlamalıdır (zaman ölçeği ayrımı, geri besleme döngüsüne karşı).
- Parametreler `plasticity.*` (η_in,0, α_in, δ_max, τ_ψ, τ_x) YAML'dan; yalnızca ayar seed'leri + val ile seçilir.
- **Durum (KR-045):** karar kapısı 2'de okuma uyumunu (v2-B) geçmedi; η_in arttıkça zarar veriyor. Kod ve testler duruyor, varsayılan kapalı (`plasticity.eta_in_grid: []`).

### 12.14 Duygu-kapılı etiketsiz kanal yeniden ölçeklemesi (KR-045)
Koşul başına, her adımda (skorlanan adımlarda güncellenir; burn-in'de durum Day-0 değerinde kalır):
```
m(t) ← m(t−1) + λ_norm(t)·(x(t) − m(t−1))                          # kanal ortalaması                 [C]
v(t) ← v(t−1) + λ_norm(t)·((x(t) − m(t−1))² − v(t−1))              # kanal varyansı                   [C]
x'(t) = m_0 + (x(t) − m(t))·√(v_0 / (v(t) + ε))                    # Day-0 train ölçeğine dönüş; omurga x' görür
λ_norm(t) = λ_calm + (λ_surp − λ_calm)·max(0, a(t−1)),   λ = 1 − exp(−Δt/τ)
```
- m_0, v_0: Day-0 train diliminde x_norm'un kanal ortalaması / varyansı; m, v bu değerlerden başlar.
- Beyincik hedefi x̄ yeniden ölçeklenmiş x'ten hesaplanır: ölçek geri döndükçe e_cer ve sürpriz söner, yeniden ölçekleme yavaşlar (kapalı döngü).
- Sakinken (a ≤ 0) τ = τ_calm (uzun): Day-0'da sabit hızlı yeniden ölçeklemenin bedeli (val'de −0,01…−0,05) kalkmalı. Sürprizde τ → τ_surp (kısa).
- Etiket kullanmaz; KV6'nın tek istisnasıdır. Parametreler `renorm.*` (τ_calm, τ_surp, ε) YAML'dan, yalnızca ayar seed'leri + val ile.
- **Durum (KR-046):** yukarıdaki kapalı döngü söner (ölçek düzelince sürpriz ve hız düşer); eşlenmiş sabiti geçmedi. Yerine kalıcı sürpriz sensörü:

**Kalıcı sürpriz sensörü (KR-046, `persist` config; v2-D ve v2-P'de):**
```
s_mon(t) = mean (x̄_ham(t) − W_cer,0·φ_A(t−1))² / ε²_0    # izleme yolu: A'nın izleri, ham (yeniden ölçeklenmemiş) hedef, donuk W_cer
m_d ← m_d + λ_d·(x − m_d);  v_d ← v_d + λ_d·((x − m_d)² − v_d)            # kanal EMA'sı, ham x_norm, τ_d 10 s
δ(t) = mean_c ½·[ρ − 1 − ln ρ + (m_d − m_0)²/v_0'],  ρ = max(v_d, v_f)/v_0',  v_0' = max(v_0, v_f)   # Day-0 train'e Gauss KL
ε²(t) = max(s_mon, δ/δ_0)                                   # δ_0: Day-0 train boyunca en büyük δ (tolerans zarfı)
E_s  ← E_s + λ_s·(ε² − E_s);   a_s = tanh(E_s − 1)          # E_slow Day-0 tabanında (1) sabit, τ_fast 5 s
h = clamp((a_s − lo)/(hi − lo), 0, 1);   λ_norm = λ_calm + (λ_surp − λ_calm)·h        # lo 0,3, hi 0,6; τ_calm = ∞, τ_surp = 5 s
m, v ← m, v + λ_ret·(1 − h)·((m_0, v_0) − (m, v))          # sakinken Day-0 referansına dönüş, τ_ret 30 s
```
- Sensör koşulun kendi yeniden ölçeklemesinden bağımsızdır (geri besleme döngüsü yok) ve tüm dinamik duygu koşulları için ortaktır.
- Ölü kanalda (elektrot kaybı) v → 0 olur ve x' kanalın Day-0 ortalamasına iner (sonlu; test edildi).

### 12.15 Sönümlü ReFIT niyet etiketi (V7, KR-049)
Gecikmeli hız etiketinin (KR-041) iki sorunu var: gecikmeli LMS'in kararlı adım sınırı gecikmeyle küçülür (η ≈ 3·η_0'da ıraksama), ve gerçek bir BCI'da el hızı etiketi yoktur. ReFIT (Gilja vd. 2012) ve retrospektif hedef çıkarımı (Jarosiewicz vd. 2015) bunun yerine kullanıcının niyetini etiket sayar: imleç hedefe gitmek istiyordu. Her akış adımında:
```
p(t), g(t)   = imleç ve hedef konumu (mm; .mat'taki cursor_pos, target_pos, aynı 4 ms ızgarası)   [2]
Δ(t)         = g(t) − p(t);   ρ(t) = ‖Δ(t)‖
û(t)         = Δ(t) / (ρ(t) + ε)
σ(t)         = 1 − exp(−ρ(t) / r_d)                            # sönüm: hedefe yaklaştıkça hız → 0
s(t)         = ‖v̂_fiz(t)‖                                       # dekoderin kendi hız büyüklüğü (fiziksel birim)
y_int,fiz(t) = 0                     ρ(t) ≤ r_on (hedefte)
             = σ(t)·s(t)·û(t)        aksi halde
y_int(t)     = (y_int,fiz(t) − μ_y) / σ_y                       # standart ölçek (train istatistikleri)
m_int(t)     = 0  hedef değişiminden sonraki ilk t_rt boyunca (tepki süresi), aksi halde 1
e_int(t)     = m_int(t)·(y_int(t) − v̂(t))
```
- Prequential: v̂(t) önce skorlanır, sonra e_int(t) ile güncellenir; gecikme d = 0, halka tampon yok.
- Sızıntı yok: y_int hız etiketini (y) kullanmaz. Yönü görev durumundan (p, g), büyüklüğü dekoderden gelir. Tanı olarak etiket kalitesi raporlanır: R²(σ·‖y‖·û, y), yani gerçek hız büyüklüğüyle oluşturulmuş niyet etiketinin hıza benzerliği.
- ReFIT hız büyüklüğünü öğretmez; yalnızca yönü düzeltir. Sönüm, hedefe yaklaşırken yavaşlamayı öğretir.
- Parametreler `refit.*` (r_d, r_on, t_rt, ε) YAML'dan, yalnız ayar oturumlarında.

### 12.16 BOCPD değişim noktası sensörü (V7, KR-049; tek boyutlu skor ve Day-0'a dönüş KR-050)
Kalıcı sürpriz sensörünün (§12.14) EMA'ları kanıtı biriktirmek yerine unutuyor; eşik δ_0 keyfî bir tolerans zarfı (Indy 0131'de 31 s gecikme). Yerine Bayesçi çevrimiçi değişim noktası tespiti (Adams & MacKay 2007), bir kaynak sinyalinin blok ortalamalarından türeyen **tek boyutlu skor** üzerinde:
```
v(t)    kaynak [C]:  x       → x_norm(t)
                     ar1     → x_norm(t)                  (dönüşüm blok düzeyinde, aşağıda)
                     cer_mon → x̄(t) − W_b·φ_mon(t−1)      (donuk Day-0 beyincik artığı, izleme yolu: koşul 0 = donuk A)
o_k     = (1/B)·Σ_{t∈blok k} v(t)                                    # B = block_ms/Δt adım   [C]
z_k,c   = (o_k,c − m_c) / s_c                                         # x, cer_mon
z_k,c   = ((o_k,c − m_c) − ρ_c·(o_{k−1},c − m_c)) / s_c               # ar1: kanal başına AR(1) yeniliği
y_k     = ln( Σ_c canlı_c·z²_k,c / Σ_c canlı_c + ε_y )                # tek boyutlu skor
Day-0 rejimi D: y_k ~ N(m_y, s²_y)
Yeni rejim r:   y_k ~ Normal-Gamma(μ_r, κ_r, α_r, β_r) → Student-t;  önsel μ = m_y, κ = κ_new, α = α_new, β = α_new·s²_y
log ℓ   = τ_T · log p(y_k)                                            # τ_T: tavlama
Z       = π_D·ℓ_D + Σ_r π_r·ℓ_r
π'_D    = π_D·ℓ_D·(1 − H) + H·p_0·Z                                   # Day-0 sürüyor ya da değişimden sonra Day-0'a dönülüyor
π'_{r+1}= π_r·ℓ_r·(1 − H)                                             # yeni rejim sürüyor (r = 0 … R_max − 2; sonuncu birikir)
π'_0    = H·(1 − p_0)·Z                                               # yeni rejim başlıyor;  H = B·Δt / hazard
π       ← π' / Σπ'
a(t)    = 1 − π_D                                                     # arousal: P(şu anki segment Day-0 rejiminde değil) ∈ [0, 1]
n(t)    = Σ_{r < r_new} π_r                                           # son r_new blokta değişim olasılığı (loglanır)
```
- **Neden tek boyut ve dönüş (KR-050):** İlk biçimde (kanal başına bağımsız Gauss, dönüşsüz Day-0) Day-0 train'in ayrılmış diliminde yanlış alarm her τ_T'de %90–97'ydi. İki sebep: (i) 96–192 kanalın olabilirlikleri bağımsız sayılıp çarpıldığı için kanallardaki küçük ortak sapmalar sınırsız kanıt biriktiriyordu; (ii) değişim noktası her zaman *yeni* rejim açtığı için π_D yalnızca azalabiliyordu: tek bir geçici davranış patlaması a'yı kalıcı olarak yükseltiyordu ve hazard tek başına 600 s'de a'yı ~0,63'e taşıyordu. Kanal başına önce beyazlatma (AR(1), beyincik artığı) ya da unutmalı simetrik Day-0 rejimi bunu çözmedi; tek boyutlu skor + dönüş (p_0) her kaynakta kalibre oldu (teşhis: `scripts/diagnose_cer_bocpd.py`, `scripts/diagnose_bocpd_models.py`).
- a(t) artık "şu anki segment Day-0 değil" olasılığıdır; geçici sapmadan sonra geri inebilir. Day-k'de Day-0 rejimi skoru açıklayamadıkça a yüksek kalır. p_0 = 0 ilk (dönüşsüz) biçimdir.
- Ölü sayılan kanallar (§12.17) skordan çıkarılır (canlı_c = 0).
- **Referans (yalnız Day-0 train, KV6):** x ve ar1'de m, s (ve ρ) Day-0 train'in ilk %80'inin blok ortalamalarından. cer_mon'da W_b, Day-0 train başındaki [başlangıç + burn-in, + fit_steps) dilimde ridge'le uydurulur (çevrimiçi beyinciğin W_0'ından ayrı: onun train sonu ridge'i kalibrasyon dilimiyle örtüşür), m, s artığın bu dilimden sonra ilk %80'e kadarki kısmından. m_y, s²_y aynı dilimin skorlarından. τ_T, train'in son %20'sinde yanlış alarm (a > 0,5 blok oranı) ≤ `bocpd.calib_false_alarm` olacak en büyük ızgara değeri.
- **Kaynak seçimi:** ar1 (beyinciksiz beyazlatma) ve cer_mon (beyincik artığı) ayar oturumlarında yan yana koşar (D7, D7_split ve kaynak geçişi). Önceden kayıtlı kural: D7'nin ayar oturumlarındaki Day-k R² ortalaması yüksek olan; fark < 0,005 ise Day-0 val yanlış alarmı düşük olan. Teşhiste (tek seed) cer_mon tespiti ~2× hızlandırdı, Day-0 val yanlış alarmı %2,4 / %5,4 (ar1: %0 / %0,5).
- a(t) blok sınırında güncellenir ve sonraki bloğun adımlarında kullanılır (nedensel). Kapı h = clamp((a − lo)/(hi − lo), 0, 1); λ_norm = λ_calm + (λ_surp − λ_calm)·h; Day-0'a dönüş (τ_ret) §12.14'teki gibi.
- Parametreler `bocpd.*` YAML'dan.

### 12.17 Ölü kanal dedektörü (V7, KR-049)
Day-0 train'de kanalın 4 ms'lik kutuda spike olasılığı p_c. Kanalın son spike'ından beri geçen kutu sayısı z_c(t):
```
ölü_c(t) = [ z_c(t)·(−ln(1 − p_c)) ≥ −ln α_dead ]                 # Poisson: z_c kutu sessizliğin olasılığı < α_dead
x'_c(t)  = m_0,c        ölü_c(t)                                  # nötr: Day-0 train ortalaması
         = x'_c(t)      aksi halde (yeniden ölçeklenmiş / hizalanmış)
```
- Spike gelince kanal hemen geri açılır. Hiç ateşlemeyen kanal (p_c = 0) değerlendirilmez.
- "Sıfır sayım" x_norm = normalize(0)_c ile tanınır (kodlayıcı deterministik); ham sayım gerekmez.
- Ölü kanalın yeniden ölçekleme istatistikleri (m, v) güncellenmez; Degenhart hizalamasında kararsız sayılır ve yeniden kurulur.

### 12.18 Degenhart hizalaması (V7, KR-049)
Degenhart vd. (2020) kararlı gizil uzay hizalaması, kanal uzayına döndürülmüş biçimde (omurga x_norm görür):
```
Day-0:  FA(x_kutu; q) → Λ_0 [C, q], μ_0 [C], Ψ_0 [C]             # Day-0 train, fa_bin_ms'lik kutu ortalamaları
Yeni gün (a(t) ≥ a_on olduktan sonra, akışın geçmiş girdisinden):
        FA(son ≤ T_win s; q) → Λ_k, μ_k, Ψ_k                      # T_cal s birikince ilk uydurma, her T_refit s'de yeniden
        S ← tüm canlı kanallar; n_iter kez:
            O = argmin_{OᵀO=I} ‖Λ_0[S] − Λ_k[S]·O‖_F               # Procrustes (SVD)
            d_c = ‖Λ_0[c] − Λ_k[c]·O‖;   S ← en küçük d'li ⌈f_s·C_canlı⌉ kanal
        M_k = Λ_kᵀ (Λ_kΛ_kᵀ + Ψ_k)⁻¹                              # E[z | x] (gizil, yeni gün koordinatı)  [q, C]
        x̂_0(t) = μ_0 + Λ_0·Oᵀ·M_k·(x_norm(t) − μ_k)               # Day-0 kanal uzayında yeniden kurulum    [C]
x'_c(t) = yeniden ölçeklenmiş x_c(t)    c ∈ S (kararlı)
        = x̂_0,c(t)                      c ∉ S ya da ölü
```
- Hizalama hazır olana kadar (T_cal dolmadan) ve a(t) < a_on iken tüm kanallar yeniden ölçekleme yolundan geçer.
- FA, EM ile (`align.em_iters`) uydurulur; yalnızca geçmiş girdi, etiket yok.
- Risk: yeniden kurulan kanallar yalnız ortak bileşeni taşır (özel varyans yok); omurga eğitimde böyle kanal görmedi. Augmentation (§12.19) bunu kısmen karşılar. NoMAD (Karpowicz vd. 2025) V7 çıkış kriteri karşılanmazsa sonraki adımdır; şimdi yazılmaz.

### 12.19 Kanal bozulması augmentation'ı (V7, KR-049)
Sussillo vd. (2016) esinli; Aşama 1–2'de her pencere için (x_norm alanında, GPU'da, CUDA Graph'ın dışında):
```
olasılık p_drop:  k ~ U{1, …, k_max} rastgele kanal → x_c(t) = z0_c (sıfır sayımın x_norm'u), tüm pencere boyunca
olasılık p_gain:  x_c(t) ← clip·tanh((x_c(t)·(1 + σ_g·ξ_c) + σ_b·ζ_c) / clip),   ξ, ζ ~ N(0, 1) kanal başına
```
- Rastgelelik eğitim seed'inden; val seçimi ve akış bozulmasız veriyle. Parametreler `augment.*` YAML'dan; `augment.enabled: false` V6 davranışıdır.

### 12.20 Tonik / fazik iş bölümü (V7, KR-050)
İki "değişim" sinyali farklı soruları cevaplar ve farklı düğmeleri yönetir:
```
Tonik   a_ton(t) = a_BOCPD(t)                                  # §12.16: "Day-0 rejiminde değiliz" (kalıcı)
Fazik   e(t)  = x̄'(t) − x̂(t)                                   # plastik beyincik (§12.5, η_cer > 0); x̄': koşulun yeniden ölçeklenmiş girdisi
        N(t)  = EMA_{λ_a} mean_c e_c(t)²
        K(t)  = EMA_{λ_a} mean_c (x̄'_c(t) − x̄'_c(t−1))²        # "son değeri tekrarla" tahmininin hatası
        s_ph  = (N / K) / r_0                                   # r_0: Day-0 ayrılmış dilimde donuk beyincikle ort. e² / ort. Δx̄²
        ū     = EMA_{λ_a} u                                     # başlık uyuşmazlığı / u_0 (§12.6)
        a_ph  = tanh((1 − w_u)·s_ph + w_u·ū − 1)                # "şu an hâlâ beklenmedik bir şey oluyor" (beyincik alıştıkça söner)
Yönlendirme (a_mode = split):
        η(t), g_fast ← a_ph:   η = η_0·(1 + α_η·a_ph⁺)·(1 − ρ_v·v⁺)·(1 − b),   g_fast = 1 + α_gate·a_ph
        α(t), λ_norm ← a_ton:  α = α_0·(1 − a_ton⁺·d⁻) + α_b·b,   h = clamp((a_ton − lo)/(hi − lo), 0, 1)
```
- **Gerekçe:** Çapayı gevşetmek ve girdiyi yeniden ölçeklemek pahalı ve kalıcı kararlardır; kalibre ve kalıcı kanıta (BOCPD) bağlanır. Öğrenme hızını kısa süre artırmak ucuz ve geri alınabilir; hızlı tepki veren ve uyum sağladıkça sönen beyincik sürprizine bağlanır.
- **Normalizasyon (KR-050):** §12.6'daki s_int = mean e²/ε²_0 mutlak büyüklüktür. Indy'nin sonraki günlerinde ateşleme düşer (ortalama sayım 0,084 → 0,037), Day-0 ölçeğiyle kodlanan girdi ve onunla beyincik hatası küçülür; sürpriz yeni günde *azalır* (KR-047: iç arousal Day-k'de a⁺ 0,00, Day-0'da 0,18 — bu yüzden V6'da elenmişti). Kopya tahminine oran ölçekten bağımsızdır: beyincik "son değeri tekrarla"dan ne kadar iyi tahmin ediyor. Teşhiste (`scripts/diagnose_phasic.py`, ayar oturumları) oran, donuk beyincikle Indy Day-k başında a⁺ 0,24–0,59 verdi (eski: 0); plastik beyincikle (η_cer 1e-3) Day-0 a⁺ ≈ 0 ve Loco'da Day-k başı 0,34–0,39'dan sonu 0,06–0,13'e söndü. Girdinin yerel enerjisine ya da kanal varyansına bölmek Day-0'da yanlış sürpriz verdi.
- Beyincik yalnız e_cer ile güncellenir (KV3); η_cer sabittir, regülatör üretmez (KV4). Aralıklar §12.2'deki gibi. `regulators.enabled: false` her zaman çalışır (KV5).
- a_mode = sensor (V6/V7 D7) ve internal / max biçimleri korunur; split yalnız yeni koşullarda (D7_split, D7_ph0, D7_N0). r_0, Day-0 tabanlarıyla (u_0, q_0) aynı dilimde ölçülür.
- Parametreler `v7.phasic.*` ve `v7.cer_family` YAML'dan.
