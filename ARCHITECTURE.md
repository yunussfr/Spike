# Mimari Kuralları: Projeksiyon–Beyincik–Duygu SNN

Bu dosya kod tabanının **değişmez kurallarını** tanımlar. Her yeni kod, her refaktör ve her deney bu kurallara uymak zorundadır. Bir kural değişecekse önce bu dosya güncellenir, sonra kod.

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

### K2 — Global geri yayılım yok
- Hiçbir modül `loss.backward()` kullanmaz. Tüm güncellemeler `torch.no_grad()` içinde, açık formüllerle yapılır.
- Her güncelleme **yereldir**: bir ağırlığın değişimi yalnızca o ağırlığın girdisine (s(t)), çıktısına ve bir hata sinyaline bağlıdır.

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
v(t) = V(T)                          # ortam adımı sonundaki zar potansiyelleri     [B, N]
r(t) = [f(t), v(t)]                  # okuma vektörü                                [B, 2N]
```
Sebep: Uzun τ'da bellek spike'lardan çok zar potansiyelinde saklanır. Okuma yalnızca spike'lara bakarsa bu bilgi görünmez. Bu dosyada `r(t)` yazan her yer bu birleşik `[B, 2N]` vektörü ifade eder.

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
- Hâlâ yereldir (K2): bir ağırlığın değişimi yalnızca kendi girdisine ve e'ye bağlıdır; `μ`, `σ²` gelecekten bilgi taşımaz (K6). Yalnızca `W_c` değişir.
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

### 3.5 Regülatörler
```
ham     = σ(W_r · z(t) + b_r)                # şekil: [4]
g, η, β = alt + ham · (üst − alt)
τ       = exp(log τ_alt + ham_τ · (log τ_üst − log τ_alt))
```
**W_r eğitimi (perturbasyon yöntemi):**
1. Her `M` adımda bir `W_r`'ye küçük bir rastgele gürültü eklenir.
2. Sonraki `M` adımın ortalama `q`'su öncekiyle karşılaştırılır.
3. İyileşme varsa değişiklik tutulur, yoksa geri alınır.

### 3.6 Karar bloğu
- **Regresyon görevleri** (örneğin primat motor tahmini): `ŷ(t) = W_a · r(t)`. β kullanılmaz.
- **Ayrık görevler** (örneğin SSC, Crafter): `p(a) = softmax(β · W_a · r(t))`.
```
δ(t)  = y(t) − ŷ(t)
ΔW_a  = η · δ(t) · r(t)ᵀ
```

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
2. `g, τ, η, β = regulators(z_{t−1})` (ilk adımda varsayılanlar kullanılır)
3. `r_t = reservoir(u, g, τ)`
4. `ŷ_t = decision(r_t, β)` → çıktı
5. `x̂_{t+1} = cerebellum(r_t, ŷ_t)`
6. Hedef `y_t` ve sonraki girdi `x_{t+1}` gelir.
7. `δ_t = y_t − ŷ_t` → `decision.update(δ_t, η)`
8. `e_{t+1} = x_{t+1} − x̂_{t+1}` → `cerebellum.update(e_{t+1}, η)`
9. `z_t = affect(r_t)` → `affect.update(q, u, s)`
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

**Ana iddianın kanıtı:** C'nin, kayma testinde A ve B'den istatistiksel olarak anlamlı biçimde iyi olması (5 seed, t-testi ya da Wilcoxon testi).

## 9. Kodlama kuralları

- Python 3.11+, PyTorch. snnTorch yalnızca yardımcı fonksiyonlar için kullanılabilir. Rezervuar kendi kodumuzla yazılır, çünkü K1 ve k-WTA üzerinde tam kontrol gerekir.
- Tüm fonksiyonlarda tip ipuçları (type hints) ve tensör şekli yorumları zorunludur: `# [B, N]`.
- Yeni bir modül ancak testi yazıldıktan sonra `agent.py`'ye bağlanır.
- `tests/test_invariants.py` her commit öncesi geçmelidir.

## 10. Yasaklar

- Projeksiyon bloğu ağırlıklarını güncellemek (K1).
- `loss.backward()` ya da herhangi bir otomatik türev kullanımı (K2).
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
| 2026-10-24 | `(1−α)` çarpanı kaldırıldı | τ büyüdükçe girdi gücü küçülüyor, nöronlar ateşlemiyordu (τ=100'de ateşleme oranı %0.08) |
| 2026-10-24 | k-WTA "tam k" yerine "eşiği geçenlerden en fazla k" oldu | "Tam k" kuralı ile "girdi yokken aktivite sönsün" testi birbiriyle çelişiyordu |
| 2026-10-24 | Okuma vektörü `[f, v]` (`[B, 2N]`) oldu | Bellek zar potansiyelinde duruyor, yalnızca spike'lara bakan okuma onu göremiyordu |
| 2026-10-24 | τ aralığı 5–100 ms'den 20–2000 ms'ye, varsayılan 100 ms oldu | Bir ortam adımı 20 ms sürüyor, eski aralıkta bellek en fazla ~1–2 adım olabilirdi |
| 2026-10-24 | θ sabitten θ(τ) tablosuna geçti (geçici) | `(1−α)` kalkınca ateşleme oranının τ ile değişmesini engellemek için; Faz 1 teşhisiyle doğrulanacak |
| 2026-10-24 | Nöron başına zaman sabitleri `τ_i = τ · m_i` ve göreli denge katsayısı `c_i` eklendi | Bellek testi: tüm nöronlar aynı τ'yu kullanınca bellek her τ'da ~2 adımda bitiyordu; aynı filtreyi paylaşan nöronlardan gecikmeler ayrıştırılamıyor |
| 2026-10-24 | Sıfırlama `V ← 0`'dan `V ← V − θ`'ya (yumuşak) geçti | ~%4 ateşleme oranında sıfıra çekme, aktif nöronların belleğini her ~1.4 ortam adımında siliyordu |
| 2026-10-24 | Rekürrent kazanç `γ_rec` eklendi | Seyrek ikili spike'lar ve θ ≈ 7 ile `W_rec` katkısı eşiğe göre çok küçüktü, nöronlar arası karışım yoktu |
| 2026-10-28 | Beyincik varsayılan kuralı normalize edilmiş çevrimiçi LMS oldu (§3.3) | Faz 2: ham delta kuralı `‖φ‖² ~ 1e6` ile ıraksadı (5 seed'de NaN); normalize kural MSE'yi "önceki değeri tekrarla"nın %17.9'una indirdi, ridge'e göre fark %6.8 |
| 2026-10-24 | Kodlayıcı `DeltaEncoder` → `SpikeEncoder`: delta kanallarına seviye (termometre) kanalları eklendi | Delta kodlama x'in seviyesini iletmiyordu; ideal bir hafızayla bile pencere ortalaması geri çıkarılamıyordu (Bölüm 3.1) |
