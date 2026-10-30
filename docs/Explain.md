# Rezervuar Durum Güncellemesi (Sızıntılı Entegratör Denklemi):

Kodumda yer alan V (zar potansiyeli veya durum vektörü), W_in, W_rec ve tau_mult parametreleri her $t$ zaman adımında aşağıdaki kesikli zamanlı (discrete-time) dinamik denklemle çalışır:$$v(t) = (1 - \alpha) \odot v(t-1) + \alpha \odot f\Big( W_{in} u(t) + W_{rec} v(t-1) \Big)$$Buradaki sembollerin senin kodundaki karşılıkları ve matematiksel rolleri şunlardır:$u(t) \in \mathbb{R}^{n_{in}}$: $t$ anındaki girdi vektörüdür.$v(t) \in \mathbb{R}^{N}$: $N$ adet (örneğin 1000) nöronun $t$ anındaki durumunu (zar potansiyelini) tutan vektördür.$\alpha = \frac{\Delta t}{\tau_{mult}} \in (0, 1]^N$ (Sızıntı Oranı / Leaking Rate): Her nöronun kendi zaman sabiti çarpanından (tau_mult) türetilen vektördür ($\odot$ eleman bazlı Hadamard çarpımıdır). Bir nöronun $\tau$ değeri ne kadar büyükse $\alpha$ değeri o kadar küçüktür; bu da $(1 - \alpha)$ çarpanını 1'e yaklaştırarak nöronun geçmiş durumunu ($v(t-1)$) çok yavaş unutmasını sağlar.$f(\cdot)$ (Aktivasyon veya Ateşleme Fonksiyonu): Sürekli ağlarda (ESN) genellikle $\tanh(\cdot)$ kullanılırken, zar potansiyelli (Spiking / LIF) ağlarda potansiyel belirli bir eşiği ($V_{th}$) geçtiğinde 1 üreten basamak fonksiyonu (Heaviside step function) kullanılır. Doğrusal olmayan (non-linear) zenginliği yaratan kısım burasıdır; eğer $f$ olmasaydı tüm geçmiş adımlar basit bir matris toplamına indirgenirdi.




# Yüksek Boyutlu Uzayda Doğrusal Ayrılabilirlik (Cover Teoremi):
Sabit $W_{in}$ ve $W_{rec}$ matrislerinin neden eğitilmeden işe yaradığının matematiksel kanıtı Cover'ın Ayrılabilirlik Teoremi (Cover's Theorem, 1965) ile açıklanır. Bu teorem der ki: Düşük boyutlu uzayda doğrusal olarak ayrılamayan karmaşık bir sınıflandırma veya regresyon problemi, doğrusal olmayan dönüşümlerle yeterince yüksek boyutlu bir uzaya (örneğin 3 girdiden 1000 nörona) rastgele fırlatıldığında yüksek olasılıkla doğrusal olarak ayrılabilir hale gelir.W_in girdileri farklı kombinasyonlarla karıştırır, W_rec geçmiş adımları şimdiki adımla çarpıştırır, tau_mult ise farklı zaman dilimlerini üst üste bindirir. Sonuçta elindeki $v(t)$ vektörü, geçmiş girdilerin binlerce farklı doğrusal olmayan polinom açılımını (Volterra serisi gibi) hazır olarak barındıran 1000 boyutlu bir koordinat noktası olur.

# Öğrenme Nasıl Gerçekleşir? (Türevsiz, Tek Hamlede Çözüm):

Rezervuar sabit olduğu için eğitim sırasında zaman içinde geri yayılım (BPTT) yapılmaz. Girdi serisi ağa baştan sona bir kez verilir ve her $t$ adımında oluşan 1000 nöronluk durum vektörleri yan yana dizilerek büyük bir Durum Matrisi ($V \in \mathbb{R}^{T \times N}$) toplanır. Hedeflenen gerçek çıktılar da $Y \in \mathbb{R}^{T \times n_{out}}$ matrisidir.Amacımız sadece $V \cdot W_{out} \approx Y$ denklemini sağlayan optimal çıktı ağırlıklarını ($W_{out}$) bulmaktır. Bu tamamen dışbükey (convex) bir Ridge Regresyon (Tikhonov Regülasyonu) problemidir ve yerel minimum (local minima) tuzağı yoktur. Çözümü gradyan inişiyle binlerce epoch bekleyerek değil, tek bir kapalı form (closed-form) matris denklemiyle saniyeler içinde bulunur:$$W_{out} = \left( V^T V + \lambda I \right)^{-1} V^T Y$$Burada $\lambda I$, ağırlıkların aşırı büyümesini (overfitting) engelleyen L2 regülasyon terimidir. (Eğer model uçtan uca bir PyTorch mimarisinin parçasıysa, bu son katman basit bir nn.Linear(1000, n_out) olarak tanımlanıp standart Adam veya SGD ile de eğitilebilir; bu durumda bile gradyanlar sadece tek bir katmanda hesaplandığı için eğitim son derece hızlıdır.)

# Neden 1 Ortam Adımında 20 İç Adım Dönüyor? (Zaman Çözünürlüğü ve Yankılanma):

Ortamdan (örneğin bir simülasyondan veya sensörden) gelen her 1 veri adımı ($t$), rezervuarın içinde 20 mikro adıma ($\Delta t$) bölünür. Eğer 1 adıma karşılık 1 hesaplama yapılsaydı, girdi sinyali sadece ilk bağlandığı nöronları uyarabilir, ağın derinliklerine yayılacak vakit bulamazdı. 20 iç adım döndüğünde, SpikeEncoder tarafından üretilen ilk darbe (spike), W_rec matrisi üzerinden nöronlar arasında 20 kez sekerek ilerler. Tıpkı suya atılan bir taşın dalgalarının havuzun duvarlarına çarpıp yayılması gibi, veri bu 20 adımda rezervuarın tamamına yayılır ve geçmiş adımların dalgalarıyla girişim (interference) yapar.

# Dört İç Adımın Matematiksel ve Biyolojik Anlamı_integrate:

 Döngüsünün içindeki 4 satır, biyolojik bir nöronun diferansiyel denkleminin kesikli zamandaki (discrete-time) birebir karşılığıdır:Akım Hesabı ($I_k = W_{in} \cdot s_{in}(k) + W_{rec} \cdot s_{rec}(k-1)$): Nörona o mikro adımda giren toplam elektrik akımıdır. Dışarıdan gelen anlık spike'lar ile bir önceki mikro adımda ($k-1$) rezervuardaki diğer nöronların ürettiği spike'lar ağırlıklarla çarpılıp toplanır.Zar Potansiyeli Güncellemesi ($V_k = \alpha \odot V_{k-1} + c \odot I_k$): Sızıntılı Entegrasyon (Leaky Integration) adımıdır. Nöron bir kapasitör (kondansatör) gibidir; eski voltajının bir kısmını hücre zarından dışarı sızdırarak unutur ($\alpha \cdot V$), yeni gelen akımı ise üzerine ekler ($c \cdot I$). Buradaki $\alpha$ değeri önceki soruda bahsettiğin tau_mult (hafıza uzunluğu) vektöründen gelir.k-WTA (k-Winners-Take-All — En Güçlü %5 Ateşler): Bu adım mimarinin en kritik kararlılık (stabilite) mekanizmasıdır. Biyolojik beyindeki Lateral İnhibisyon (yanal baskılama / GABAerjik ara nöronlar) mekanizmasını taklit eder. Eşiği geçen herkes ateşleseydi, W_rec içindeki döngüler yüzünden birkaç adımda 1000 nöronun tamamı aynı anda ateşlemeye başlar ve ağ "epileptik nöbet" geçirerek (patlayarak) bilgiyi yok ederdi. kwta sayesinde her iç adımda voltajı en yüksek olan en fazla 50 nöron (%5) ateşleyebilir; geri kalanlar susturulur. Bu, bilginin seyrek kodlanmasını (sparse coding) garantiler.Yumuşak Sıfırlama / Soft Reset ($V_k = V_k - s_k \cdot V_{th}$): Ateşleyen ($s_k = 1$) nöronların voltajını doğrudan $0$'a çekmek (Hard Reset) yerine, içlerinden sadece eşik değeri ($V_{th}$) kadar voltaj düşülür. Bunun sebebi bilgi kaybını önlemektir: Bir nöron eşiği $0.01$ farkla geçtiyse başka, $5.0$ farkla geçtiyse başka güçtedir. Çıkarma işlemi yapıldığında o fazladan biriken voltaj silinmez, bir sonraki adıma devreder.



# Düzeltme Notları (yukarıdaki bölümler için)

**1. Rezervuar denklemi.** En üstteki denklem, sürekli yankı ağlarının (ESN) biçimidir: $v(t) = (1-\alpha)\,v(t-1) + \alpha\, f(\dots)$. Bizim LIF rezervuarımız biraz farklı çalışır (ARCHITECTURE.md §3.2):
$$V_i(k) = \alpha_i\, V_i(k-1) + c_i\, I_i(k), \qquad \alpha_i = e^{-dt/\tau_i}, \qquad \tau_i = \tau \cdot m_i$$
- $\alpha$, `dt / tau_mult` değil; **üstel** sızıntıdır ve nöronun asıl zaman sabiti $\tau_i = \tau \cdot m_i$'dir (`tau_mult` = $m_i$, yalnızca bir çarpan).
- Girdi akımının önünde $(1-\alpha)$ çarpanı **yoktur** (KR-001). Onun yerine $c_i$ denge katsayısı vardır: sabit girdide tüm nöronlar aynı seviyeye çıkar, sadece hızları farklıdır.
- $f$ (eşik) potansiyelin *içinde* değil, potansiyelden *sonra* uygulanır: önce $V$ güncellenir, sonra $V > \theta(\tau)$ olanlardan en fazla $k$ tanesi ateşler, ateşleyenlerden $\theta$ çıkarılır.

**2. Öğrenme nasıl gerçekleşir?** Ridge regresyon bu projede sistemin öğrenme kuralı **değildir**. K2 kuralı gereği sistem tek hamlede çözüm (ridge) ya da gradyan (Adam/SGD) kullanmaz; okuma blokları her adımda yerel bir kuralla, adım adım öğrenir (aşağıda "Normalize LMS"). Ridge'i yalnızca bir **üst sınır referansı** olarak hesaplarız: "aynı rezervuar çıktılarıyla en iyi doğrusal okuma ne kadar iyi olabilirdi?" Yerel kuralın sonucu buna ne kadar yakınsa, kural o kadar iyi çalışıyor demektir.

---

# Veri: Primat Motor Tahmini (Faz 3)

Bir maymunun (Indy ya da Loco) motor korteksine yerleştirilmiş elektrotlardan gelen spike'lar girdi, maymunun parmak ucunun hızı (x ve y bileşeni) hedeftir. Görev, beyin aktivitesinden hareketi okumaktır (beyin–makine arayüzü).

- **6 oturum:** 3 Indy (96 kanal), 3 Loco (192 kanal). Her oturum ayrı bir kayıttır ve ayrı ayrı eğitilir/değerlendirilir. Faz 6'daki "kayma" deneyi bu oturumlar arasındaki farktan yararlanacak.
- **20 ms kutular (KR-012):** Ham spike'lar 4 ms çözünürlükte gelir. Biz her 20 ms'yi bir ortam adımı sayarız ve o 20 ms içindeki spike'ları kanal başına sayarız. Sayım 0–5 arasıdır (her 4 ms dilimde en fazla bir spike). Neden 20 ms? Çünkü rezervuarın bir ortam adımı `T · dt = 20 · 1 ms` sürer; τ'nun anlamı (K4) bu zaman ölçeğine göre seçilmiştir.
- **Bölme:** Kayıt zaman sırasıyla bölünür: ilk %80 train (öğrenme), sonraki %10 val (ayar seçimi), son %10 test (yalnızca rapor). Hiçbir ayar test'e bakılarak seçilmez (CLAUDE.md kural 11).

# Ön İşleme: Kanal Başına Ölçek (KR-013)

Bazı elektrotlar çok aktif, bazıları neredeyse sessizdir. Kodlayıcının seviye kanalları (termometre kodu) sabit eşiklere bakar: $x > b_j$. Tüm kanallar için tek bir aralık kullanılırsa sessiz kanalların hiçbir seviye biti yanmaz, aktif kanallar ise hep doyar; iki durumda da bilgi kaybolur.

Çözüm: her kanal, **train** bölmesindeki sayımlarının 99. yüzdeliğine bölünür. Böylece her kanalın "yüksek" değeri ≈ 1 olur ve kodlayıcı aralığı $[0, 1]$ alınır. Ölçek yalnızca train'den hesaplanır; val/test'e bakılsaydı gelecekten bilgi sızardı.

# Normalize LMS: Okuma Bloklarının Öğrenme Kuralı (KR-010, KR-011)

Okuma bloklarının en basit yerel kuralı delta kuralıdır: $\Delta W = \eta \cdot \text{hata} \cdot r^\top$. Sorun şu: $r = [f, v]$ vektörünün enerjisi $\|r\|^2 \approx 10^6$ mertebesinde ($v$ zar potansiyelleri büyük). Delta kuralının bir adımda hatayı ne kadar değiştirdiği $\eta \|r\|^2$ ile orantılıdır; $\eta = 0{,}01$ iken bu $10^4$ eder ve her adım hatayı küçültmek yerine büyütür → ağırlıklar patlar (Faz 2'de 5 seed'in hepsinde NaN).

Normalize kural iki şey yapar:
1. **Standartlaştırma:** her öznitelik, kendi geçmişinin üstel ortalaması $\mu$ ve varyansı $\sigma^2$ ile ölçeklenir: $\tilde\varphi = (\varphi - \mu)/\sqrt{\sigma^2 + \varepsilon}$, sonuna sabit bir bias girişi $b = \sqrt{F}$ eklenir. $\mu$ ve $\sigma^2$ yalnızca o ana kadar görülen veriden hesaplanır (nedensel).
2. **Adımı normalize etme:** $\Delta W = \eta \cdot \text{hata} \cdot \tilde\varphi^\top / (\varepsilon + \|\tilde\varphi\|^2)$. Böylece adım büyüklüğü girdinin ölçeğinden bağımsız olur; $\eta$ artık "hatanın ne kadarını bu adımda düzelt" anlamına gelir.

Kural hâlâ **yereldir** (K2): bir ağırlığın değişimi yalnızca kendi girdisine ve çıktının hatasına bağlıdır, geri yayılım yoktur.

# Karar Bloğu (Faz 3)

Görev çıktısını üreten bloktur: $\hat y(t) = W_a \cdot \tilde\varphi(r(t))$, burada $\hat y$ iki boyutlu hız tahminidir. Hedef $y(t)$ gelince sonuç hatası $\delta = y - \hat y$ hesaplanır ve yalnızca $W_a$ güncellenir. K3 gereği karar bloğu beyinciğin tahmin hatası $e$'yi hiç görmez.

# Bir Adımın Akışı: Ajan (`src/agent.py`)

Her 20 ms'de şu olur:
1. `u = encoder(x_t)` — sayımlar spike kanallarına çevrilir.
2. Regülatör değerleri alınır (Faz 3–4'te hep varsayılan: g = 1, τ = 100 ms, η = 0,01).
3. `r_t = reservoir(u, g, τ)` — 20 iç adım, sonunda $r = [f, v]$.
4. `ŷ_t = decision(r_t)` — hız tahmini.
5. `x̂_{t+1} = cerebellum(r_t, ŷ_t)` — bir sonraki 20 ms'nin sayımlarını tahmin et. Beyincik kararın çıktısının bir kopyasını ($a = \hat y$) da görür: "ne yapmaya karar verdim" bilgisi, "sonra ne göreceğim" tahminine yardım edebilir.
6. Gerçek $y_t$ ve bir sonraki sayımlar $x_{t+1}$ gelir.
7. $\delta = y_t - \hat y_t$ → yalnızca karar bloğu öğrenir.
8. $e = x_{t+1} - \hat x_{t+1}$ → yalnızca beyincik öğrenir.

Kod bunu üç çağrıya böler: `perceive` (1–3), `act` (4–5), `learn` (6–8). Regülatörler kapalıyken $r$ öğrenmeye bağlı olmadığı için deneyler $r$'yi bir kez üretip A ve B'ye aynı anda verir (KR-016).

# A, B ve C Konfigürasyonları

| | İçerik | Soru |
|---|---|---|
| **A** | Rezervuar + karar bloğu | Taban çizgisi |
| **B** | A + beyincik | Beyincik eklemek karar performansını değiştiriyor mu? |
| **C** | B + duygu uzayı + regülatörler | Asıl sistem (Faz 5–6) |

Önemli bir ayrıntı: B'deki beyincik, karar bloğunun ağırlıklarına ya da girdisine dokunmaz (K3). Regülatörler kapalıyken (Faz 3–4) bu yüzden A ve B **birebir aynı** $\hat y$'yi üretir; testler bunu doğrular. Beyincik ancak Faz 5'te, duygu uzayı ve regülatörler üzerinden (belirsizlik $u = \|e\|^2$) davranışı etkileyebilir. Yani Faz 3'te "B − A farkı yok" beklenen ve doğru bir sonuçtur.

# R² Nasıl Okunur?

$R^2 = 1 - \dfrac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}$. Tahmin her zaman ortalama hızı söyleseydi $R^2 = 0$, kusursuz olsaydı $R^2 = 1$ olurdu; negatif değer ortalamadan bile kötü demektir. NeuroBench x ve y bileşenlerinin $R^2$'sini ayrı hesaplayıp ortalamasını alır. Bizim üç referansımız:
- **ortalama tabanı:** train ortalamasını tahmin etmek (≈ 0),
- **ridge üst sınırı:** aynı $r$ üzerinde en iyi doğrusal okuma,
- **NeuroBench SNN taban çizgisi:** makalede Indy 0,593, Loco 0,568 (arXiv 2304.04640, Tablo 4).
