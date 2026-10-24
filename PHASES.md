# Uygulama Fazları

Her faz bir öncekinin üzerine kurulur. Bir fazın **çıkış kriteri** sağlanmadan sonraki faza geçilmez. Tüm fazlar `ARCHITECTURE.md` kurallarına uyar.

---

## Faz 0 — İskelet ve altyapı

**Hedef:** Kodu yazmadan önce çalışma ortamını hazırlamak.

**Yapılacaklar:**
- `ARCHITECTURE.md`'deki dizin yapısını oluştur.
- `pyproject.toml`: `torch`, `numpy`, `pyyaml`, `matplotlib`, `pytest`, `neurobench`.
- `utils/seeding.py`: tek fonksiyonla torch, numpy ve random seed'lerini sabitle.
- `utils/logging.py`: her koşuda config, seed, git hash ve metrikleri bir klasöre kaydet.
- `configs/base.yaml`: Bölüm 7'deki tüm varsayılan hiperparametreler.
- `experiments/run.py`: config okuyup boş bir döngü çalıştıran iskelet.

**Çıkış kriteri:**
- `python experiments/run.py --config configs/base.yaml` hatasız çalışıyor ve bir log klasörü oluşturuyor.
- `pytest` çalışıyor (henüz test olmasa bile).

---

## Faz 1 — Projeksiyon bloğu

**Hedef:** Sabit, kararlı ve seyrek bir rezervuar.

**Yapılacaklar:**
- `encoder.py`: `SpikeEncoder` (delta + seviye kanalları, Bölüm 3.1).
- `reservoir.py`: `LIFReservoir` (Bölüm 3.2). Başlatma, k-WTA, sızıntı ve `reset_state()`.
- Testler:
  - Aynı seed iki kez aynı `W_in` ve `W_rec`'i üretiyor.
  - `W_in` ve `W_rec` tek bir güncelleme döngüsünden sonra değişmemiş (K1).
  - Ateşleyen nöron sayısı hiçbir iç adımda `k`'yı aşmıyor. Güçlü girdide `k`'ya eşit.
  - Sıfır girdi verildiğinde aktivite en geç bir zaman sabiti (`τ / (T·dt)` ortam adımı, en az 10) içinde sönüyor (kararlılık; kendi kendini sürdüren aktivite yok).
  - İzin verilen her τ değerinde (20, 100, 500, 2000 ms) ortalama ateşleme oranı `ρ/2` ile `2ρ` arasında (K7).
- **Bellek testi:** Rastgele bir girdi dizisi ver. Rezervuar çıktısından ridge regresyonla (λ doğrulama setinde seçilir, R² ayrı tutulan test verisinde raporlanır) iki hedefi geri çıkar:
  - **item:** d adım önceki tek değer `x(t−d)`. Bilgi amaçlı raporlanır. Bağımsız rastgele girdide tek bir öğenin uzun τ'lu bir toplamdaki payı küçük olduğu için bu hedefin τ ile iyileşmesi beklenmez.
  - **window:** son d adımın ortalaması `mean(x(t−d+1), …, x(t))`. **Çıkış kriteri bu hedeften ölçülür.**
  - τ ızgarası: {20, 100, 500, 2000} ms. d ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}.
  - Dizi uzunluğu, en uzun pencerenin test bölümünde birkaç kez bağımsız olarak tekrarlanmasına yetecek kadar olmalıdır; aksi halde uzun pencerelerde R² güvenilmez olur.
  - Okuma iki şekilde raporlanır: yalnızca `f` ile ve `r = [f, v]` ile. Aradaki fark, belleğin zar potansiyelinde ne kadar saklandığını gösterir.
  - Her koşuda bir **ideal referans** da raporlanır: aynı kodlanmış girdi, aynı `W_in` ve aynı `τ_i`'lerle spike'sız doğrusal filtre bankası. Bu, kodlayıcının ve zaman sabitlerinin izin verdiği üst sınırı gösterir.
  - 5 seed, ortalama ± standart sapma.
- **Teşhis tablosu (bellek testi beklenmedik çıkarsa):** Hangi değişikliğin τ–bellek ilişkisini etkilediğini gösteren tablo. Sonuca göre `ARCHITECTURE.md` önce güncellenir, sonra kod.

**Çıkış kriteri:**
- Tüm testler geçiyor.
- **Bellek ufku** = `r = [f, v]` okumasıyla window hedefinde R² ≥ 0.1 olan en uzun pencere. 5 seed ortalamasında:
  - τ = 2000 ms'nin ufku, τ = 20 ms'ninkinin en az 5 katı.
  - Uzun pencerelerde (d ≥ 20) en iyi sonucu veren τ, kısa pencerelerde (d ≤ 2) en iyi sonucu veren τ'dan büyük.

---

## Faz 2 — Beyincik bloğu

**Hedef:** Rezervuar temsili üzerinden bir sonraki girdiyi tahmin etmek.

**Veri:** NeuroBench kaotik fonksiyon tahmini görevi.

**Yapılacaklar:**
- `cerebellum.py`: `Cerebellum` (Bölüm 3.3). Bu fazda `a(t)` yok, sıfır vektörü verilir.
- `configs/chaotic.yaml`.
- Karşılaştırma için çevrimdışı bir okuma: aynı rezervuar çıktıları üzerinde ridge regresyon. Bu, yerel kuralın ne kadar iyi olabileceğinin üst sınırını gösterir.
- Testler:
  - Beyincik tahmin anında `x(t+1)`'e erişmiyor (K6).
  - Güncelleme yalnızca `W_c`'yi değiştiriyor.

**Çıkış kriteri:**
- Tek adımlık tahmin hatası, "bir önceki değeri tekrarla" taban çizgisinden belirgin şekilde düşük.
- Yerel kuralın sonucu, ridge regresyonun makul bir yakınında (örneğin hata farkı %20'nin altında).

---

## Faz 3 — Karar bloğu ve ilk gerçek görev

**Hedef:** Konfigürasyon A ve B'yi gerçek bir görevde çalıştırmak.

**Veri:** NeuroBench primat motor tahmini (parmak ucu hızı, regresyon).

**Yapılacaklar:**
- `decision.py`: `DecisionReadout`, regresyon modu (Bölüm 3.6).
- `agent.py`: Bölüm 6'daki akışın 1–8. adımları. Regülatörler varsayılan değerlerde sabit.
- Beyincik artık `a(t) = ŷ(t)` alıyor.
- `configs/nhp_motor.yaml`.
- NeuroBench değerlendirme aracıyla R² hesabı.

**Çıkış kriteri:**
- Konfigürasyon A ve B için 6 oturumun hepsinde R² raporlanmış (5 seed).
- NeuroBench'in SNN taban çizgisi sonuçlarıyla karşılaştırma tablosu var.
- B'nin A'ya göre farkı not edilmiş (fark olmaması da bir sonuçtur).

---

## Faz 4 — Duygu uzayı (sabit regülatörlerle)

**Hedef:** z uzayını öğrenmek ve anlamlı bir yapı oluşup oluşmadığını görmek. Henüz davranışı etkilemiyor.

**Yapılacaklar:**
- `affect.py`: `AffectSpace` ve üç tahmin başı (Bölüm 3.4).
- `agent.py`: 9. adım. Regülatörler hâlâ sabit.
- Görselleştirme:
  - z(t)'nin 3B çizimi, noktalar anlık performansa (q) göre renklendirilmiş.
  - Her başın tahmini ile gerçek hedefinin zaman grafikleri.

**Çıkış kriteri:**
- Üç başın tahmin hatası, eğitim boyunca düşüyor.
- z uzayında iyi ve kötü performans anları görsel olarak ayrışıyor.
- Bu aşamada sistemin görev performansı Faz 3 ile aynı olmalı (z henüz hiçbir şeyi değiştirmiyor). Değilse bir sızıntı var demektir.

---

## Faz 5 — Regülatörler

**Hedef:** z'nin davranışı ayarlaması: tam sistem, yani konfigürasyon C.

**Yapılacaklar:**
- `affect.py`: `Regulators` (Bölüm 3.5), sınırlar ve log-ölçekli τ.
- `agent.py`: 2. ve 10. adımlar.
- **Ara adım (önerilir):** Perturbasyondan önce elle yazılmış basit bir eşleme dene. Örneğin "belirsizlik yüksekse η'yı artır, τ'yu kısalt". Böylece regülatör mekanizmasının kendisinin çalıştığını doğrularsın.
- Sonra elle yazılmış eşlemeyi perturbasyonla öğrenilen `W_r` ile değiştir.
- Testler:
  - Regülatör çıktıları her zaman sınırlar içinde (K4).
  - `regulators.enabled: false` iken değerler sabit (K5).

**Çıkış kriteri:**
- g, τ, η, β'nın zaman grafikleri anlamlı biçimde değişiyor (sabit kalmıyor, sınırlara yapışmıyor).
- Konfigürasyon C eğitim oturumlarında A ve B'den kötü değil.

---

## Faz 6 — Asıl deney: kayma ve ablation

**Hedef:** Ana araştırma sorusunu cevaplamak.

**Yapılacaklar:**
- `experiments/ablation.py`: A, B ve C'yi aynı seed'lerle sırayla koşturur.
- **Kayma protokolü:** Erken tarihli oturumlarda eğit, sonraki oturumlarda test et. Testte öğrenme açık kalsın (çevrimiçi uyum), çünkü regülatörlerin katkısı tam olarak burada görülecek.
- İstatistiksel test: A–C ve B–C farkları için 5 seed üzerinden Wilcoxon testi.
- Ek analiz: Oturum geçişlerinde regülatörler nasıl davranıyor? Örneğin yeni oturum başlarken η yükseliyor mu?

**Çıkış kriteri:**
- Sonuç tablosu ve grafikler hazır.
- Ana iddia ya destekleniyor ya da desteklenmiyor. İkisi de raporlanabilir bir sonuçtur.

---

## Faz 7 — Genişletme (temel başarılı olursa)

Bu faz, Faz 6'daki sonuçlara göre seçilir. Olası yönler:
- **SSC (Spiking Speech Commands):** Ayrık karar modu ve β regülatörünün testi.
- **Rezervuarın kısmen öğrenilmesi:** Yerel STDP ile K1 kuralının kontrollü olarak gevşetilmesi. Bunun için önce `ARCHITECTURE.md` güncellenir.
- **Crafter:** Hayatta kalma ortamı, pekiştirmeli öğrenme moduna geçiş.
- **Duygu uzayı boyutu:** k = 2, 3, 4, 8 karşılaştırması.

---

## Faz özeti

| Faz | Ana çıktı | Kullanılan veri |
|---|---|---|
| 0 | Çalışan iskelet | — |
| 1 | Kararlı, seyrek, τ ile belleği ayarlanabilen rezervuar | Rastgele diziler |
| 2 | Tahmin yapan beyincik | Kaotik fonksiyon (NeuroBench) |
| 3 | A ve B taban çizgileri | Primat motor tahmini (NeuroBench) |
| 4 | Anlamlı bir z uzayı | Primat motor tahmini |
| 5 | Çalışan regülatörler (C) | Primat motor tahmini |
| 6 | Ana sonuç | Primat motor tahmini, oturumlar arası |
| 7 | Genişletme | SSC, Crafter |
