# Spike — kayma tespitiyle kapılanan çevrimiçi uyum yapan spike'lı nöral dekoder

Maymun motor korteksinden kaydedilen spike'lardan el hızını çözen, **çevrimdışı eğitilmiş bir spike'lı ağ** ve onun üstünde **yalnızca yerel kurallarla** çevrimiçi uyum sağlayan bloklar. Veri: NeuroBench Primate Reaching (Indy, Loco).

Ana iddia istatistiksel niceliklere dayanır: **kayma tespitine bağlı (sensörlü) girdi yeniden ölçeklemesi ve öğrenme hızı kapılaması.** Kodda ve belgelerde geçen "duygu" (arousal, valens, dominans) ve "beyincik" adları, ARCHITECTURE.md §12'de tanımlanmış niceliklerin etiketidir. Örneğin V7'de arousal = P(şu anki rejim Day-0 değil) (BOCPD, §12.16). Biyolojik yorum tartışmaya aittir, iddia bu niceliklere dayanır.

![Mimari ve ilerleyiş](docs/figures/architecture_v2.svg)

## Hedefler

| Hedef | Ölçüt | Durum |
|---|---|---|
| Oturum içi çözümleme | NeuroBench test R² ≥ 0,65–0,70 | ✅ Indy 0,750 / Loco 0,710 (NeuroBench SNN 0,593 / 0,568) |
| Günler arası kaymaya uyum | Day-0'da eğitilen model sonraki günlerde çalışsın; etiketler 500 ms gecikmeli (V7: sönümlü ReFIT niyet etiketi) | ◐ Donuk model −0,07 → 0,28 |
| Elektrot kaybına dayanıklılık | 20 kanal ölünce hızlı toparlanma | ◐ Sürekli ölçekleme 11 s; duygulu sistem 17,5 s |
| Araştırma sorusu | Kayma tespitiyle kapılanan uyum; aynı ortalama hızdaki sabit kuraldan (B), sensörlü sabit kuraldan (P), zamanlaması karıştırılmış kuraldan (C) ve sürekli z-skorundan (R5) iyi mi? | Oturum birimli (Day-k 4 oturum, ulaşılabilir en küçük p 0,125): D − B büyük ve tutarlı (4/4, +0,24); D − P küçük ve tutarlı (4/4, +0,007); D − C ve D − R5 belirsiz. Anlamlılık iddiası yok ([results/v6](results/v6/table.md)) |
| Biyolojik kısıt | Çevrimiçi gradyan yok; omurga donuk; yalnız yerel kurallar (NLMS, homeostaz) | ✅ (ARCHITECTURE.md §12, KV1–KV8) |

## Neler denendi

### Kötü sonuç verenler

| Deneme | Sonuç | Neden |
|---|---|---|
| v1: sabit LIF rezervuarı + yerel okuma | Oturum içi 0,44 / 0,37 | Sabit rezervuarın temsili doğrusal okumaya yetmiyor; aynı ızgarada GRU 0,74 / 0,69 (KR-033) |
| v1: duygu / regülatörlerle τ, g, η ayarı | Kontrolü geçmedi; Loco'da −0,08 | Uzun τ'ya kayma yıkıcı; kapılı η eşlenmiş η'yı geçmedi (KR-028–032) |
| Günler arası, yalnız okuma uyumu (B) | Day-k 0,04 | Kayma girişte (kanal ölçeği / kimliği); okuma bozuk temsili düzeltemiyor |
| Duygu-kapılı W_in plastisitesi | Okuma uyumunu geçmedi; büyük adımda çöküyor (−0,93) | 500 ms gecikmeli hata girişi yeniden eşlemek için fazla gürültülü (KR-045) |
| Kapalı döngü duygu-kapılı ölçekleme | Day-k 0,11 < eşlenmiş sabit 0,12 | Ölçek düzelince sürpriz sönüyor, uyum duruyor (KR-046) |
| İç arousal (beyincik + başlık uyuşmazlığı) | Day-k 0,19 (D 0,28) | Başlık uyuşmazlığı hareketi izliyor ve kaymada düşüyor; arousal ilk günde daha yüksek (KR-047) |
| Valens freni (ölçülü öğrenme hızında) | −0,05 | Valens uzanma hareketlerinin gürültüsünü izliyor; fren yeni günü öğrenmeyi geri çekiyor |
| Sabit yüksek öğrenme hızı (10·η_0) | Day-k −0,35; 30× ıraksıyor | Gecikmeli NLMS 3×'in üstünde kararsız |

### Umut verici olanlar

| Deneme | Sonuç | Neden çalıştı |
|---|---|---|
| Eğitilebilir spike omurgası (plif, hızlı / yavaş alt-popülasyon) | Oturum içi 0,75 / 0,70 (v1'in ~2 katı) | Surrogate gradyanla çevrimdışı BPTT; çoklu oturum ön-eğitimi + Day-0 ince ayarı (KR-039, KR-042) |
| Etiketsiz kanal yeniden ölçekleme | Day-k 0,02 → 0,25 | Kaymanın büyük kısmı kanal ölçeği; girdiyi Day-0 ölçeğine geri çekmek temsili kurtarıyor (KR-045) |
| Kalıcı sürpriz sensörü | İlk gün korunuyor (0,775; sürekli ölçeklemede 0,738) | Ham hedefli donuk beyincik + girdi sapması, sabit Day-0 tabanı: yalnız gerçekten yeni günde açılıyor (KR-046) |
| Sensör = "ne zaman uyum sağla" anahtarı (v2-D) | Day-k **0,279** (ortalama); B, P, Pg, Pm3'ten tutarlı biçimde iyi; R5'e karşı belirsiz (3/4 oturum) | Sensör açılınca okuma gazı (≤ 3·η_0) ve çapa gevşetme birlikte açılıyor (KR-047, KR-048) |
| Valens freni, aşırı gazda | 10·η_0'da 0,12 → 0,20 | Fren kararsız öğrenmeyi durduruyor: güvenlik ağı |
| Sürekli ölçekleme, elektrot kaybında | Kayıp sonrası 0,652, toparlanma 11 s | Ölü kanal varyansını kaybedince kendi ortalamasına iner ve nötrleşir |

### Literatürle yaklaşık karşılaştırma

| Karşılaştırma | Literatür | Bu çalışma | Farklı yapılan |
|---|---|---|---|
| NeuroBench Primate Reaching, SNN (oturum içi) | NeuroBench tabanları Indy 0,593 / Loco 0,568 (arXiv 2304.04640); BioCAS 2024 yarışması en iyi R² ≈ 0,70 (AEGRU, RSNN-L) | 0,750 / 0,710 (test, 5 seed) | 4 ms çözünürlükte eğitilen omurga; 500 hızlı + 500 yavaş nöron (τ 4–400 ms); 5 zaman ölçekli okuma (16–160 ms); çoklu oturum ön-eğitimi; CUDA Graph ile 4 ms'de tam BPTT (KR-038) |
| 4 ms GRU tavanı (kendi ölçümümüz) | Val 0,714 / 0,571 | plif 0,769 / 0,658 (val, tek oturum) | Spike'a geçmek kaybettirmedi; alt-popülasyon zaman ölçekleri GRU'dan iyi |
| Sabit kurallı çevrimiçi uyum | Okuma NLMS: Day-k 0,04 | Duygulu uyum: 0,28 | Uyumun girişte yapılması (yeniden ölçekleme) ve "ne zaman" kararının sürprizden gelmesi |

Karşılaştırma yaklaşıktır, "geçtik" diye okunmamalı:
- NeuroBench sonuçları 75/25 bölmeyle raporlanmıştır; biz NeuroBench kodunun 80/10/10 bölmesini kullanıyoruz (KR-035).
- Modelimiz 1000 nöronlu ve çoklu oturumla ön-eğitilmiş; NeuroBench SNN tabanları ve BioCAS girdileri verimlilik odaklı küçük modellerdir.
- NeuroBench işlem sayısı ve bellek ayak izini de raporlar; bu depo henüz ölçmüyor.
- Günler arası kaymada standart tabanlar (Wiener / Kalman yeniden kalibrasyonu, ReFIT, Degenhart 2020, NoMAD, ADAN) V6'da yoktu; V7'de Degenhart (G7), ReFIT etiketi ve R5 tabanları eklendi (KR-049).

## V7: hakem raporuna cevap (KR-049, kod hazır, koşular bekliyor)

| Sorun (V6) | V7'deki değişiklik |
|---|---|
| Sahte tekrar: birim (oturum, seed) | Birim oturum; küme bootstrap %95 GA; Holm birincil ve tüm ailede; toparlanma Kaplan–Meier ([results/v6](results/v6/table.md)) |
| Ayar oturumu = test oturumu | Test dışı 6 ayar oturumu (`tune_primates`, Zenodo 583331); çakışma hata verir |
| Tespit gecikmesi (Indy 0131 ~30 s) | BOCPD değişim noktası sensörü: arousal = P(rejim ≠ Day-0) (§12.16) |
| Elektrot kaybında sensörlü kapı R5'in gerisinde | Kanal başına ölü kanal dedektörü (Poisson sessizlik testi, §12.17); ön-eğitimde kanal düşürme augmentation'ı (§12.19) |
| Kanal kimliği kayması (Indy 0630 ≤ 0,14) | Degenhart kararlı kanal hizalaması (§12.18); yetmezse NoMAD |
| 500 ms gecikmeli LMS η'yı tıkıyor; gerçek BCI'da hız etiketi yok | Sönümlü ReFIT niyet etiketi, gecikmesiz (§12.15) |
| Loco ateşleme oranı 0,016 (KV7 bandı altı) | Raporlanıyor; ayrı iş |

## Kurulum ve çalıştırma

```bash
uv sync
uv run pytest
uv run python scripts/download_primate.py --config configs/v2_base.yaml          # 6 NeuroBench + 6 ayar oturumu
uv run python experiments/v2_pretrain.py --config configs/v2_pretrain_v7.yaml ...  # augmentation'lı ön-eğitim
uv run python experiments/v2_v7.py --config configs/v2_v7.yaml --runs <koşular> --protocol tune --split val
uv run python scripts/make_tables_v2.py --runs <koşular> --main D7 --claims B7 P7 C7 R5 --export results/v7
```

## Belgeler

| Dosya | İçerik |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Kurallar (v2: §12, KV1–KV8), denklemler, hiperparametreler |
| [docs/PhasesV2.md](docs/PhasesV2.md) | v2 fazları V0–V7 ve sonuç satırları |
| [docs/Decision.md](docs/Decision.md) | Karar kayıtları KR-001 … KR-049 (her denemenin sayıları ve gerekçesi) |
| [results/v6/table.md](results/v6/table.md) | V6 test tabloları, oturum birimli istatistik; `rows.json` birim satırları |
| [docs/v6_results.md](docs/v6_results.md) | V6 tablolarının ilk ((oturum, seed) birimli) sürümü, kayıt |
| [docs/State.md](docs/State.md) | Günlük |
| `docs/figures/` | Mimari, duygu izleri, elektrot kaybı toparlanma eğrileri |
| [LICENSE](LICENSE) | MIT |

v1 kodu (sabit LIF rezervuarı, Faz 0–6) `v1-final` git tag'indedir.
