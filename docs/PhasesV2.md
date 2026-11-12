# Uygulama Fazları — Spike v2

Bu dosya, `docs/spike_v2_mimari_spesifikasyon.md`'deki **Spike v2** mimarisinin fazlara bölünmüş uygulama planıdır. v1 fazları (`docs/PHASES.md`, Faz 0–6) kayıt olarak durur; v2 fazları **V0–V6** diye numaralanır, karıştırılmaz.

Kurallar:
- Bir fazın **çıkış kriteri** sağlanmadan sonraki faza geçilmez.
- Tüm v2 kodu `ARCHITECTURE.md` §12'deki v2 kurallarına (**KV1–KV8**) uyar. Çelişki varsa önce ARCHITECTURE.md güncellenir (kullanıcı onayıyla), sonra kod.
- Her fazın sonunda: sonuç bu dosyaya (faz başlığının altına `> Sonuç:` satırı), karar `docs/Decision.md`'ye (KR-xxx), kısa günlük `docs/State.md`'ye yazılır.
- Ayar yalnızca ayar seed'leri (100–102) ve val bölmesinde yapılır. Rapor seed'leri (0–4) ve test bölmesi her faz için bir kez koşulur.
- Spesifikasyondaki sayılar (Δt = 4 ms, 100/150 epoch, α_gate = 0,25 …) **başlangıç değeridir**; ölçüm başka bir değer gösterirse KR kaydıyla değiştirilir.

**Hedef:** NeuroBench Primate Reaching (Indy & Loco) oturum içi test R² ≥ 0,65–0,70; günler arası kaymada ve elektrot kaybında, sürpriz modülasyonlu çevrimiçi uyumun (D) sabit-η̄ (B) ve karıştırılmış-η (C) kontrollerinden iyi olup olmadığını ölçmek.

**Başlangıç noktası (KR-033):** Aynı 20 ms ızgarada GRU tavanı Indy 0,736 / Loco 0,685; v1 rezervuarı + doğrusal okuma Indy 0,442 / Loco 0,365. Kayıp okuma kuralında değil, doğrusal okunan temsilde. v2'nin eğitilebilir omurgası bu bulguya dayanır.

**Bilinen risk (KR-031, KR-032):** v1'de sürpriz/duygu kapılı η, eşlenmiş sabit η'ya göre katkı vermedi. v2'nin ana iddiası (D > B) bu yüzden önceden kanıtlanmış sayılmaz; B ve C kontrolleri zorunludur.

---

## Faz V0 — Altyapı, veri ve 4 ms tavanı  ✅ Tamamlandı (2026-11-04)

> Sonuç: 4 ms veri ve 80/10/10 bölme hazır (KR-035). 4 ms test R²: GRU Indy 0,708 / Loco 0,616, Wiener (400 ms) 0,513 / 0,432 (KR-037). GRU 20 ms'deki değerinin altında (Indy −0,03, Loco −0,07), yani alt sınır. Bütçe ölçümü V2'de (KR-036). Testler 153/153. Koşu: `runs/20261103-214626_v2_ceiling_seeds0-1-2-3-4`.

**Hedef:** v2 kodunu v1'i bozmadan yazabilecek iskelet ve yeni protokolün tavan ölçümü.

**Yapılacaklar:**
- Kod yerleşimi (öneri, bu fazda onaylanır): v2 modülleri ayrı dosyalarda (`src/v2/encoder.py`, `backbone.py`, `cerebellum.py`, `affect.py`, `decision.py`, `model.py`); v1 sınıfları ve testleri olduğu gibi kalır. Config: `configs/v2_base.yaml`, `configs/v2_nhp.yaml`.
- Veri: `bin_width = stride = 0,004 s` önbelleği (NeuroBench yükleyicisi zaten 4 ms çözünürlükte açıyor). İsteğe bağlı 8 ms hızlı mod.
- Bölme protokolü kararı (KR): spesifikasyon %65 train / %10 val / %25 test; v1 80/10/10; NeuroBench 75/25. Seçilen protokol tüm v2 fazlarında sabit kalır.
- Günler arası (cross-day) protokol: her primat için oturumların tarih sırası, Day-0 kaynak oturumu ve Day-k hedef oturumları. Kanal sayısı oturumlar arasında farklıysa (Indy/Loco) eşleme kuralı.
- Pencereleme: T = 500 adım (2 s), stride = 250, yalnızca train diliminden.
- Tavan yeniden ölçümü (`experiments/ceiling.py`'nin 4 ms sürümü): Wiener (H ∈ {25, 50, 100}), GRU (BPTT). Bu, v2 hedefinin yeni protokoldeki referansıdır.
- ~~Hesap bütçesi ölçümü~~ → Faz V2'ye ertelendi (KR-036): prototip adım yerine gerçek omurgayla ölçülür.

**Testler:**
- v1 testlerinin hepsi geçiyor (`uv run pytest`).
- 4 ms önbellek: kutu toplamları 20 ms önbelleğinin 5'li toplamlarına eşit.
- Pencereler train sınırını aşmıyor; val/test'e taşma yok.

**Çıkış kriteri:**
- 4 ms veri ve bölmeler hazır, protokol KR olarak kayıtlı. ✔ (KR-035)
- 4 ms GRU tavan tablosu (6 oturum, 5 seed) var. ✔ (KR-037)
- ~~Epoch süresi ölçülmüş~~ → V2 çıkış kriterine taşındı (KR-036).

---

## Faz V1 — Kodlayıcı v2 (Anscombe + kanal z-skoru + yumuşak kırpma)  ✅ Tamamlandı (2026-11-05)

> Sonuç: `src/v2/encoder.py`, 6 test. 4 ms'de sayımlar ikili olduğundan dönüşüm kanal başına afin; kodlanmış ve ham Wiener R² birebir aynı (KR-038). Bilgi kaybı yok.

**Hedef:** M1/S1 genlik farkını veri katmanında dengeleyen sürekli girdi ve beyincik için yumuşatılmış hedef (§12.3).

**Yapılacaklar:**
- `SpikeEncoderV2`: `fit(x_train)` (μ_ch, σ_ch yalnızca train), `forward(x_raw) → (x_norm, x̄)`, `reset_state()`.
- Sabitler (0,375, kırpma ölçeği 3, τ_smooth = 48 ms, ε) config'ten (KV8).

**Testler:**
- İstatistikler yalnızca train'den; val/test verisi değişince μ, σ değişmiyor.
- |x_norm| < kırpma ölçeği; x̄ nedensel (x̄[t], x[t+1]'e bağlı değil).
- Toplu (`[B, T, C]`) ve adım adım akış aynı sonucu veriyor.

**Çıkış kriteri:**
- Testler geçiyor.
- Kodlanmış girdi üzerinde Wiener R², ham sayım Wiener'inden kötü değil (bilgi kaybı yok).

---

## Faz V2 — Eğitilebilir zamansal omurga (`plif` | `spiking_gru` | `tau_gru`)  ✅ Tamamlandı (2026-11-05)

> Sonuç (ayar seed'leri, val, tek oturum, KR-039): plif Indy 0,769 / Loco 0,658; tau_gru 0,770 / 0,666; spiking_gru 0,688 / 0,535. 4 ms GRU tavanı (val) 0,714 / 0,571. Ana mod: plif (kullanıcı kararı). KV7: plif Loco yavaş alt-popülasyon oranı 0,014 < 0,02 (✗), düzeltilmeden raporlanıyor. Bütçe: 4 ms'de seed başına tam protokol ~0,6–0,8 saat (KR-038). Koşu: `runs/20261104-233122_v2_backbone_seeds100-101-102`.

**Hedef:** 500 hızlı + 500 yavaş alt-popülasyonlu, çevrimdışı eğitilen omurga (§12.4). Önce tek oturumda, tek doğrusal okumayla.

**Yapılacaklar:**
- `TemporalBackbone`: üç mod, öğrenilebilir τ_mem (alt-popülasyon sınırları içinde), τ_syn, surrogate gradyan, dalgalanma güdümlü / chrono başlatma, homeostatik kayıp, `reset_state()`, `step(...)`.
- Eğitim döngüsü: AdamW + CosineAnnealing, kayıp = RMSE + λ_homeo·L_homeo; val'de en iyi epoch.
- Ateşleme oranı alt-popülasyon başına loglanır (KV7).
- Hesap bütçesi ölçümü (V0'dan taşındı, KR-036): `experiments/v2_bench.py` gerçek `TemporalBackbone` ile; N = 1000, T = 500, batch 32'de bir epoch süresi (GPU, RTX 4060). 100 + 150 epoch'un gerçekçi olup olmadığı burada görülür.

**Testler:**
- τ_mem her zaman kendi alt-popülasyon aralığında; τ_syn aralığında.
- Surrogate gradyan: ileri yönde ikili spike, geri yönde sonlu gradyan.
- Çevrimiçi (eval) modda omurga parametreleri değişmiyor; yalnızca homeostatik eşik değişiyor (KV1).
- `tau_gru` modunda eşik/spike yolu yok.
- Durum dizi içinde sıfırlanmıyor (KV6).

**Çıkış kriteri:**
- Tek oturum (ayar seed'i, val): `tau_gru` 4 ms GRU tavanına yakın (fark ≤ ~0,03).
- `plif` ve `spiking_gru` R² raporlanmış; ortalama ateşleme oranı [f_min, f_max] içinde.
- Üç modun karşılaştırma tablosu; hangi modun ana hat olacağı KR ile seçilmiş.
- Epoch süresi ölçülmüş; epoch sayıları gerekiyorsa KR ile güncellenmiş (KR-036).

---

## Faz V3 — Çoklu oturum ön-eğitimi, Day-0 ince ayarı, 5 başlıklı LI okuma  ✅ Tamamlandı (2026-11-08)

> Sonuç (KR-042): ön-eğitimli donuk model (A) test R², 5 seed: Indy 0,794 / 0,652 / 0,816, Loco 0,742 / 0,669 / 0,702 (ortalama 0,754 / 0,704; kriter ≥ 0,65 sağlandı). İnce ayar testte ön-eğitimle aynı. Ayar seed'lerinde 1 / 5 başlık ve ön-eğitimli / sıfırdan farkı ±0,02 içinde. Day-0 modeli kendi gününde 0,80 / 0,73, sonraki günlerde ≈ 0 (Indy −0,02 / 0,11, Loco 0,01 / −0,37). KV7: Loco hızlı alt-popülasyon ~0,01–0,015 ✗, raporlanıyor. Uzun süreçte CUDA Graph çöküşü → seed / aşama başına süreç (KR-043). Tablolar: `runs/v3_report_{indy,loco}_table.md`.

**Hedef:** Tam çevrimdışı boru hattı (Aşama 1–2) ve donuk modelle oturum içi hedefe ulaşmak.

**Yapılacaklar:**
- `MultiHeadDecisionBlock`: τ_read ∈ {16, 32, 64, 100, 160} ms sızıntılı entegratör izleri, başlık ortalaması, `save_anchor_weights()`.
- Aşama 1: tüm oturumların train pencereleriyle ön-eğitim (batch 32, AdamW, cosine).
- Aşama 2: hedef oturumda ince ayar, val'de en iyi epoch, ardından W_0 çapası kaydı.
- Checkpoint: ön-eğitim ve Day-0 ağırlıkları, config + seed + git hash (KV8).

**Testler:**
- Çapa kaydından sonra W_0 hiçbir yolda değişmiyor.
- LI izleri nedensel; tek başlık + τ_read → 0 limitinde tek doğrusal okumaya iniyor.

**Çıkış kriteri:**
- Koşul **A (donuk)**: 6 oturum × 5 rapor seed'i, oturum içi test R². Hedef ≥ 0,65 (Indy ve Loco ortalaması ayrı).
- 1 başlık / 5 başlık ve ön-eğitimli / ön-eğitimsiz farkı ayar seed'lerinde ölçülmüş.
- Hedefe ulaşılamazsa: teşhis tablosu (mod, Δt, pencere, epoch) ve kullanıcı kararı.

---

## Faz V4 — Akışlı çevrimiçi uyum: çapalı NLMS ve çevrimiçi homeostaz  ✅ Tamamlandı (2026-11-08)

> Sonuç (KR-041, KR-042): etiket 500 ms gecikmeli. B − A (test, η val'den): oturum içi Indy +0,001, Loco +0,006; günler arası Day-k'de en çok +0,2, R² 0 civarında kalıyor. Sabit hedefli (ρ = 0,05) çevrimiçi homeostaz oturum içinde zararlı (Indy 0,80 → 0,63), Indy 0131 günler arasında yararlı (0,11 → 0,49). 500 ms kopya tabanı −0,85 … −1,03. Koşular: `runs/20261108-045545_v2_online_test_seeds0-1-2-3-4`, `runs/20261108-050007_v2_online_test_seeds0-1-2-3-4`.

**Hedef:** Aşama 3'ün temel çizgisi: koşul **B** (sabit η, kapılama yok).

**Yapılacaklar:**
- Akış değerlendirmesi: test dilimi pencerelenmeden adım adım (`eval`, gradyan kapalı), prequential sıra (önce ŷ_t skorlanır, sonra güncellenir).
- `online_anchored_nlms_update` (§12.7); `online_homeostasis` açık/kapalı.
- η, α_anchor taraması yalnızca ayar seed'leri + val.

**Testler:**
- Sıfır sızıntı: y_t değişince ŷ_t değişmiyor (v1 `test_zero_leakage` eşdeğeri).
- η = 0 → A ile bit düzeyinde aynı çıktı.
- α_anchor = 1 → W her adımda W_0'a dönüyor; büyük hata altında W sonlu kalıyor.

**Çıkış kriteri:**
- B − A: oturum içi ve Day-0 → Day-k, 5 rapor seed'i.
- Çevrimiçi homeostazın ateşleme oranı ve R²'ye etkisi ayrı raporlanmış.

---

## Faz V5 — Beyincik v2, duygu uzayı v2 ve duygu-kapılı W_in plastisitesi (KR-044)  ✅ Tamamlandı (2026-11-08)

> Sonuç (KR-045–KR-047, ayar seed'leri, val): etiketsiz kanal yeniden ölçekleme günler arası Day-k R²'yi 0,02 → 0,20'ye çıkarıyor ama ilk günde −0,04; W_in kuralı okuma uyumunu geçmedi (kapalı). Kapalı döngü duygu-kapılı ölçekleme söndü (eşlenmiş sabiti geçmedi); kalıcı sürpriz sensörü (ham hedefli donuk izleme beyinciği + girdi sapması) ilk günü koruyor (Pm Day-0 0,744, R5 0,707). Dinamik duygu: kullanıcının iç arousal'ı günü ayırmıyor (başlık uyuşmazlığı kaymada düşüyor); sensör arousal'ıyla gaz + çapa gevşetme (v2-D) tüm günlerde 0,397 (R5 0,371, O 0,382, günler arası eşlenmiş sabit Pg 0,373), Day-k 0,224. Gün içi zamanlamanın payı küçük (C +0,003, P +0,006); valens freni yalnız aşırı gazda (10·η_0) koruyor (0,278 → 0,365), ölçülü gazda zararlı. g kapısı nötr. İzler `docs/figures/v5_*.png`.

**Hedef:** Tam önerilen sistem **D** ve kontrolleri **C (karıştırılmış η, η_in)**, **W (duygusuz W_in kuralı)**, **L (etiketsiz)**.

**Yapılacaklar:**
- `CerebellumV2`: φ = [r̄, v̂(t−1)] (efference copy), hedef x̄(t+1), yerel NLMS.
- `AffectSpaceV2`: E_fast / E_slow, a(t) = tanh((E_fast − E_slow)/E_slow), valence, dominance; g_fast, g_slow, η(t). τ çevrimiçi değişmez (KV4).
- D'nin η(t) dizisi kaydedilir; C bu diziyi zamanda karıştırarak uygular; B, D'nin η̄'sını kullanır (eşlenmiş η).
- **Önce etiketsiz kontroller (L):** Day-0 hedefli homeostaz, akan kanal z-skoru; ayar seed'leri, val, günler arası. Sonuç kullanıcıya.
- `TemporalBackbone.step(x, w_in=...)` koşul başına W_in; W_in üç faktörlü kuralı (§12.13), η_in(t) = η_in,0·max(0, a). W_cer Day-0 train'de ridge ile başlatılır.
- **Karar kapısı:** W val'de B'yi geçmiyorsa duygu kapısı anlamsız → kullanıcıya dönülür.
- **Eklenenler (KR-045–KR-047):** duygu-kapılı kanal yeniden ölçeklemesi (§12.14); kalıcı sürpriz sensörü (ham hedefli donuk izleme beyinciği + girdi sapması, sabit Day-0 tabanı, eşikli kapı, dönüş); dinamik duygu (arousal düzey, valens e_v eğilimi, dominans e_v düzeyi → gaz, fren, çapa gevşetme, akış içi konsolidasyon; §12.6, §12.7). Kontroller v2-R, v2-P, Pm3 (sabit 3·η_0), O (gün bilgisiyle üst sınır).

**Testler:**
- K3/KV3: e_cer yalnızca W_cer'i, e_v yalnızca okuma başlıklarını ve W_in'i günceller; duygu uzayı ikisini yalnızca okur.
- KV1: W_rec, τ, bias hiç değişmiyor; ‖W_in − W_in,0‖ ≤ δ_max·‖W_in,0‖; akış sonrası W_in geri yükleniyor; a ≤ 0 iken W_in yalnız çapaya doğru gidiyor.
- η_in = 0 ve duygu kapalı → V4 akışıyla aynı çıktı. `step(x, w_in=weight)` = `step(x)`.
- Beyincik x̄(t+1)'i tahmin anında görmüyor (KV6).
- Regülatör çıktıları sınırlar içinde (KV4); `regulators.enabled: false` → g = 1, η sabit (KV5).
- Karıştırılmış dizinin ortalaması ve dağılımı D'ninkiyle aynı.

**Çıkış kriteri:**
- Ayar seed'lerinde D, B (eşlenmiş η̄) ve C val sonuçları; a(t), g, η izleri grafikte.
- Kapılama ile η modülasyonunun katkısı ayrı ayrı ölçülmüş (D − kapılamasız D).
- L, W, D val sonuçları; ‖W_in − W_in,0‖, a(t), η_in(t) izleri.

---

## Faz V6 — Asıl deney: günler arası kayma ve elektrot kaybı  ✅ Tamamlandı (2026-11-08)

> Sonuç (KR-048, test, 5 seed, Indy + Loco): günler arası Day-k R² A −0,066, v2-B 0,039, R5 0,251, v2-P 0,272, v2-C 0,277, **v2-D 0,279**; tüm günler D 0,444 (R5 0,413, O 0,425, Pg 0,418); Day-0 D 0,775 (R5 0,738). D > B (20/20, p_Holm < 0,001) ve D > P (Day-k +0,007, 16/20, p_Holm = 0,0008) destekleniyor; D > C desteklenmiyor (+0,002, p = 0,10): kazanç gün içi zamanlamadan değil, günün algılanmasından. Elektrot kaybında sürekli yeniden ölçekleme en iyi (R5 kayıp sonrası 0,652, toparlanma 11 s; D 0,604, 17,5 s). Oturum içinde fark yok (Indy 0,750, Loco 0,710; NeuroBench SNN 0,593 / 0,568). Kullanıcının üç kuralı yazıldığı gibi (D_user) Day-k 0,187. Tablo `docs/v6_results.md`, grafikler `docs/figures/v6_*.png`.

**Hedef:** v2 araştırma sorusunu cevaplamak.

**Yapılacaklar:**
- Koşullar (§12.9): A, v2-B (η̄, ᾱ eşlenmiş), v2-P (sensörlü yeniden ölçekleme + η̄, ᾱ), v2-C (η, α, c dizileri karıştırılmış), v2-D; bağlam için Pm, Pm3, R5, O. v2-W dışarıda (KR-045). Aynı test akışı, 5 rapor seed'i, Indy + Loco (`experiments/v2_v6.py`, `runs/chain_v6.sh`).
- Senaryolar: oturum içi test, Day-0 → Day-k, E (oturum içi model; değerlendirme diliminin ortasında seed'e bağlı 20 kanalın ham sayımı 0).
- Metrikler: R², kayıp sonrası R² ve toparlanma süresi (E), ateşleme oranı, a/v/d/η/α/g izleri.
- İstatistik: D–B, D–C, D–P için kesin Wilcoxon (5 seed × oturum) ve Holm (`src/v2/stats.py`).
- `scripts/make_tables_v2.py` v2 tabloları; `scripts/plot_v2_traces.py` izler ve toparlanma eğrileri.

**Çıkış kriteri:**
- Sonuç tablosu ve grafikler hazır.
- Ana iddia (D > B, D > C ve D > P) ya destekleniyor ya desteklenmiyor; ikisi de raporlanabilir.

---

## Faz V7 — Değişim noktası tespitiyle kapılanan girdi uyumu ve sönümlü ReFIT (KR-049)  ⏳ Kod hazır, koşu bekliyor

**Hedef:** Hakem raporuna (2026-11-09) cevap: (1) istatistik birimi oturum, (2) ayar oturumları test dışı, (3) "duygu" yerine tanımlı istatistiksel nicelik (BOCPD: a = P(rejim ≠ Day-0)), (4) literatür tabanları (Degenhart, R5), (5) gecikmeli LMS sınırını kaldıran niyet etiketi. Araştırma sorusu §12.1 (V7).

**Yapılanlar (kod, 2026-11-09):**
- §12.15 `src/v2/intent.py` (sönümlü ReFIT; `.mat`'tan imleç / hedef konumu, `load_task_state`), §12.16 `src/v2/changepoint.py` (BOCPD, Day-0 ayrılmış diliminde tavlama kalibrasyonu), §12.17 `src/v2/dead_channel.py`, §12.18 `src/v2/align.py` (FA + Procrustes + kararlı kanallar, graf dışında yeniden uydurma), §12.19 `src/v2/augment.py` (eğitim döngüsünde).
- Akışa bağlama: `Condition.sensor / dead / align`, `V7Inputs` (`src/v2/stream_spec.py`); `online_v5.py` bölündü (`drift.py`, `stream_spec.py`).
- Ayar / test ayrımı: `tune_primates`, `data.tune_files` (6 test dışı oturum), `check_tune_disjoint`; `scripts/download_primate.py --only tune`.
- İstatistik: `src/v2/stats.py` (oturum ortalaması, küme bootstrap, Kaplan–Meier, ulaşılabilir en küçük p); `scripts/make_tables_v2.py` oturum birimli; V6 yeniden tablolandı: `results/v6/table.md`.
- Deney: `experiments/v2_v7.py` (`configs/v2_v7.yaml`), ön-eğitim `configs/v2_pretrain_v7.yaml`.
- Testler: `tests/v2/test_v7.py` (20), augmentation ve ayar / test ayrımı testleri; v1 kodu `v1-final` tag'ine taşındı.

**Ek (KR-050, 2026-11-10):** Teşhiste ilk BOCPD Day-0'da %90–97 yanlış alarm verdi. §12.16 tek boyutlu skor + Day-0'a dönüş (CR); kaynak `bocpd.source` (ar1 ana, cer_mon = donuk beyincik artığı karşılaştırma). Yeni §12.20 tonik / fazik iş bölümü: BOCPD → α, λ_norm; plastik beyincik (hatanın kopya tahminine oranı) → η, g. Beyincik ailesi D7_split, D7_ph0, D7_N0 ve geçiş 5 (D7@cer_mon, D7_split@cer_mon) ikincil, keşifsel aile. Teşhis betikleri `scripts/diagnose_{cer_bocpd,bocpd_models,bocpd_cr,phasic}.py`. Ayar zinciri `runs/chain_v7_tune.sh`, Colab rapor eğitimi `notebooks/`.

**Yapılacaklar (koşu):**
1. Ayar oturumlarını indir: `uv run python scripts/download_primate.py --config configs/v2_base.yaml --only tune`.
2. Augmentation'lı V3 protokolü (ayar seed'leri, sonra rapor seed'leri): `experiments/v2_pretrain.py --config configs/v2_pretrain_v7.yaml`, primat × seed ayrı süreç (KR-043).
3. Ayar: `experiments/v2_v7.py --protocol tune --split val` (ayar seed'leri). Seçilecekler: r_d, r_on, t_rt; BOCPD hazard, kapı, p_0 ve kaynak (önceden kayıtlı kural, §12.16); η_cer (fazik kol); hizalama q, stable_frac, T_cal; η_0. Test oturumlarına dokunulmaz.
4. Rapor: `--protocol test --split test` (seed 0–4), `--runs_v6` ile D7@v6. Tablolar: `scripts/make_tables_v2.py --main D7 --claims B7 P7 C7 R5 --extra G7 D7_nodead D7_noalign D7_delay D7@v6 A --title "Faz V7" --export results/v7`.

**Testler:** ReFIT etiketi y'yi okumuyor ve prequential; BOCPD olasılıkları normalize, maske / ölü kanal dışlama, sabit veride sakin, kaymada birkaç blokta tespit; ölü kanal dedektörü nedensel; FA + Procrustes sentetik dönmeyi ve kararsız kanalları buluyor; hizalayıcı nedensel ve kapılı; CUDA Graph = adım adım (BOCPD + ölü kanal + hizalama + ReFIT birlikte); augmentation yalnız eğitimde, kapalıyken V6 ile aynı.

**Çıkış kriteri:**
- Ayar yalnız ayar oturumlarında yapılmış; test koşusu bir kez, seçilen değerlerle.
- Birincil aile (D7 − B7, D7 − P7, D7 − C7, D7 − R5) oturum birimli raporlanmış (4 Day-k oturumunda ulaşılabilir en küçük p 0,125 açıkça yazılı). Yeterli oturum yoksa sonuç "etki büyüklüğü + %95 GA" olarak raporlanır, anlamlılık iddiası yapılmaz.
- Tespit gecikmesi (Indy 0131 Day-k'de a ≥ 0,5 anı) ve elektrot kaybında D7 − R5 raporlanmış.
- **NoMAD kararı:** D7, Indy 0630'da (kanal kimliği kayması) G7 / R5'ten iyi değilse ya da Day-k ortalaması R5'i geçmiyorsa, sonraki adım NoMAD tipi dinamik hizalamadır (ayrı faz, ayrı KR).

---

## Faz özeti

| Faz | Ana çıktı | Koşul |
|---|---|---|
| V0 | 4 ms veri, protokol, 4 ms GRU tavanı | — |
| V1 | Kodlayıcı v2 | — |
| V2 | Eğitilebilir omurga, mod seçimi | — |
| V3 | Ön-eğitim + Day-0 + 5 başlıklı okuma | A |
| V4 | Çapalı NLMS akışı | B |
| V5 | Beyincik v2 + duygu v2 + duygu-kapılı W_in, kalıcı sürpriz sensörü, dinamik duygu | C, D, W, L, R, P |
| V6 | Kayma ve elektrot kaybı deneyi | A, B, P, C, D, E (+ Bg, Pg) |
| V7 | BOCPD sensörü (CR), ölü kanal, Degenhart, sönümlü ReFIT, augmentation, tonik / fazik iş bölümü; oturum birimli istatistik, test dışı ayar | A, B7, R5, G7, P7, C7, D7 (+ ablasyonlar, beyincik ailesi) |
