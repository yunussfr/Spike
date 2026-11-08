# Uygulama Fazları — Spike v2

Bu dosya, `docs/spike_v2_mimari_spesifikasyon.md`'deki **Spike v2** mimarisinin fazlara bölünmüş uygulama planıdır. v1 fazları (`docs/PHASES.md`, Faz 0–6) kayıt olarak durur; v2 fazları **V0–V6** diye numaralanır, karıştırılmaz.

Kurallar:
- Bir fazın **çıkış kriteri** sağlanmadan sonraki faza geçilmez.
- Tüm v2 kodu `ARCHITECTURE.md` §12'deki v2 kurallarına (**KV1–KV8**) uyar. Çelişki varsa önce ARCHITECTURE.md güncellenir (kullanıcı onayıyla), sonra kod.
- Her fazın sonunda: sonuç bu dosyaya (faz başlığının altına `> Sonuç:` satırı), karar `docs/Desicion.md`'ye (KR-xxx), kısa günlük `docs/State.md`'ye yazılır.
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

## Faz V5 — Beyincik v2 ve duygu uzayı v2 (sürpriz + alt-popülasyon kapılama)

**Hedef:** Tam önerilen sistem **D** ve kontrolü **C (karıştırılmış η)**.

**Yapılacaklar:**
- `CerebellumV2`: φ = [r̄, v̂(t−1)] (efference copy), hedef x̄(t+1), yerel NLMS.
- `AffectSpaceV2`: E_fast / E_slow, a(t) = tanh((E_fast − E_slow)/E_slow), valence, dominance; g_fast, g_slow, η(t). τ çevrimiçi değişmez (KV4).
- D'nin η(t) dizisi kaydedilir; C bu diziyi zamanda karıştırarak uygular; B, D'nin η̄'sını kullanır (eşlenmiş η).

**Testler:**
- K3/KV3: e yalnızca W_cer'i, e_v yalnızca okuma başlıklarını günceller; duygu uzayı ikisini yalnızca okur.
- Beyincik x̄(t+1)'i tahmin anında görmüyor (KV6).
- Regülatör çıktıları sınırlar içinde (KV4); `regulators.enabled: false` → g = 1, η sabit (KV5).
- Karıştırılmış dizinin ortalaması ve dağılımı D'ninkiyle aynı.

**Çıkış kriteri:**
- Ayar seed'lerinde D, B (eşlenmiş η̄) ve C val sonuçları; a(t), g, η izleri grafikte.
- Kapılama ile η modülasyonunun katkısı ayrı ayrı ölçülmüş (D − kapılamasız D).

---

## Faz V6 — Asıl deney: günler arası kayma ve elektrot kaybı

**Hedef:** v2 araştırma sorusunu cevaplamak.

**Yapılacaklar:**
- Koşullar A–E (§12.9), aynı test akışı, 5 rapor seed'i, Indy + Loco.
- Senaryolar: oturum içi test, Day-0 → Day-k, E (akış ortasında 20 kanal sıfırlanır).
- Metrikler: R², toparlanma süresi (E), ateşleme oranı, η/g izleri.
- İstatistik: D–B ve D–C için Wilcoxon (5 seed × oturum).
- `scripts/make_tables.py` v2 tabloları.

**Çıkış kriteri:**
- Sonuç tablosu ve grafikler hazır.
- Ana iddia (D > B ve D > C) ya destekleniyor ya desteklenmiyor; ikisi de raporlanabilir.

---

## Faz özeti

| Faz | Ana çıktı | Koşul |
|---|---|---|
| V0 | 4 ms veri, protokol, 4 ms GRU tavanı | — |
| V1 | Kodlayıcı v2 | — |
| V2 | Eğitilebilir omurga, mod seçimi | — |
| V3 | Ön-eğitim + Day-0 + 5 başlıklı okuma | A |
| V4 | Çapalı NLMS akışı | B |
| V5 | Beyincik v2 + duygu v2 | C, D |
| V6 | Kayma ve elektrot kaybı deneyi | A–E |
