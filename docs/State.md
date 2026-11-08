# Proje Durumu (Günlük)

Her gün sonunda güncellenir. En yeni gün en üstte.

**Şu an:** Spike v2, V0–V6 tamamlandı (KR-048). Günler arası test Day-k R² v2-D 0,279 (A −0,066, v2-B 0,039, R5 0,251); D > B ve D > P destekleniyor, D > C desteklenmiyor. Oturum içi Indy 0,750 / Loco 0,710.
**Açık kararlar:** v2 — sensör tespit gecikmesi (Indy 0131) ve elektrot kaybında sensörlü kapı (R5'in gerisinde). Loco KV7 ihlali raporlanıyor. v1 — θ(τ) "geçici", `γ_rec` = 1.0.

---

## 2026-11-08 — Faz V5 kapanışı ve Faz V6 (KR-046–KR-048)

**Yaptıklarım**
- [x] Kalıcı sürpriz sensörü (ham hedefli donuk izleme beyinciği + girdi sapması, sabit Day-0 tabanı, eşikli kapı, dönüş; KR-046).
- [x] Dinamik duygu: arousal / valens / dominans → gaz, fren, çapa gevşetme, konsolidasyon (`AffectDynamics`, KR-047); ortak sensör, akış içi çapa, eşlenmiş kontroller (v2-P, Bg, Pg), elektrot kaybı senaryosu, kesin Wilcoxon (`src/v2/stats.py`), `experiments/v2_v6.py`, `scripts/make_tables_v2.py`, `scripts/plot_v2_traces.py`; v2 + v1 değişmez testleri geçiyor.
- [x] Teşhis: sabit η 3×'ün üstünde zararlı, 10× ıraksıyor; başlık uyuşmazlığı kaymada düşüyor, iç beyincik hatası günü ayırmıyor → arousal kaynağı val'de sensör seçildi; valens kontrolü ölçülü gazda zararlı.
- [x] V6 test (seed 0–4):

  | | Day-k | Tüm günler | Day-0 | Elektrot sonrası |
  |---|---|---|---|---|
  | v2-B | 0,039 | 0,284 | 0,773 | 0,545 |
  | R5 | 0,251 | 0,413 | 0,738 | 0,652 |
  | v2-C | 0,277 | 0,443 | 0,775 | 0,608 |
  | v2-D | 0,279 | 0,444 | 0,775 | 0,604 |

**Etkisi**
- Duygu "ne zaman uyum sağla" anahtarı olarak işe yarıyor (D > B, P, Pg, Pm3; ilk gün korunuyor); gün içi zamanlama katkısız (D ≈ C).

**Açık**
- 0131 tespit gecikmesi; elektrot kaybında sensörlü kapı R5'in gerisinde.

---

## 2026-11-08 — Faz V3 + V4 sonuçları, CUDA Graph çöküşü (KR-042, KR-043)

**Yaptıklarım**
- [x] V3 ayar (seed 100–102) ve rapor (seed 0–4, test) koşuları; V4 val (η seçimi) ve test.
- [x] Çöküş (replay'de access violation): seed / aşama başına süreç, `--resume` birden çok klasör ve ince ayar / day0 checkpoint'leri, `experiments/v2_collect.py`.

  | | Indy test R² (3 oturum) | Loco test R² (3 oturum) |
  |---|---|---|
  | Ön-eğitimli donuk (A) | 0,794 / 0,652 / 0,816 | 0,742 / 0,669 / 0,702 |
  | Day-0 modeli, Day-k (A) | −0,017 / 0,108 | 0,010 / −0,366 |
  | Day-0 modeli, Day-k, okuma NLMS (B) | −0,021 / 0,128 | 0,030 / −0,162 |

**Etkisi**
- V3 kriteri (≥ 0,65) sağlandı. Günler arası boşluk büyük ve okuma uyumu kapatmıyor → V5 tasarımı genişletildi: W_in sürpriz-kapılı plastik, W_rec donuk (KR-044).

**Açık**
- Etiketsiz kontroller (Day-0 hedefli homeostaz, akan kanal z-skoru) V5'in ilk adımı.

---

## 2026-11-05 — Faz V1 + V2: kodlayıcı, omurga, CUDA Graph, mod karşılaştırması (KR-038, KR-039)

**Yaptıklarım**
- [x] `src/v2/encoder.py`, `backbone.py`, `decision.py`, `model.py`, `train.py`; testler `tests/v2/` (37 test).
- [x] Eğitim adımı CUDA Graph ile: batch başına 2,4 s → 0,094 s. Bütçe (KR-036 kapandı): 4 ms'de seed başına tam protokol plif ~0,6–0,8 saat.
- [x] `spiking_gru` ϑ başlangıcı 0,5 (ϑ = 1 ile hiç ateşlemiyordu).
- [x] Mod karşılaştırması `runs/20261104-233122_v2_backbone_seeds100-101-102` (ayar seed'leri, val):

  | | plif | spiking_gru | tau_gru | 4 ms GRU tavanı |
  |---|---|---|---|---|
  | indy_20170131_02 | 0,769 | 0,688 | 0,770 | 0,714 |
  | loco_20170215_02 | 0,658 | 0,535 | 0,666 | 0,571 |

**Etkisi**
- Eğitilen omurga v1'in ~2,5–4 katı val R²; plif ≈ tau_gru, yani spike'a geçmek kaybettirmiyor.
- V1 kriteri tanım gereği sağlandı: 4 ms'de sayımlar ikili, kodlayıcı afin.

**Açık**
- 2026-11-07: ana mod plif onaylandı; Loco KV7 ihlali düzeltilmeden raporlanacak (KR-039).
- Değişiklikler commit edilmedi.

---

## 2026-11-04 — Faz V0: veri katmanı, protokol kararı, 4 ms tavanı (KR-035, KR-036)

**Yaptıklarım**
- [x] `src/v2/data.py`: Δt tutarlılık kontrolü, train-içi pencereler, günler arası sıra; testler `tests/v2/test_v0_data.py`.
- [x] `configs/v2_base.yaml`, `configs/v2_ceiling.yaml`, `experiments/v2_ceiling.py`, `experiments/v2_bench.py`.
- [x] KR-035: bölme 80/10/10 (v1 ile aynı), kod `src/v2/`, Day-0 = primat başına ilk oturum. ARCHITECTURE.md §12.10 nokta 1–2 kapandı.
- [x] KR-036: epoch süresi ölçümü V2'ye ertelendi (omurga yokken prototip ölçüyordu).
- [x] 4 ms tavanı (`runs/20261103-214626_v2_ceiling_seeds0-1-2-3-4`, KR-037). V0 kapandı.

  | Oturum | Wiener 400 ms (test) | GRU 4 ms (5 seed) | GRU 20 ms (KR-033) |
  |---|---|---|---|
  | indy_20170131_02 | 0,580 | 0,752 | 0,779 |
  | indy_20160630_01 | 0,359 | 0,597 | 0,623 |
  | indy_20160622_01 | 0,599 | 0,775 | 0,806 |
  | loco_20170301_05 | 0,518 | 0,632 | 0,689 |
  | loco_20170215_02 | 0,430 | 0,616 | 0,662 |
  | loco_20170210_03 | 0,348 | 0,600 | 0,705 |
  | **Indy / Loco ort.** | 0,513 / 0,432 | **0,708 / 0,616** | 0,736 / 0,685 |

**Etkisi**
- `indy_20160630_01`'de GRU tavanı (0,597) v2 hedefinin (≥ 0,65) altında; bu oturumda hedef gerçekçi değil, değerlendirme tavana göre yapılmalı.
- Tavan koşusu v2 dosyaları commit edilmeden başladı (`meta.json`: 33d5036, dirty). Kod sonradan değişmeden commit edildi.
- İki seed ~11 saat sürdü (uyku); KR-022 geçerli.

---

## 2026-11-03 — Spike v2 dokümanları (KR-034)

**Yaptıklarım**
- [x] `docs/PhasesV2.md`: v2 fazları V0–V6, her biri testler ve çıkış kriteriyle.
- [x] ARCHITECTURE.md: "Sürümler" notu, §12 Spike v2 (KV1–KV8, denklemler, protokol, v2-A…E, açık noktalar), §11 kaydı.
- [x] CLAUDE.md v2'ye göre güncellendi; Desicion.md'ye KR-034.

**Etkisi**
- v1 kuralları v1 kodunda geçerli; v2 kodu yalnızca §12'ye bağlı. Kod değişikliği yok.

**Açık**
- V0: 4 ms önbellek, bölme protokolü kararı, 4 ms GRU tavanı, epoch süresi ölçümü.
- Değişiklikler commit edilmedi.

---

## 2026-11-03 — Faz 6a: sürpriz / duygu kapılı η (KR-032)

**Yaptıklarım**
- [x] `decision.fast.eta_mode`: `surprise` (‖δ‖² kısa/uzun EMA oranı) ve `affect` (duygu uzayı `s` başı); rezervuar sabit τ = 100, varsayılan kapalı. ARCHITECTURE.md değişmedi (onay bekliyor).
- [x] Yeni testler `tests/test_faz6.py`'de (sıfır sızıntı, kapı sınırı, hata adımı, η = 0 bit eşitliği).
- [x] Koşu `runs/20261103-121154_nhp_motor_surprise_faz6_seeds100-101-102` (B+Z, Indy + Loco, seed 100–102, val).

  | | donuk | η 2e-3 | η 4e-3 | η 8e-3 | η 1,6e-2 | sürpriz η | affect η |
  |---|---|---|---|---|---|---|---|
  | Indy | 0,304 | 0,310 | 0,313 | 0,320 | 0,335 | 0,309 | 0,312 |
  | Loco | 0,169 | 0,203 | 0,216 | 0,232 | 0,260 | 0,209 | 0,209 |

**Etkisi**
- Eşlenmiş η'ya göre: sürpriz Indy −0,005 (3/3 kötü), Loco −0,002; affect Indy −0,001, Loco −0,002. Kapılar katkı vermiyor.
- Sabit η arttıkça R² tekdüze artıyor, 1,6e-2'de tepe yok; Loco donuğa göre +0,091 (3/3).

**Açık**
- η ve λ_pull taraması (ayar seed'leri, val; B+Z ve C eşit bütçe); kapılar ayarlı η'da yeniden denenmeli.
- Darboğaz ölçümü (Wiener filtresi / ridge / NLMS) ve K7 ateşleme oranı.
- Oturumlar arası kayma protokolü; rapor seed'leri ve test.
- KR-032 değişiklikleri commit edilmedi.

---

## 2026-11-03 — Faz 6a: eşlenmiş η kontrolü (KR-031 eki)

**Yaptıklarım**
- [x] Faz 6a commit edildi (`e3ecbcc`).
- [x] `faz6.variants` isteğe bağlı üçüncü öğe aldı (`decision.fast` üzerine yazılanlar); config'e iki eşlenmiş η varyantı eklendi.
- [x] Koşu `runs/20261103-104631_nhp_motor_etamatch_faz6_seeds100-101-102` (Indy + Loco, seed 100–102, val, sabit η = 4e-3). Eski varyantlar birebir tekrar etti.

  | | B+Z donuk | B+Z 1e-3 | B+Z 4e-3 | C 1e-3 | C 4e-3 | C η_t |
  |---|---|---|---|---|---|---|
  | Indy | 0,304 | 0,308 | 0,313 | 0,313 | 0,320 | 0,317 |
  | Loco | 0,169 | 0,192 | 0,216 | 0,172 | 0,190 | 0,189 |

**Etkisi**
- C η_t ≈ C eşlenmiş η (Indy −0,002, Loco −0,001): τ kapısı katkı vermiyor.
- C − B+Z (4e-3): Indy +0,006 ± 0,012, Loco −0,025 ± 0,013 (3/3 kötü).
- B+Z 4e-3 Loco'da +0,046 (3/3), şimdiye kadarki en iyi Loco.
- Loco val'inde oturum içi kayma var (R² ilk yarı 0,207 → ikinci yarı 0,136); Indy'de yok.

**Açık**
- η'nın kendisi ayarlanmamış (ΔW sınırdan çok uzak); yalnızca ayar seed'leri/val'de taranmalı, B+Z ve C eşit bütçeyle.
- Çapanın Loco kazancı η'dan bağımsız mı?
- Hata/sürpriz tabanlı η kapısı (τ sapması yerine).
- Oturumlar arası kayma protokolü; rapor seed'leri ve test.
- Eşlenmiş η değişikliği ve dokümanlar commit edilmedi.

---

## 2026-11-02 — Faz 6a: çevrimiçi okuma adaptasyonu (KR-031)

**Yaptıklarım**
- [x] Kod:
  - `DecisionReadout`: W_base + ΔW, elastik NLMS, norm sınırı, d adım gecikme.
  - `Agent.adapt` ve η_t kapısı.
  - Rezervuarda opsiyonel çapa alt-popülasyonu.
  - ARCHITECTURE.md §3.2, §3.6, §8, §11.
- [x] 14 yeni test (sızıntı yok, norm/çapa, η = 0 ile bit eşitliği); toplam 139/139.
- [x] Koşu `runs/20261102-112732_nhp_motor_faz6_seeds100-101-102` (Indy + Loco, seed 100–102, val). R², ortalama:

  | | B+Z donuk | B+Z sabit η | C donuk | C sabit η | C η_t | C+çapa η_t |
  |---|---|---|---|---|---|---|
  | Indy | 0,304 | 0,308 | 0,307 | 0,313 | 0,317 | 0,318 |
  | Loco | 0,169 | 0,192 | 0,157 | 0,172 | 0,189 | 0,206 |

**Etkisi**
- Uyum 12/12 eşli karşılaştırmada pozitif.
- Aynı η'da C, B+Z'den iyi değil: Loco −0,020, 3/3 seed kötü. KR-030'daki "faz kayması" hipotezi desteklenmedi.
- η_t > sabit η (6/6 seed), ama η̄ ~4 kat büyük; eşlenmiş kontrol gerekiyor.
- Çapa Loco'da 3/3 seed iyi (+0,016).

**Açık**
- Eşlenmiş η kontrolü (sabit η ≈ 4e-3).
- Rapor seed'leri ve test bölmesi.
- Oturumlar arası kayma protokolü (PHASES Faz 6).
- Faz 6 kodu commit edilmedi.

---

## 2026-11-02 — Faz 5 kapanışı

**Yaptıklarım**
- [x] Faz 5 kullanıcı kararıyla kapatıldı ve commit edildi. Testler: 125/125.
- [x] Bilimsel sonuç (KR-030 kapanış notu):
  - τ modülasyonunun yarattığı temsil patlaması, nedensel normalizasyon ve κ(τ) ile çözüldü (−4,68 → en kötü −0,078).
  - C, B+Z'den iyi değil. E sürümü: Indy +0,004 ± 0,013, Loco −0,012 ± 0,020.
  - Hipotez: donuk okuma, τ'nun yarattığı zamansal temsil kaymasını izleyemiyor. Faz 6'da sınanacak.

**Açık**
- `faz5_full.py` tam koşusu yapılmadı (rapor seed'leri 0–4, test). Faz 5 sonuçları ayar seed'leri ve val ile sınırlı.

---

## 2026-11-02 — Temsil kaymasının giderilmesi (KR-030)

**Yaptıklarım**
- [x] Teşhis: τ 100 → 300'de V ~2,5 kat büyüyor. τ = 100'de sessiz kalan nöronlar uzun τ'da ateşleyince donuk normda 70σ'yı aşan değerler üretiyor. KR-029'daki −4,68'in mekanizması bu.
- [x] ARCHITECTURE.md §3.2, §3.5, §8 ve §11 güncellendi. Koddaki değişiklikler:
  - `agent.freeze_norm_eval: false`: nedensel EMA, ağırlıklar donuk; duygu uzayının r normu da dahil.
  - `regulators.tau_ceiling: 1.2`: asimetrik tanh eşlemesi.
  - `reservoir.v_gain_exp: 0.5`: v = (τ_ref/τ)^0,5·V.
- [x] 7 yeni test; 121/121 geçiyor.
- [x] Adil karşılaştırma: B+Z de nedensel normla koşuldu. Indy + Loco, seed 100–102, 3 tur. C − B+Z:
  - E sürümü: Indy +0,004 ± 0,013, Loco −0,012 ± 0,020.
  - z sürümü: Indy −0,001 ± 0,007, Loco −0,049 ± 0,027.
  - En kötü fark −4,68'den −0,078'e indi. Val'de τ ≤ 118 ms.

- [x] `experiments/faz5_full.py` yazıldı. İş birimi (oturum, seed); sabit akış A/B/B+Z'yi besler, C kendi rezervuar kopyasıyla akar. r diske yazılmaz. `jobs/*.json` ile sürdürülebilir (`--run`). Çıktılar: R² val/test, eşli C − X farkları, C için τ kutularına göre K7, seyreltilmiş g/τ/η/β/E izleri.
  - Config: `faz5` bölümü (`configs/nhp_motor.yaml`): `[A, B, B+Z, C:perturb+EMR]`, 3 tur. σ = 0,1 config'e alındı (KR-025).
  - 4 yeni test; duman koşusu (Indy, seed 0–1, 2000 adım) sorunsuz: leak_free = True, resume 6 s.
  - Maliyet: adım başına ~20 ms (sabit akış 6,6 + C 13,4). Tam koşu 5,55 M adım ≈ 31 CPU-saat. 14 süreçle ~3–5 saat; en uzun iş (indy_20160622_01) tek başına ~2–3 saat. Fişe takılı ve uyku kapalı olmalı.

**Açık**
- Üç değişikliğin ayrı katkısı ölçülmedi.
- Tam koşu (rapor seed'leri 0–4, test) henüz başlatılmadı.
- Commit yapılmadı.

---

## 2026-11-01 — E/M/R duygu dinamiği

**Yaptıklarım**
- [x] Kullanıcının istediği E/M/R dinamiği (homeostatik + karşıt süreç): ARCHITECTURE.md §3.5/§6/§11, `AffectDynamics` (`src/affect.py`), config `affect.dynamics`, ajan bağlantısı. Kullanıcı onayıyla girdi merkezleme eklendi. Testler: 110/110.
- [x] Çevrimdışı tekrar (30 z izi): titreme ~4–5 kat azalıyor; doygunluk merkezlemesiz %51–80, merkezli %2–11; R zamanın %81–95'inde devrede (KR-026).
- [x] Kapalı döngü denemesi (Indy + Loco, seed 100, 3 tur, val): C:perturb Indy 0,330 / Loco 0,035; C:perturb+EMR 0,219 / 0,123; B+Z 0,322 / 0,139. İki sürüm de bir oturumda kötü (KR-027).

- [x] 3 ayar seed'i (100–102): z okuyan C Loco'da 3/3 seed kötü (−0,078 ± 0,020); E/M/R ile −0,024 ± 0,026; Indy'de belirgin fark yok. İyileştirme yok (KR-028).

- [x] Eşli perturbasyon (gölge A ajanı, §3.5, kullanıcı onayıyla) uygulandı ve testlendi. 3 seed: iyileşme yok; Loco'da felaket koşular (−4,68). Sebep: val'de regülatörler train'de nadir görülen τ rejimine geçiyor (örn. 74 → 314 ms), donuk okuma bunu kaldıramıyor (KR-029).

**Açık**
- Tam koşu bekletiliyor. Rejim kayması için seçenek kararı gerekiyor (KR-029: normalizasyonu çözmek / W_r'yi sınırlamak / val'de öğrenme açık).

---

## 2026-10-31 (akşam) — Faz 5 başlangıcı

**Yaptıklarım**
- [x] İş bölümü kararı: geliştirme ve küçük denemeler dizüstünde, uzun koşular (Faz 5 çıkışı, Faz 6) Colab Pro'da (CPU runtime, çıktılar Drive'a, devam ettirilebilir işler).
- [x] `src/affect.py`: `Regulators` (K4 sınırları, log-ölçekli τ, perturbasyon ve elle eşleme modları). Ayrıntılar: KR-023.
- [x] `src/agent.py`: konfigürasyon C, adım 2 (`regulators(z_{t−1})`) ve adım 10 (`regulators.update(q)`).
- [x] Testler: `tests/test_regulators.py` (12), ajan testlerine C bölümü, `test_invariants.py`'ye K4/K5. Tümü geçiyor.
- [x] Deneme koşusu `experiments/regulator_trial.py` (indy_20170131_02, 20 bin adım, seed 100, val): elle eşleme τ'yu 35–452 ms aralığında oynatıyor ama R² 0,214 → 0,166. Perturbasyon (σ = 0,01) neredeyse sabit kalıyor (τ 92–108). Bkz. KR-024.

- [x] Perturbasyon ızgarası (σ × M, 3 tur, val, seed 100): σ = 0,1, M = 500 en iyi aday (R² 0,330 vs B+Z 0,322; τ 39–412 ms, sınırda %0). Tek seed, fark gürültü düzeyinde. Bkz. KR-025.

**Açık**
- σ = 0,1 config'e alınacak mı (onay) + ikinci oturumda doğrulama. Regülatör girdisinin yavaşlatılması (§3.5 değişikliği, onay gerekir).
- Hız: C ajanı adım başına ~6–7 ms (rezervuar dahil, CPU). Tam oturum × 12 tur × 30 ajan Colab'da paralel koşulmalı.

---

## 2026-10-31

**Yaptıklarım**
- [x] Tam koşu bitti: `runs/20261030-101907_nhp_motor_full_seeds0-1-2-3-4`. 6 oturum × 5 seed × {A, B, B+Z}, 30/30 hücre dolu, NaN yok.
  - Süre: aşama 1 ≈ 8,5 saat; aşama 2 oturum başına ~19–20 saat (duvar saati). Bunun ~7,5 saati gece uykusu, bir kısmı da pilde yavaşlama. Uyanık ve fişe takılıyken tahmini ~10 saat.
- [x] `scripts/make_tables.py` → `table.md`; `experiments/affect_viz.py` → `affect_*.png` (oturum başına, seed 0) ve `affect_summary.json` (30 satır).

**Faz 3 sonucu** (test R², 5 seed ortalama ± std; A = B = B+Z)

| Oturum | Yerel kural | Ridge (üst sınır) |
|---|---|---|
| indy_20170131_02 | 0,446 ± 0,018 | 0,500 |
| indy_20160630_01 | 0,352 ± 0,009 | 0,368 |
| indy_20160622_01 | 0,529 ± 0,010 | 0,553 |
| loco_20170301_05 | 0,399 ± 0,014 | 0,355 |
| loco_20170215_02 | 0,344 ± 0,009 | 0,378 |
| loco_20170210_03 | 0,352 ± 0,007 | 0,319 |

- Maymun ortalaması: Indy 0,442 (NeuroBench SNN 0,593), Loco 0,365 (SNN 0,568). Taban çizgisinin ~0,15–0,20 altındayız (protokol farkı: 20 ms adım, 80/10/10 bölme, KR-018).
- **B − A = 0** her hücrede (K3: beyincik bu fazda kararı etkilemiyor). Beklenen sonuç.
- Yerel kural ridge'e çok yakın: Indy'de 0,02–0,05 geride, iki Loco oturumunda ridge'den **iyi**. Ayar seed'indeki ~0,14'lük fark (KR-020) rapor seed'lerinde büyük ölçüde kapanmış. Ridge'in λ'sı val'de seçiliyor; oturum içi kaymada çevrimiçi kural avantajlı olabilir (yorum, test edilmedi).
- K7: ateşleme oranı %1,1–1,6, tüm oturumlarda ✗ (KR-014, kural değişmedi).

**Faz 4 sonucu** (oturum ortalaması, 5 seed)

| Oturum | AUC iyi/kötü (z) | q baş MSE (ilk → son tur) | u | s |
|---|---|---|---|---|
| indy_20160622_01 | 0,788 ± 0,025 | 0,65 → 0,43 | 0,55 → 0,47 | 0,18 → 0,06 |
| indy_20160630_01 | 0,883 ± 0,015 | 0,63 → 0,48 | 0,73 → 0,67 | 0,25 → 0,11 |
| indy_20170131_02 | 0,753 ± 0,030 | 0,77 → 0,52 | 0,62 → 0,56 | 0,23 → 0,12 |
| loco_20170210_03 | 0,773 ± 0,026 | 0,56 → 0,42 | 0,72 → 0,67 | 0,28 → 0,11 |
| loco_20170215_02 | 0,752 ± 0,009 | 0,61 → 0,49 | 0,82 → 0,80 | 0,33 → 0,15 |
| loco_20170301_05 | 0,731 ± 0,032 | 0,70 → 0,53 | 0,80 → 0,78 | 0,34 → 0,17 |

- Çıkış kriterleri: (1) üç baş 30/30'da düşüyor ✓; (2) iyi/kötü ayrışıyor ✓ (AUC 0,73–0,88; 3B grafiklerde iyi anlar z'nin bir köşesinde toplanıyor); (3) `leak_free = True` ve `r2_B+Z == r2_B`, 30/30 ✓.
- **Dikkat:** başların val/test (donuk) MSE'si çoğu oturumda ≥ 1 (q: 0,7–2,0). Eğitimde öğrenilen baş ölçekleri test dağılımına tam taşınmıyor: donuk karar bloğunda δ büyüyor, hedef standartlaştırması da donuk. AUC sıralamaya baktığı için bundan etkilenmiyor ama Faz 5'te regülatörler bu başları okuyacak. Ele alınmalı.
- Loco'da u başı hâlâ zayıf (0,82 → 0,80), KR-021'deki açık konu sürüyor.

### Teknik borç (Faz 5 öncesi)
1. Commit: kod `16e50ce`'de commit'li; bu kapanış doküman değişiklikleri henüz commit edilmedi.
2. ~~`runs/.../r/*.npy` (~17 GB)~~ kontroller bittikten sonra kullanıcı onayıyla silindi (2026-10-31). Gerekirse aşama 1 ile yeniden üretilir.
3. ~~AUC kontrolleri~~ **Tamamlandı** (`experiments/affect_controls.py` → `affect_controls.json`). Aynı ölçüt (train'de q'ya doğrusal okuma, eval'de iyi/kötü AUC), 5 seed:

   | Oturum | Eğitilmiş z | Eğitilmemiş z | Yalnızca hız \|y\| | Tüm r (2000 boyut, tavan) |
   |---|---|---|---|---|
   | indy_20170131_02 | 0,753 ± 0,027 | 0,587 ± 0,033 | 0,698 ± 0,011 | 0,807 ± 0,010 |
   | indy_20160630_01 | 0,883 ± 0,013 | 0,607 ± 0,060 | 0,697 ± 0,005 | 0,923 ± 0,006 |
   | indy_20160622_01 | 0,788 ± 0,022 | 0,624 ± 0,027 | 0,635 ± 0,010 | 0,816 ± 0,017 |
   | loco_20170301_05 | 0,731 ± 0,029 | 0,558 ± 0,061 | 0,545 ± 0,014 | 0,778 ± 0,009 |
   | loco_20170215_02 | 0,752 ± 0,008 | 0,568 ± 0,041 | 0,644 ± 0,010 | 0,754 ± 0,010 |
   | loco_20170210_03 | 0,773 ± 0,023 | 0,575 ± 0,053 | 0,470 ± 0,019 | 0,771 ± 0,011 |

   - **Öğrenme gerçek:** eğitilmiş z, eğitilmemiş z'den her oturumda ~0,15–0,28 yüksek (ort. 0,78 vs 0,59).
   - **z sadece hız değil:** 6 oturumun 6'sında z, hızdan daha iyi ayırıyor. Fark en az `indy_20170131_02`'de (0,75 vs 0,70); bu oturumda kötü anların önemli kısmı hızlı hareketlerle örtüşüyor.
   - **3 boyut neredeyse yetiyor:** z, 2000 boyutlu r'nin tavanının %93–100'üne ulaşıyor.
   - "z + hız" birleşik okuması da hesaplandı. Ancak Loco'da hız özelliğinin train → test ölçeği kaydığı için bozuluyor (ör. 0,54), bu sütun yorum için güvenilir değil (json'da duruyor).
   - Eğitilmemiş z standartlaştırıcıyı iki geçişle yeniden yürütür (EMA sabit oranlı, ayrıntı script başında). Eğitilmiş z sütunu `affect_viz` AUC'siyle birebir aynı (çapraz kontrol).
4. Baş ölçeklerinin değerlendirmede taşınmaması (yukarıdaki not).
5. Hız: tüm oturum × seed ajanlarını (30) tek GPU batch'inde koşmak ve/veya Colab (Linux, düşük çekirdek başlatma yükü) üzerinde hız testi. Faz 5–6'dan önce. Uzun koşularda fişe takılı ve uyku kapalı olmalı.
6. KR-016 Faz 5'te geçersiz: regülatörler açılınca r öğrenmeye bağlı olur, iki aşamalı akış yeniden tasarlanmalı.

---

## 2026-10-29

**Yaptıklarım** (hiçbiri commit edilmedi)
- [x] **Veri:** 6 NeuroBench primat oturumu indirildi (≈4,3 GB, md5 doğrulandı, `data/primate/`). 20 ms ızgaraya işlenip önbelleğe alındı (`data/primate/cache/`, özet `summary.json`). Oturumlar 13–41 dk, 96/192 kanal, sorun yok.
- [x] **Ortak altyapı:**
  - `src/utils/online_norm.py`: `OnlineStandardizer`, `nlms_step`. Beyincik buna taşındı, davranış aynı.
  - `src/data/primate.py`: oturum yükleme, bölme, kanal başına ölçek.
  - `experiments/common.py`: rezervuar kurulumu, r toplama, parçalı ridge, R².
- [x] **Script'ler:**
  - `scripts/download_primate.py`, `scripts/preprocess_primate.py`
  - `scripts/diagnose_nhp.py` (teşhis tablosu), `scripts/tune_affect.py`, `scripts/make_tables.py`
  - `experiments/tune_nhp.py` (val ayarı, r önbelleği), `experiments/nhp_motor.py` (A, B, B+Z; `--workers` ile paralel), `experiments/affect_viz.py`
- [x] **Faz 3 kodu:** `src/decision.py` (`DecisionReadout`, normalize LMS) ve `src/agent.py` (adım 1–8, A/B; beyincik `a = ŷ` alır).
  - Testler: `tests/test_decision.py`, `tests/test_agent.py`, `test_invariants.py`'ye K3 kontrolleri.
- [x] **Faz 4 kodu:** `src/affect.py` (`AffectSpace`, `AffectTargets`) ve ajanın 9. adımı (`affect.enabled`, B+Z).
  - Testler: `tests/test_affect.py` ve sızıntı testi (duygu uzayı açıkken ŷ, x̂ ve ağırlıklar birebir aynı).
- [x] **Mimari:** ARCHITECTURE.md §3.1 (kanal başına ölçek), §3.4 (duygu uzayı ayrıntıları), §3.6 (karar bloğunda normalize LMS) ve §11 güncellendi.
- [x] **Dokümanlar:**
  - `docs/Desicion.md`: KR-001…020, geçmiş kararlar dahil.
  - `docs/Explain.md`: düzeltme notları + Faz 3 anlatımı (veri, ön işleme, normalize LMS, karar bloğu, ajan akışı, A/B/C, R²).
- [x] **Teşhis (KR-014):**
  - Rezervuar hız bilgisini taşıyor: ridge val R² 0,516, ham sayım 0,11, 10 adım geçmiş 0,51.
  - `c_in`/`L` değişince fark gürültü düzeyinde.
  - Ateşleme oranı gerçek veride %1,0 → **K7 ihlali**. Kullanıcı kararı: §3.2 kalibrasyon kuralı **değişmez**, ihlal raporlarda not edilir.
- [x] **Faz 3 ayarı (KR-020, yalnızca val):** `nhp.epochs: 12`, `decision.norm_rate: 0.0003`, η = 0,01 (K4 varsayılanı). Val R² 0,267 → 0,381.
- [x] **NeuroBench taban çizgisi** bulundu (arXiv 2304.04640, Tablo 4): SNN Indy 0,593, Loco 0,568; ANN 0,593 / 0,558.
- [x] Koşturulan testler geçti: 35 test (decision, agent, affect, cerebellum). Tam `pytest` en son değişikliklerden sonra koşturulmadı.

**Sonuç (şimdilik)**
- `indy_20170131_02`, ayar seed'i 100, ilk ayarsız koşu: A test R² 0,334 = B (beklenen, K3), ridge 0,493.
- Duygu uzayı (duman + ilk ayar): baş hataları düşüyor, z'den iyi/kötü ayrımı val AUC 0,74–0,84 (`lr_z` 0,03–0,1). Ancak z'nin %50–75'i doymuş (|z| > 0,99).

### Teknik borç / kalan işler (buradan devam)

**Hemen**
1. ~~Tam `uv run pytest`~~ 75/75 geçti.
2. ~~Duygu uzayı `lr_z` seçimi~~ **Tamamlandı (2026-10-30, KR-021).**
   - Seçilen: `lr_z = 0,03` + z ön-aktivasyonuna L2 cezası `λ = 0,03` (`affect.preact_l2`), baş ağırlıkları [1, 1, 1].
   - Val sonucu: AUC 0,807 (Indy) / 0,814 (Loco); doygunluk %60–68 → %1–3; üç baş da iki oturumda öğreniyor.
   - Yedek seçenek (`lr_z = 0,01`) gerekmedi.
   - Tam tablo Desicion.md KR-021'de. ARCHITECTURE.md §3.4'e `− λ·a` terimi eklendi.
   - Açık: Loco'da u başı zayıf (0,81 → 0,80).
   - Ayrıca: tam `pytest` 75/75 geçti (2026-10-29, L2 eklenmeden önce). Ardından affect/agent/invariants testleri yeniden geçti.
3. ~~KR-019 kaydı~~ yazıldı.

**Faz 3 tam koşu**
- **2026-10-30:** Batch'li okuma blokları (`src/batched.py`, KR-022) ve iki aşamalı tam koşu script'i (`experiments/nhp_full.py`) yazıldı.
  - Eşdeğerlik testi geçti, duman koşusu temiz.
  - Tam koşu başlatıldı: `uv run python experiments/nhp_full.py --config configs/nhp_motor.yaml --workers1 5 --workers2 6 --threads2 2`.
  - Yarıda kalırsa `--run runs/<klasör>` ile sürdürülür; biten işler atlanır.
4. `uv run python experiments/nhp_motor.py --config configs/nhp_motor.yaml --set "nhp.configs=[A, B, B+Z]" --workers 8`
   - 6 oturum × 5 seed; tahmini süre ~1–2 saat; RAM: büyük oturumda süreç başına ~3 GB.
   - Kontrol: tüm hücreler dolu, `leak_free: true`, NaN yok.
5. `uv run python scripts/make_tables.py --run runs/<koşu>` → `table.md`. A/B tablosu, B − A notu, NeuroBench karşılaştırması.
6. Faz 3 kapanışı: State.md sonuç girişi; PHASES.md Faz 3 ✅; CLAUDE.md "aktif faz" → Faz 4; Explain.md'ye sonuç yorumu.

**Faz 4 tam koşu**
7. `uv run python experiments/affect_viz.py --run runs/<aynı koşu>` → `affect_*.png`, `affect_summary.json`.
8. Faz 4 çıkış kriterleri:
   - (1) üç baş hatası düşüyor (5 seed);
   - (2) z'de iyi/kötü ayrışıyor (AUC + 3B grafik);
   - (3) görev performansı Faz 3 ile aynı (`r2_B+Z == r2_B`, `leak_free`).
9. Explain.md'ye duygu uzayı bölümü (z, üç baş, hedef standartlaştırma, W_z güncellemesi, sızıntı testi); Faz 4 kapanış dokümanları.

**Açık konular / bilinen sınırlar**
- K7 ihlali (KR-014): kural değişmedi, tablolarda ✗ olarak raporlanıyor. Faz 5'te τ değişince durum yeniden ölçülmeli.
- Yerel kural ile ridge arasında ~0,14 R² farkı var (NLMS yavaş yakınsıyor). Faz 3 kriteri bunu istemiyor ama Faz 6 için not.
- Hız / GPU: adım başına ~4 ms (CPU, 1 iş parçacığı); GPU tek deneme için daha yavaş (KR-015). GPU'yu doldurmak için bağımsız denemeleri batch'le: aynı r'yi okuyan ayar kombinasyonları `[K, …]` ve 5 seed'in rezervuarları `[5, N, N]` (`bmm`). Tahmini 5–20× hız (ölçülmedi). Faz 5–6 ablation'larından önce yapılmalı.
- Uzun koşularda bilgisayar uykuya geçmemeli: 2026-10-29 gecesi ayar koşusu ~10 saat askıda kaldı. İç döngüde `drive` her T adımı için ayrı hesaplanıyor (~0,4 ms/adım), iyileştirilebilir.
- r önbelleği (`data/primate/cache/r_*.pt`, ~0,3–1 GB/oturum) yalnızca ayar seed'i için. Rapor seed'leri her koşuda yeniden hesaplanıyor.
- Bölme farkı (KR-018): NeuroBench makalesi 75/25, biz kodun 80/10/10'u. Karşılaştırma birebir değil.
- Commit: Faz 2 sonrası hiçbir şey commit edilmedi. Faz 3 kapanınca commit kullanıcıya sorulacak.

---

## 2026-10-28

**Yaptıklarım**
- **Faz 2 kapatıldı.** `Cerebellum` (`src/cerebellum.py`), `configs/chaotic.yaml`, `experiments/cerebellum_test.py`, `tests/test_cerebellum.py`, K3/K6 arayüz testi.
- Normalize edilmiş çevrimiçi LMS kuralı ARCHITECTURE.md §3.3'e işlendi ve `base.yaml` varsayılanı oldu; §11'e kayıt eklendi.

**Etkisi**
- Ham delta kuralı ıraksıyordu çünkü okuma vektörü `[f, v]` büyük ölçekli (`‖φ‖² ~ 1e6`); veri setinin kendisi sorun değildi. Girişi nedensel standartlaştırıp adımı `‖φ̃‖²`'ye bölmek bunu çözdü.
- Aynı kural Faz 3'te karar bloğunda da gerekecek (ortak yardımcı önerilir).

**Sonuç** (`runs/20261025-230246_chaotic_seeds0-1-2-3-4`, Mackey-Glass τ=17, 5 seed)
- Normalize kural: MSE 0.00132 (NRMSE 0.161); "önceki değeri tekrarla" 0.00736 → oran **0.179** (kriter ≤ 0.5 ✔).
- Ridge üst sınırı: MSE 0.00124 (NRMSE 0.156) → fark **%6.8** (kriter < %20 ✔).
- Çevrimiçi ve dondurulmuş okuma neredeyse aynı. Ateşleme oranı ≈ %4.1 (K7 ✔). `plain` kural 5 seed'de de NaN.
- Testler: 46 geçti.

---

## 2026-10-25

**Yaptıklarım**
- `docs/` klasörü açıldı; `PHASES.md` buraya taşındı, bu `State.md` eklendi.

**Etkisi**
- Dokümanlar tek yerde; günlük ilerleme artık kayıt altında.

**Sonuç**
- Kod değişikliği yok.

---

## 2026-10-24

**Yaptıklarım**
- **Faz 0:** uv + `pyproject.toml` (CUDA torch), YAML config (base devralma, `--set`), seed sabitleme, koşu kaydı (config/seed/git/metrik), `experiments/run.py` iskeleti.
- **Faz 1:** `SpikeEncoder` (delta + termometre kanalları) ve `LIFReservoir` yazıldı; 37 test (K1, K2, K4, K7 dahil); bellek testi (`experiments/memory_test.py`, item/window hedefi, ideal doğrusal referans).
- Bellek testi ilk denemede başarısız çıktı → teşhislerle mimaride 8 değişiklik (ARCHITECTURE.md §11):
  - `(1−α)` kaldırıldı (uzun τ'da nöronlar ateşlemiyordu, %0.08).
  - k-WTA "tam k" → "eşiği geçenlerden en fazla k".
  - Okuma `r = [f, v]` (bellek zar potansiyelinde duruyor).
  - τ aralığı 5–100 → 20–2000 ms (ortam adımı 20 ms).
  - θ sabit → θ(τ) tablosu (geçici).
  - Nöron başına `τ_i = τ·m_i` ve `c_i` (tek τ'da bellek ~2 adımda bitiyordu).
  - Yumuşak sıfırlama `V ← V − θ`.
  - `γ_rec` rekürrent kazancı eklendi.
  - Kodlayıcıya seviye kanalları (delta kodlama seviye bilgisini taşımıyordu).

**Etkisi**
- Rezervuar artık τ ile gerçekten ölçeklenen bir belleğe sahip → regülatörün τ'yu ayarlaması (Faz 5) anlamlı hale geldi.
- Faz 1 çıkış kriteri sağlandı; Faz 2'ye geçilebilir.

**Sonuç** (`runs/20261024-131643_memory_seeds0-1-2-3-4`, 5 seed, window hedefi, `[f,v]` okuma)
- Bellek ufku (R² ≥ 0.1): τ = 20 / 100 / 500 / 2000 ms → **100 / 200 / 500 / 1000** adım (5 seed'de aynı).
- τ=2000 ufku / τ=20 ufku = **10×** (kriter ≥5×). Kısa pencerede en iyi τ=20, uzun pencerede τ=2000 ✔.
- Ateşleme oranı her τ'da ≈ %3.7 (K7 aralığında).
- Yalnız `f` okuması çok daha zayıf (ör. τ=2000, d=100: R² 0.07 vs `[f,v]` 0.80) → belleğin çoğu `v`'de.
- İdeal doğrusal referansa yakın (τ=2000, d=1000: 0.38 vs 0.42).
