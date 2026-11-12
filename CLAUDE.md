# CLAUDE.md

Bu dosya, Claude Code'un bu depoda nasıl çalışacağını tanımlar. Proje dili Türkçedir; yorumlar, dokümanlar ve commit mesajları Türkçe yazılır (kod tanımlayıcıları İngilizce).

## Proje özeti

Depoda iki mimari hattı var (ARCHITECTURE.md başındaki "Sürümler" notu):

- **Spike v2 (aktif):** Çevrimdışı (surrogate gradyan / BPTT) eğitilen, 500 hızlı + 500 yavaş nöronlu zamansal omurga (`plif` | `spiking_gru` | `tau_gru`). Üstünde yalnızca **yerel kurallarla** çevrimiçi uyum sağlayan bloklar: **beyincik** (yumuşatılmış sonraki girdiyi efference copy ile tahmin eder), **duygu uzayı** (beklenmedik sürpriz a(t) → g_fast, g_slow, η), **5 başlıklı çapalı okuma** (görev çıktısı).
  Hedef: NeuroBench Primate Reaching test R² ≥ 0,65–0,70. Araştırma sorusu: sürpriz modülasyonlu η ve alt-popülasyon kapılaması, günler arası kaymada ve elektrot kaybında, eşlenmiş sabit η'dan (v2-B) ve karıştırılmış η'dan (v2-C) iyi mi? (v2-A … v2-E)
- **Spike v1 (kayıt):** Sabit LIF rezervuarı + yerel okumalar + z → g, τ, η, β regülatörleri; A / B / C ablation'ı. Faz 0–6a tamam/askıda; sonuçlar KR-001–KR-033. **Kod master'da yok; `v1-final` git tag'inde (KR-049).**

## Kaynak dokümanlar (önce bunları oku)

- [ARCHITECTURE.md](ARCHITECTURE.md) — **v2 kuralları §12 (KV1–KV8)**, v1 kuralları §2 (K1–K8), denklemler, arayüzler, hiperparametreler. Kural değişecekse önce bu dosya güncellenir, sonra kod.
- [docs/PhasesV2.md](docs/PhasesV2.md) — **v2 fazları V0–V7 (aktif)**. Bir fazın çıkış kriteri sağlanmadan sonraki faza geçilmez.
- [docs/spike_v2_mimari_spesifikasyon.md](docs/spike_v2_mimari_spesifikasyon.md) — kullanıcının v2 spesifikasyonu ve referans PyTorch kodu. Kodu birebir kopyalamadan önce ARCHITECTURE.md §12.10'daki açık noktalara bak.
- [docs/PHASES.md](docs/PHASES.md) — v1 fazları (Faz 0–6), kayıt.
- `docs/Decision.md` (KR kayıtları), `docs/State.md` (günlük), `docs/Explain.md` (açıklamalar).

## Mevcut durum

- **Faz V8 (KR-051, 2026-11-12): kod hazır, koşular Colab'da bekliyor** (Dryad yükleyicisi denetimden sonra) (plan `docs/PlanV8.md`, kurallar §12.21). Çoklu-gün omurgası (M: ÖN + Day-0) ile tek-gün omurgası (S) V7 sisteminin tamamıyla karşılaştırılır. Hayvan başına oturum rolleri ÖN < Day-0 < AYAR / TEST / EK, kodda zorunlu (`check_roles`); ÖN Day-0 istatistikleriyle kodlanır. Loco V8 Day-0 = 2017-02-14. İkinci veri seti Dryad xd2547dkt, ayarsız. Config `configs/v2_v8.yaml`, `configs/v2_pretrain_v8.yaml`. V8, V7'nin ayar sonuçlarını kullanır.
- **Faz V7 (KR-049, 2026-11-09): kod hazır, koşular bekliyor.** Hakem raporuna cevap: BOCPD sensörü (arousal = P(rejim ≠ Day-0), §12.16), ölü kanal dedektörü (§12.17), Degenhart hizalaması (§12.18; yetmezse NoMAD), sönümlü ReFIT niyet etiketi (gecikmesiz, §12.15), ön-eğitimde kanal bozulması augmentation'ı (§12.19). İstatistik birimi **oturum** (seed'ler ortalanır; Holm birincil + tüm aile; toparlanma Kaplan–Meier). **Ayar yalnızca test dışı ayar oturumlarında** (`tune_primates`, `--protocol tune`); test oturumlarında ayar yok. Deney `experiments/v2_v7.py` (`configs/v2_v7.yaml`), ön-eğitim `configs/v2_pretrain_v7.yaml`, özet sonuçlar `results/` (git'te).
- V0–V6 tamamlandı (2026-11-08, KR-048). v2-D = kalıcı sürpriz sensörü (yeniden ölçekleme kapısı, KR-046) + sensör arousal'ıyla okuma gazı ve çapa gevşetme (KR-047). Test Day-k R²: D 0,279, C 0,277, P 0,272, R5 0,251, B 0,039, A −0,066; D > B ve D > P destekleniyor, D > C desteklenmiyor; elektrot kaybında R5 en iyi. Ana mod plif; Loco KV7 ihlali raporlanıyor. Eğitim adımı CUDA Graph ile (KR-038); uzun koşular seed / aşama başına ayrı süreçte, GPU'da tek süreç (KR-043). Deneyler `experiments/v2_v6.py` (`configs/v2_v6.yaml`), tablolar `scripts/make_tables_v2.py`. (Faz bitince bu satırı güncelle.)
- v1 son durumu: Faz 3–4 test R² Indy 0,442 / Loco 0,365. Faz 6a: kapılı η eşlenmiş η'yı geçmedi (KR-031, KR-032). KR-033: GRU tavanı Indy 0,736 / Loco 0,685 → kayıp doğrusal okunan temsilde; v2'ye geçişin gerekçesi bu. v1 Faz 6 kayma deneyi askıda.
- Açık v2 kararları (ARCHITECTURE.md §12.10): beyincik girdisinin kapılanıp kapılanmaması (V5). Kapananlar: bölme 80/10/10 ve `src/v2/` yerleşimi (KR-035), sıfırlama gecikmesi ve epoch bütçesi (KR-038), bias donuk (KR-040).
- **Çevrimiçi etiket 500 ms gecikmeli (KR-041):** anında etiket 4 ms'de dejenere (y_{t−1} kopyası R² 0,999). Uyum tablolarında gecikmeli kopyalama tabanı verilir. V3 ve V4 birlikte koşuyor (`runs/chain_<primat>.sh`: ayar → V4 val → V3 rapor).
- ARCHITECTURE.md kural değişikliği gerekiyorsa önce kullanıcıya teşhis tablosu getir; dosyayı kullanıcı onayı olmadan değiştirme.
- Faz sonuçları birlikte değerlendirildikten sonra dokümanlara işlenir: PhasesV2.md (`> Sonuç:`), Decision.md (KR), State.md (günlük), gerekirse ARCHITECTURE.md §12 ve §11.
- Depo kökü (`Spike/`) ARCHITECTURE.md §4'teki `snn_affect/` klasörüne karşılık gelir; ayrıca bir `snn_affect/` alt klasörü açılmaz.

## Kesin kurallar — v2 kodu (ARCHITECTURE.md §12.2)

1. **KV1:** Omurga (ağırlıklar, τ_mem, τ_syn) yalnızca çevrimdışı aşamalarda gradyanla eğitilir. Çevrimiçi akışta `requires_grad=False`; W_rec, τ, bias değişmez. Çevrimiçi değişebilenler: homeostatik eşik ϑ ve (plif, `plasticity.enabled`) W_in — yalnız §12.13'teki duygu-kapılı, çapalı, norm topu sınırlı yerel kuralla; akış sonunda W_in,0 geri yüklenir (KR-044).
2. **KV2:** Çevrimiçi öğrenme yereldir: gradyan / optimizer yok; yalnızca beyincik NLMS'i, çapalı okuma NLMS'i, homeostaz ve W_in üç faktörlü kuralı.
3. **KV3:** e_v (hız hatası) çevrimiçi okuma başlıklarını ve W_in'i (okuma ağırlıklarıyla geri besleme), e_cer (tahmin hatası) yalnızca W_cer'i eğitir. Beyincik kaybı omurgaya gradyan vermez. Duygu uzayı yalnızca okur (e_cer, girdi sapması, başlık uyuşmazlığı, gecikmeli e_v; KR-047). V7'de `label_source: refit` iken okuma e_int ile (sönümlü ReFIT, gecikmesiz, y kullanılmaz; §12.15) eğitilir.
4. **KV4:** Regülatörler yalnızca g_fast, g_slow ∈ [1−α_gate, 1+α_gate], η(t) ∈ [η_0, η_0(1+α_η)] (dinamik duyguda [0, η_0(1+α_η)]), η_in(t) ∈ [0, η_in,0], λ_norm(t) ∈ [λ_calm, λ_surp] (KR-045, KR-046), α_anchor(t) ∈ [0, α_0+α_b] ve konsolidasyon adımı c(t) ∈ [0, 1] (KR-047) üretir. τ ve W_rec'e dokunmaz.
5. **KV5:** `regulators.enabled: false` her zaman çalışmalı (g = 1, η ve η_in sabit).
6. **KV6:** Akışta durum sıfırlanmaz; beyincik x(t+1)/x̄(t+1)'i göremez; prequential (önce skorla, sonra güncelle); kodlayıcı istatistikleri yalnızca train'den — istisnalar etiketsiz ve nedenseldir: §12.14 kanal yeniden ölçeklemesi (KR-045), V7'de BOCPD sensörü, ölü kanal dedektörü ve Degenhart hizalaması (§12.16–§12.18; referanslar Day-0 train'den, KR-049).
7. **KV7:** k-WTA yok; hedef oran ρ = 0,05, homeostatik bant [0,02; 0,08]. Oran alt-popülasyon başına loglanır, bant dışı hata. `tau_gru` muaf.
8. **KV8:** Kodda sabit hiperparametre yok (spesifikasyon kodundaki sabitler dahil) — her şey YAML'dan. Config + seed + git hash + metrik + checkpoint hash'leri kaydedilir. Sonuçlar ≥5 seed, ortalama ± std.
9. Test seti üzerinde hiperparametre/regülatör ayarı yapılmaz. Ayar: seed 100–102 ve val; V7'den itibaren yalnız test dışı ayar oturumlarında (`tune_primates`; çakışma `check_tune_disjoint` ile hata, KR-049). İstatistik birimi oturumdur (seed'ler ortalanır).

## v1 kodu

v1 kodu ve testleri `v1-final` tag'indedir (`git checkout v1-final`); master'da v1 dosyası yoktur. ARCHITECTURE.md §1–§10 (K1–K8) kayıt olarak durur ve yalnızca o tag'deki kodu bağlar.

## Kodlama kuralları

- Python 3.11+, PyTorch. snnTorch yalnızca yardımcı fonksiyonlar için; omurga/rezervuar kendi kodumuz.
- Tüm fonksiyonlarda tip ipuçları ve tensör şekli yorumları zorunlu: `r = ...  # [B, N]`.
- Her modül `forward(...)`, `update(...)`, `reset_state()` sağlar (v2 omurgasında çevrimiçi `update` yoktur).
- Yeni modül ancak testi yazıldıktan sonra akışa (`src/v2/online_v5.py`) ya da modele (`src/v2/model.py`) bağlanır.
- `tests/v2/` testlerinin hepsi her commit öncesi geçmeli.
- Fazın gerektirmediği modülü önceden yazma; PhasesV2.md'ye sadık kal.

## Ortam ve komutlar

- Windows 11, PowerShell. `python` komutu PATH'te değil → `uv` (veya `py -3.11`) kullan.
- GPU: RTX 4060 Laptop (CUDA). Kod `device` seçimini config'ten almalı, CPU'da da çalışmalı. v2 çevrimdışı eğitimi GPU'da koşar; uzun koşularda fişe takılı + uyku kapalı (KR-022).
- Paket/ortam yönetimi: **uv** + `pyproject.toml`, sanal ortam `.venv/`.
- **Kurulum işlemlerini (ortam oluşturma, paket ekleme/kaldırma, `uv sync`) kullanıcı yönetir.** Claude bu komutları kendiliğinden çalıştırmaz; gerekirse komutu önerir.

```bash
uv run pytest                                                                     # testler (tests/v2)
uv run python scripts/download_primate.py --config configs/v2_base.yaml --only tune   # V7 ayar oturumları
uv run python experiments/v2_v7.py --config configs/v2_v7.yaml --runs <koşular> --protocol tune --split val
```

## Çıktılar

- Koşu logları: `runs/<tarih-saat>_<config-adı>_seed<seed>/` altında `config.yaml`, `meta.json` (seed, git hash, zaman), `metrics.jsonl`. `runs/` git'e eklenmez.
- v2 checkpoint'leri (ön-eğitim, Day-0, W_0) koşu klasöründe; yolları ve hash'leri `meta.json`'a yazılır.
