# CLAUDE.md

Bu dosya, Claude Code'un bu depoda nasıl çalışacağını tanımlar. Proje dili Türkçedir; yorumlar, dokümanlar ve commit mesajları Türkçe yazılır (kod tanımlayıcıları İngilizce).

## Proje özeti

Depoda iki mimari hattı var (ARCHITECTURE.md başındaki "Sürümler" notu):

- **Spike v2 (aktif):** Çevrimdışı (surrogate gradyan / BPTT) eğitilen, 500 hızlı + 500 yavaş nöronlu zamansal omurga (`plif` | `spiking_gru` | `tau_gru`). Üstünde yalnızca **yerel kurallarla** çevrimiçi uyum sağlayan bloklar: **beyincik** (yumuşatılmış sonraki girdiyi efference copy ile tahmin eder), **duygu uzayı** (beklenmedik sürpriz a(t) → g_fast, g_slow, η), **5 başlıklı çapalı okuma** (görev çıktısı).
  Hedef: NeuroBench Primate Reaching test R² ≥ 0,65–0,70. Araştırma sorusu: sürpriz modülasyonlu η ve alt-popülasyon kapılaması, günler arası kaymada ve elektrot kaybında, eşlenmiş sabit η'dan (v2-B) ve karıştırılmış η'dan (v2-C) iyi mi? (v2-A … v2-E)
- **Spike v1 (kayıt):** Sabit LIF rezervuarı + yerel okumalar + z → g, τ, η, β regülatörleri; A / B / C ablation'ı. Faz 0–6a tamam/askıda; sonuçlar KR-001–KR-033.

## Kaynak dokümanlar (önce bunları oku)

- [ARCHITECTURE.md](ARCHITECTURE.md) — **v2 kuralları §12 (KV1–KV8)**, v1 kuralları §2 (K1–K8), denklemler, arayüzler, hiperparametreler. Kural değişecekse önce bu dosya güncellenir, sonra kod.
- [docs/PhasesV2.md](docs/PhasesV2.md) — **v2 fazları V0–V6 (aktif)**. Bir fazın çıkış kriteri sağlanmadan sonraki faza geçilmez.
- [docs/spike_v2_mimari_spesifikasyon.md](docs/spike_v2_mimari_spesifikasyon.md) — kullanıcının v2 spesifikasyonu ve referans PyTorch kodu. Kodu birebir kopyalamadan önce ARCHITECTURE.md §12.10'daki açık noktalara bak.
- [docs/PHASES.md](docs/PHASES.md) — v1 fazları (Faz 0–6), kayıt.
- `docs/Desicion.md` (KR kayıtları), `docs/State.md` (günlük), `docs/Explain.md` (açıklamalar).

## Mevcut durum

- **Aktif hat: Spike v2. Aktif faz: V1 — kodlayıcı v2 (V0 tamamlandı 2026-11-04).** v2 veri katmanı `src/v2/data.py`; protokol KR-035 (80/10/10, Δt 4 ms). 4 ms tavanı GRU Indy 0,708 / Loco 0,616 (alt sınır, KR-037). Epoch bütçesi ölçümü V2'de (KR-036). (Faz bitince bu satırı güncelle.)
- v1 son durumu: Faz 3–4 test R² Indy 0,442 / Loco 0,365. Faz 6a: kapılı η eşlenmiş η'yı geçmedi (KR-031, KR-032). KR-033: GRU tavanı Indy 0,736 / Loco 0,685 → kayıp doğrusal okunan temsilde; v2'ye geçişin gerekçesi bu. v1 Faz 6 kayma deneyi askıda.
- Açık v2 kararları (ARCHITECTURE.md §12.10): plif sıfırlama gecikmesi, beyincik girdisinin kapılanıp kapılanmaması, okuma bias'ının çevrimiçi güncellenmesi, epoch bütçesi (V2, KR-036). Kapananlar: bölme 80/10/10 ve `src/v2/` yerleşimi (KR-035).
- ARCHITECTURE.md kural değişikliği gerekiyorsa önce kullanıcıya teşhis tablosu getir; dosyayı kullanıcı onayı olmadan değiştirme.
- Faz sonuçları birlikte değerlendirildikten sonra dokümanlara işlenir: PhasesV2.md (`> Sonuç:`), Desicion.md (KR), State.md (günlük), gerekirse ARCHITECTURE.md §12 ve §11.
- v1 notu: θ(τ) kalibrasyonu ~1 dk sürer (CPU). v1 testlerinde `tests/conftest.py`'deki oturum fixture'ı `calibrated_reservoir` kullanılır; kalibrasyon gerekmeyen testlerde `from_config(..., calibrate=False)`.
- Depo kökü (`Spike/`) ARCHITECTURE.md §4'teki `snn_affect/` klasörüne karşılık gelir; ayrıca bir `snn_affect/` alt klasörü açılmaz.

## Kesin kurallar — v2 kodu (ARCHITECTURE.md §12.2)

1. **KV1:** Omurga (ağırlıklar, τ_mem, τ_syn) yalnızca çevrimdışı aşamalarda gradyanla eğitilir. Çevrimiçi akışta `requires_grad=False`, hiçbir güncelleme dokunmaz, τ değişmez. Çevrimiçi değişebilen tek omurga durumu homeostatik eşik ϑ (etiketsiz, yerel, sınırlı).
2. **KV2:** Çevrimiçi öğrenme yereldir: gradyan / optimizer yok; yalnızca beyincik NLMS'i, çapalı okuma NLMS'i ve homeostaz.
3. **KV3:** e_v (hız hatası) çevrimiçi yalnızca okuma başlıklarını, e_cer (tahmin hatası) yalnızca W_cer'i eğitir. Beyincik kaybı omurgaya gradyan vermez. Duygu uzayı yalnızca okur.
4. **KV4:** Regülatörler yalnızca g_fast, g_slow ∈ [1−α_gate, 1+α_gate] ve η(t) ∈ [η_0, η_0(1+α_η)] üretir. τ ve omurgaya dokunmaz.
5. **KV5:** `regulators.enabled: false` her zaman çalışmalı (g = 1, η sabit).
6. **KV6:** Akışta durum sıfırlanmaz; beyincik x(t+1)/x̄(t+1)'i göremez; prequential (önce skorla, sonra güncelle); kodlayıcı istatistikleri yalnızca train'den.
7. **KV7:** k-WTA yok; hedef oran ρ = 0,05, homeostatik bant [0,02; 0,08]. Oran alt-popülasyon başına loglanır, bant dışı hata. `tau_gru` muaf.
8. **KV8:** Kodda sabit hiperparametre yok (spesifikasyon kodundaki sabitler dahil) — her şey YAML'dan. Config + seed + git hash + metrik + checkpoint hash'leri kaydedilir. Sonuçlar ≥5 seed, ortalama ± std.
9. Test seti üzerinde hiperparametre/regülatör ayarı yapılmaz. Ayar: seed 100–102 ve val.

## Kesin kurallar — v1 kodu (ARCHITECTURE.md §2–§3, değişmeden geçerli)

K1 (sabit rezervuar), K3, K4 (g, τ, η, β sınırları), K5, K6, K7 (k-WTA ≤ ⌈ρN⌉), K8; kodlayıcı delta + termometre (§3.1); rezervuar `τ_i = τ·m_i`, `c_i`, yumuşak sıfırlama, θ(τ) tablosu, `r = [f, v]` (§3.2). v1 dosyalarını (`src/encoder.py`, `reservoir.py`, `agent.py` …) değiştirirken bunlar geçerli; v1 testleri kırılmamalı.

## Kodlama kuralları

- Python 3.11+, PyTorch. snnTorch yalnızca yardımcı fonksiyonlar için; omurga/rezervuar kendi kodumuz.
- Tüm fonksiyonlarda tip ipuçları ve tensör şekli yorumları zorunlu: `r = ...  # [B, N]`.
- Her modül `forward(...)`, `update(...)`, `reset_state()` sağlar (v1 rezervuarında ve v2 omurgasında çevrimiçi `update` yoktur).
- Yeni modül ancak testi yazıldıktan sonra ana modele (v1: `src/agent.py`, v2: `src/v2/model.py` önerisi) bağlanır.
- `tests/test_invariants.py` (v1) ve v2 değişmez testleri her commit öncesi geçmeli.
- Fazın gerektirmediği modülü önceden yazma; PhasesV2.md'ye sadık kal.

## Ortam ve komutlar

- Windows 11, PowerShell. `python` komutu PATH'te değil → `uv` (veya `py -3.11`) kullan.
- GPU: RTX 4060 Laptop (CUDA). Kod `device` seçimini config'ten almalı, CPU'da da çalışmalı. v2 çevrimdışı eğitimi GPU'da koşar; uzun koşularda fişe takılı + uyku kapalı (KR-022).
- Paket/ortam yönetimi: **uv** + `pyproject.toml`, sanal ortam `.venv/`.
- **Kurulum işlemlerini (ortam oluşturma, paket ekleme/kaldırma, `uv sync`) kullanıcı yönetir.** Claude bu komutları kendiliğinden çalıştırmaz; gerekirse komutu önerir.

```bash
uv run python experiments/run.py --config configs/base.yaml   # deney koşusu
uv run pytest                                                  # testler
uv run pytest tests/test_invariants.py                         # v1 K1–K7 kontrolü
```

## Çıktılar

- Koşu logları: `runs/<tarih-saat>_<config-adı>_seed<seed>/` altında `config.yaml`, `meta.json` (seed, git hash, zaman), `metrics.jsonl`. `runs/` git'e eklenmez.
- v2 checkpoint'leri (ön-eğitim, Day-0, W_0) koşu klasöründe; yolları ve hash'leri `meta.json`'a yazılır.
