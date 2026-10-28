# CLAUDE.md

Bu dosya, Claude Code'un bu depoda nasıl çalışacağını tanımlar. Proje dili Türkçedir; yorumlar, dokümanlar ve commit mesajları Türkçe yazılır (kod tanımlayıcıları İngilizce).

## Proje özeti

Sabit (eğitilmeyen) bir LIF spiking rezervuarı üzerine, yalnızca **yerel öğrenme kurallarıyla** eğitilen üç okuma bloğu: **beyincik** (sonraki girdiyi tahmin eder), **duygu uzayı + regülatörler** (z vektörü → g, τ, η, β), **karar bloğu** (görev çıktısı).
Araştırma sorusu: duygu uzayı ve regülatörler, veri kaydığında (oturumlar arası) uyuma yardım ediyor mu? (A / B / C ablation'ı)

## Kaynak dokümanlar (önce bunları oku)

- [ARCHITECTURE.md](ARCHITECTURE.md) — **değişmez kurallar (K1–K8)**, denklemler, modül arayüzleri, varsayılan hiperparametreler. Kural değişecekse önce bu dosya güncellenir, sonra kod.
- [docs/PHASES.md](docs/PHASES.md) — Faz 0–6. Bir fazın çıkış kriteri sağlanmadan sonraki faza geçilmez.

## Mevcut durum

- **Aktif faz: Faz 3 — Karar bloğu ve ilk gerçek görev (başlamadı).** Faz 0–1 tamam. Faz 2 çıkış kriteri 2026-10-28'de sağlandı (46 test; Mackey-Glass tek adım MSE = tekrar tabanının %17.9'u, ridge'e göre fark %6.8, `runs/20261025-230246_chaotic_seeds0-1-2-3-4`). Beyincik varsayılan kuralı normalize LMS (ARCHITECTURE.md §3.3). Faz 3 için NeuroBench primat `.mat` dosyaları indirilmedi. (Faz bitince bu satırı güncelle.)
- Açık kararlar: ARCHITECTURE.md §3.2 θ(τ) "geçici" durumu ve §7 `γ_rec` değeri (ayarda 1/3/10 arası fark gürültü düzeyinde, config'te 1.0).
- ARCHITECTURE.md kural değişikliği gerekiyorsa önce kullanıcıya teşhis tablosu getir; dosyayı kullanıcı onayı olmadan değiştirme.
- θ(τ) kalibrasyonu ~1 dk sürer (CPU). Testlerde `tests/conftest.py`'deki oturum fixture'ı `calibrated_reservoir` kullanılır; kalibrasyon gerekmeyen testlerde `from_config(..., calibrate=False)`.
- Depo kökü (`Spike/`) ARCHITECTURE.md §4'teki `snn_affect/` klasörüne karşılık gelir; ayrıca bir `snn_affect/` alt klasörü açılmaz.

## Kesin kurallar (kod yazarken asla ihlal etme)

1. **K1:** `W_in`, `W_rec` sabit seed ile üretilir, `requires_grad=False`, hiçbir `update` onlara dokunmaz. Rezervuarda değişen tek şey regülatörden gelen `g` ve `τ`.
2. **K2:** `loss.backward()`, `torch.autograd`, optimizer yok. Tüm güncellemeler `torch.no_grad()` içinde, açık formüllerle ve yerel.
3. **K3:** δ (sonuç hatası) yalnızca karar bloğunu, e (tahmin hatası) yalnızca beyinciği eğitir. Duygu uzayı bunları yalnızca okur.
4. **K4:** Regülatörler yalnızca g ∈ [0.5, 2], τ ∈ [20, 2000] ms (log-ölçek, varsayılan 100), η ∈ [1e-4, 1e-1], β ∈ [0.1, 10] üretir. Sigmoid → doğrusal ölçek.
5. **K5:** `regulators.enabled: false` her zaman çalışmalı (değerler varsayılanda sabit).
6. **K6:** Rezervuar durumu dizi içinde sıfırlanmaz; beyincik x̂(t+1) üretirken x(t+1)'i göremez.
7. **K7:** k-WTA: eşiği geçenlerden (`V > θ(τ)`) **en fazla** `k = ⌈ρ·N⌉`. Ortalama ateşleme oranı her τ için loglanır, `[ρ/2, 2ρ]` dışı hata.
8. **Kodlayıcı (§3.1):** `SpikeEncoder` = delta (2·D_in) + seviye/termometre (L·D_in) kanalları, `C = (2+L)·D_in`. Seviye aralığı yalnızca eğitim verisinden.
9. **Rezervuar (§3.2):** nöron başına `τ_i = τ·m_i` ve göreli denge `c_i`; `V_i = α_i·V_i + c_i·I_i` (ortak `(1−α)` yok); yumuşak sıfırlama `V ← V − θ`; `γ_rec` rekürrent kazanç; θ(τ) bir kez kalibre edilen sabit tablo; okuma vektörü her zaman `r = [f, v]`, şekil `[B, 2N]`. `LIFReservoir.from_config(cfg, D_in, seed)` ham girdi boyutunu alır.
10. **K8:** Kodda sabit hiperparametre yok — her şey YAML config'ten gelir. Config + seed + git hash + metrikler her koşuda kaydedilir. Sonuçlar ≥5 seed, ortalama ± std.
11. Test seti üzerinde hiperparametre/regülatör ayarı yapılmaz.

## Kodlama kuralları

- Python 3.11+, PyTorch. snnTorch yalnızca yardımcı fonksiyonlar için; rezervuar kendi kodumuz.
- Tüm fonksiyonlarda tip ipuçları ve tensör şekli yorumları zorunlu: `r = ...  # [B, N]`.
- Her modül `forward(...)`, `update(...)`, `reset_state()` sağlar (rezervuarda `update` yoktur / yasaktır).
- Yeni modül ancak testi yazıldıktan sonra `src/agent.py`'ye bağlanır.
- `tests/test_invariants.py` her commit öncesi geçmeli.
- Fazın gerektirmediği modülü önceden yazma; fazlara sadık kal.

## Ortam ve komutlar

- Windows 11, PowerShell. `python` komutu PATH'te değil → `uv` (veya `py -3.11`) kullan.
- GPU: RTX 4060 Laptop (CUDA). Kod `device` seçimini config'ten almalı, CPU'da da çalışmalı.
- Paket/ortam yönetimi: **uv** + `pyproject.toml`, sanal ortam `.venv/`.
- **Kurulum işlemlerini (ortam oluşturma, paket ekleme/kaldırma, `uv sync`) kullanıcı yönetir.** Claude bu komutları kendiliğinden çalıştırmaz; gerekirse komutu önerir.

```bash
uv run python experiments/run.py --config configs/base.yaml   # deney koşusu
uv run pytest                                                  # testler
uv run pytest tests/test_invariants.py                         # K1–K7 kontrolü
```

## Çıktılar

- Koşu logları: `runs/<tarih-saat>_<config-adı>_seed<seed>/` altında `config.yaml`, `meta.json` (seed, git hash, zaman), `metrics.jsonl`. `runs/` git'e eklenmez.
