# Spike v2: Biyolojik Esinli Nöromodüle SNN / τ-GRU Motor Dekodlama Mimari Spesifikasyonu

Bu döküman, `https://github.com/yunussfr/Spike` deposundaki mevcut mimariyi (**Spike v1**) **NeuroBench Primate Reaching (Indy & Loco)** görevinde **$R^2 \ge 0.65 - 0.70$** bandına taşımak ve biyolojik açıdan savunulabilir, yüksek yayın değerine sahip bir **"Hibrit Ön-Eğitimli Omurga + Nöromodüle Çevrimiçi Yerel Uyum"** çerçevesine (**Spike v2**) dönüştürmek için hazırlanmış teknik uygulama rehberidir. Yapay zeka kodlama asistanına doğrudan bağlam (context/prompt) olarak verilebilir.

## 1. Mevcut Durum (Spike v1) Teşhisi ve v2 Mimari Kararları

| Bileşen / Sorun | Spike v1 (Mevcut Durum) | Teşhis Edilen Darboğaz | Spike v2 (Yeni Mimari Karar) |
| :--- | :--- | :--- | :--- |
| **Zaman Adımı ($\Delta t$)** | $\Delta t = 20\text{ ms}$ | Hızlı motor korteks (M1) dinamiklerini bulanıklaştırıyor. | **$\Delta t = 4\text{ ms}$** (250 Hz standart NeuroBench) veya hızlı eğitim için **$\Delta t = 8\text{ ms}$**. |
| **Giriş Kodlama (`SpikeEncoder`)** | Min-Max normalizasyon + Poisson/Rate kodlama | Nadir patlamalar (burst) yüzünden normal adımlar $0.05$ altına sıkışıyor; ateşleme oranı %1–2'de kalıyor. | **Anscombe Karekök Dönüşümü + Kanal Bazlı Z-Skor + Yumuşak Kırpma (`3 * tanh(z/3)`)**. M1 ve S1 genlik farkını mimariyi bölmeden veri katmanında absorbe eder. |
| **Omurga (`LIFReservoir`)** | $N=1000$ sabit rastgele ağırlıklı LIF, tekil $\tau_0 = 100\text{ ms}$ | Görev-dışı gürültüyü büyütüyor; çevrimiçi $\tau[t]$ değişimi nöron fazlarını bozup Loco'da $R^2$ düşürüyor. | **$N=1000$ (500 Hızlı + 500 Yavaş)** birleşik omurga (`mode="plif" \| "spiking_gru" \| "tau_gru"`). $\tau_i$ değerleri çevrimiçi sabit tutulur, ön-eğitimde öğrenilir. |
| **Ateşleme Rejimi** | %1–2 ateşleme (%98–99 sessizlik) | Okuma vektörü $r[t]$ aşırı seyrek, shot-noise yüksek. | **Dalgalanma Güdümlü Başlatma (Fluctuation-Driven Init)** + **Homeostatik Kayıp** + **Yerel Eşik Adaptasyonu ($\vartheta_i[t]$)** ile %5 ateşleme kilidi. |
| **Serebellum (`CerebellumBlock`)** | Ham $x[t+1]$ tahmini ($e[t] = x[t+1] - \hat{x}[t+1]$) | Ham spike treni %80 Poisson gürültüsü içerdiği için sürpriz sinyali gürültüyü izliyor. | **Üstel Yumuşatılmış Popülasyon Hedefi ($\bar{x}[t+1]$)** + **Efference Copy ($\hat{v}[t-1]$)** girişi. |
| **Duygu / Regülatör (`AffectSpace`)** | Anlık hata $\|e[t]\|^2 \to \tau[t], \eta[t]$ | Yüksek hızlı hareketlerdeki beklenen gürültüde $\eta[t]$ artıp ağırlıkları bozuyor. | **Yu & Dayan (2005) Beklenmedik Sürpriz Oranı** ($\text{EMA}_{\text{fast}} / \text{EMA}_{\text{slow}}$) + **Hızlı/Yavaş Alt-Popülasyon Okuma Kapılama (Gating)**. |
| **Karar Okuma (`DecisionBlock`)** | Tek başlıklı doğrusal okuma + standart NLMS | Nöron korelasyonu ve duruş anlarında ağırlık sürüklenmesi (weight drift) yaşanıyor. | **5 Başlıklı Sızıntılı Entegratör (5-Head LI Readout)** + **Day-0 Çapalı (Anchored) NLMS**. |

## 2. Dosya Bazlı Güncelleme Haritası

Kodlama asistanı aşağıdaki dosyaları sırasıyla güncellemelidir:

1. `configs/base.yaml` & `configs/nhp_motor.yaml`: Yeni hiperparametrelerin (`dt_ms: 4.0`, `backbone_mode: "plif"`, `n_heads: 5`, `anchor_decay: 0.0001`, `target_rate: 0.05`) tanımlanması.
2. `src/encoder.py`: `SpikeEncoder` sınıfının Anscombe + Kanal bazlı Z-skor + Soft-Clipping ve genel M1/S1 uyumlu sürekli/vuruşlu giriş üretecek şekilde yenilenmesi.
3. `src/reservoir.py`: `LIFReservoir` sınıfının `TemporalBackbone` (veya geriye dönük uyumluluk için `LIFReservoir`) adı altında 3 modlu (`"plif"`, `"spiking_gru"`, `"tau_gru"`), 500 hızlı / 500 yavaş alt-popülasyonlu, surrogate-gradient ve homeostatik eşik adaptasyonlu yapıya kavuşturulması.
4. `src/cerebellum.py`: `CerebellumBlock` içine düşük geçiren filtreli hedef $\bar{x}[t+1]$ ve önceki hız tahmini (efference copy $\hat{v}[t-1]$) entegrasyonu.
5. `src/affect.py`: `AffectSpace` içinde beklenmedik sürpriz ($a[t]$), kararlılık ($d[t]$) ve hızlı/yavaş alt-popülasyon kapılama kazançlarının ($g_{\text{fast}}[t], g_{\text{slow}}[t]$) hesaplanması; $\tau[t]$ çevrimiçi modülasyonunun kapatılması.
6. `src/decision.py`: `DecisionBlock` içine 5 farklı zaman sabitli ($\tau_{\text{read}} \in \{16, 32, 64, 100, 160\}\text{ ms}$) Leaky Integrator okuma başlığı ve $W_0$ çapalı (anchored) NLMS güncellemesinin eklenmesi.
7. `experiments/nhp_motor.py` & `experiments/nhp_full.py`: Çoklu oturum ön-eğitimi (100 epoch) + Day-0 oturum ince ayarı (150 epoch) + kesintisiz akış (streaming) çevrimiçi uyum protokolünün kurulması.

## 3. Modül Spesifikasyonları ve Referans PyTorch Kodları

### 3.1. `src/encoder.py` — M1 ve S1 Farkını Veri Katmanında Absorbe Eden Kodlayıcı

M1 (96 kanal) ve S1 (96 kanal) ayrımı için mimariyi ikiye bölmek yerine, her kanalın bazal ateşleme hızı ve patlama varyansı kanal bazlı istatistiklerle dengelenir:

* **Anscombe Dönüşümü:** Poisson spike sayımlarının varyansını stabilize eder:
  $$x_{\text{ans}}[t] = \sqrt{\max(0, x_{\text{raw}}[t]) + 0.375}$$
* **Kanal Bazlı Z-Skor ve Yumuşak Kırpma (Soft-Clipping):** Yalnızca eğitim (train) diliminden hesaplanan kanal ortalaması $\mu_{\text{ch}} \in \mathbb{R}^{C}$ ve standart sapması $\sigma_{\text{ch}} \in \mathbb{R}^{C}$ kullanılır:
  $$z[t] = \frac{x_{\text{ans}}[t] - \mu_{\text{ch}}}{\sigma_{\text{ch}} + 10^{-5}}, \quad x_{\text{norm}}[t] = 3.0 \cdot \tanh\left(\frac{z[t]}{3.0}\right)$$
* **Hedef Yumuşatma (`CerebellumBlock` için):** S1 ve M1 kanallarındaki yüksek frekanslı Poisson gürültüsünü süzmek için `encoder` aynı zamanda $\tau_{\text{smooth}} = 48\text{ ms}$ ($\alpha_{\text{sm}} = \exp(-\Delta t / 48\text{ ms})$) ile yumuşatılmış hedef vektörü $\bar{x}[t]$ üretir:
  $$\bar{x}[t] = \alpha_{\text{sm}} \bar{x}[t-1] + (1 - \alpha_{\text{sm}}) x_{\text{norm}}[t]$$

```python
import torch
import torch.nn as nn

class SpikeEncoder(nn.Module):
    def __init__(self, num_channels: int, dt_ms: float = 4.0, smooth_tau_ms: float = 48.0):
        super().__init__()
        self.num_channels = num_channels
        self.dt_ms = dt_ms
        self.alpha_smooth = float(torch.exp(torch.tensor(-dt_ms / smooth_tau_ms)))
        self.register_buffer("mean", torch.zeros(1, num_channels))
        self.register_buffer("std", torch.ones(1, num_channels))
        self.register_buffer("x_smooth", torch.zeros(1, num_channels))
        self.fitted = False

    @torch.no_grad()
    def fit(self, x_train: torch.Tensor) -> None:
        # x_train shape: [T, C] or [B, T, C]
        x_ans = torch.sqrt(torch.clamp(x_train, min=0.0) + 0.375)
        dims = (0,) if x_ans.ndim == 2 else (0, 1)
        self.mean.copy_(x_ans.mean(dim=dims, keepdim=False).unsqueeze(0))
        self.std.copy_(x_ans.std(dim=dims, keepdim=False).unsqueeze(0).clamp(min=1e-4))
        self.fitted = True

    def reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        dev = device if device is not None else self.mean.device
        self.x_smooth = torch.zeros(batch_size, self.num_channels, device=dev)

    def forward(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_raw shape: [B, C] (tek zaman adımı)
        x_ans = torch.sqrt(torch.clamp(x_raw, min=0.0) + 0.375)
        z = (x_ans - self.mean) / (self.std + 1e-5)
        x_norm = 3.0 * torch.tanh(z / 3.0)
        self.x_smooth = self.alpha_smooth * self.x_smooth + (1.0 - self.alpha_smooth) * x_norm.detach()
        return x_norm, self.x_smooth
```

### 3.2. `src/reservoir.py` — Birleşik Hızlı/Yavaş Omurga (`plif` | `spiking_gru` | `tau_gru`)

Omurga $N=1000$ nöron içerir ve iki eşit alt-popülasyona ayrılır:
* **Hızlı Alt-Popülasyon ($i \in [0, 499]$):** $\tau_{\text{mem}, i} \in [4\text{ ms}, 40\text{ ms}]$
* **Yavaş Alt-Popülasyon ($i \in [500, 999]$):** $\tau_{\text{mem}, i} \in [40\text{ ms}, 400\text{ ms}]$

Üç çalışma modu tek sınıfta desteklenir:
1. `"plif"`: Öğrenilebilir $\tau_{\text{mem}, i}$ ve $\tau_{\text{syn}, i}$, soft-reset ($-\vartheta_i s_i[t-1]$), surrogate gradient ve yerel homeostatik eşik adaptasyonu.
2. `"spiking_gru"`: İç durumu $\tau_0$ dağılımlı (chrono-initialized) GRU kapılarıyla güncellenen, ancak dışarıya ve tekrarlayan bağlantılara yalnızca ikili spike $s[t] \in \{0, 1\}^N$ gönderen nöromorfik hibrit mod.
3. `"tau_gru"`: Ateşleme eşiği olmayan sürekli (float) aktivasyonlu $\tau$-GRU üst sınır (upper-bound) kontrol modu.

```python
import math
import torch
import torch.nn as nn

class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_th: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(v_minus_th)
        ctx.scale = scale
        return (v_minus_th >= 0.0).to(v_minus_th.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_minus_th,) = ctx.saved_tensors
        scale = ctx.scale
        surrogate_grad = 1.0 / ((1.0 + scale * torch.abs(v_minus_th)) ** 2)
        return grad_output * surrogate_grad, None

def spike_fn(v_minus_th: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(v_minus_th, scale)

class TemporalBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neurons: int = 1000,
        dt_ms: float = 4.0,
        mode: str = "plif",  # "plif" | "spiking_gru" | "tau_gru"
        dropout_p: float = 0.3,
        target_rate: float = 0.05,
        homeo_lr: float = 1e-3,
    ):
        super().__init__()
        assert num_neurons % 2 == 0, "num_neurons çift sayı olmalıdır (500 hızlı + 500 yavaş)."
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.half_n = num_neurons // 2
        self.dt_ms = dt_ms
        self.mode = mode
        self.target_rate = target_rate
        self.homeo_lr = homeo_lr
        self.dropout = nn.Dropout(p=dropout_p)

        # Hızlı [4, 40] ms ve Yavaş [40, 400] ms alt-popülasyon sınırları
        self.register_buffer("tau_min", torch.cat([
            torch.full((self.half_n,), 4.0),
            torch.full((self.half_n,), 40.0)
        ]))
        self.register_buffer("tau_max", torch.cat([
            torch.full((self.half_n,), 40.0),
            torch.full((self.half_n,), 400.0)
        ]))

        # Log-uniform başlangıçlı öğrenilebilir logit_tau_mem
        u_init = torch.rand(num_neurons).clamp(0.05, 0.95)
        self.logit_tau_mem = nn.Parameter(torch.log(u_init / (1.0 - u_init)))

        # Sinaptik zaman sabiti [4, 100] ms
        u_syn = torch.rand(num_neurons).clamp(0.05, 0.95)
        self.logit_tau_syn = nn.Parameter(torch.log(u_syn / (1.0 - u_syn)))

        if mode == "plif":
            self.w_in = nn.Linear(input_dim, num_neurons, bias=True)
            self.w_rec = nn.Linear(num_neurons, num_neurons, bias=False)
            self._init_fluctuation_driven()
        else:
            # GRU / Spiking-GRU kapıları
            self.w_z = nn.Linear(input_dim, num_neurons, bias=True)
            self.u_z = nn.Linear(num_neurons, num_neurons, bias=False)
            self.w_r = nn.Linear(input_dim, num_neurons, bias=True)
            self.u_r = nn.Linear(num_neurons, num_neurons, bias=False)
            self.w_h = nn.Linear(input_dim, num_neurons, bias=True)
            self.u_h = nn.Linear(num_neurons, num_neurons, bias=False)
            self._init_chrono_gru()

        # Homeostatik eşik ve ortalama ateşleme izi
        self.register_buffer("v_th", torch.ones(1, num_neurons))
        self.register_buffer("rate_ema", torch.full((1, num_neurons), target_rate))
        self.register_buffer("v", torch.zeros(1, num_neurons))
        self.register_buffer("i_syn", torch.zeros(1, num_neurons))
        self.register_buffer("s", torch.zeros(1, num_neurons))

    def get_tau_mem(self) -> torch.Tensor:
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.logit_tau_mem)

    def get_tau_syn(self) -> torch.Tensor:
        return 4.0 + (100.0 - 4.0) * torch.sigmoid(self.logit_tau_syn)

    def _init_fluctuation_driven(self) -> None:
        # Rossbroich et al. (2022) dalgalanma güdümlü başlatma
        nn.init.normal_(self.w_in.weight, mean=0.0, std=1.0 / math.sqrt(self.input_dim))
        nn.init.zeros_(self.w_in.bias)
        nn.init.orthogonal_(self.w_rec.weight, gain=0.90)

    def _init_chrono_gru(self) -> None:
        tau_0 = self.get_tau_mem().detach()
        chrono_bias = -torch.log(torch.clamp(tau_0 / self.dt_ms - 1.0, min=0.1))
        self.w_z.bias.data.copy_(chrono_bias)
        nn.init.zeros_(self.w_r.bias)
        nn.init.zeros_(self.w_h.bias)
        for layer in [self.w_z, self.w_r, self.w_h]:
            nn.init.xavier_uniform_(layer.weight)
        for layer in [self.u_z, self.u_r, self.u_h]:
            nn.init.orthogonal_(layer.weight, gain=0.90)

    def reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        dev = device if device is not None else self.v_th.device
        self.v = torch.zeros(batch_size, self.num_neurons, device=dev)
        self.i_syn = torch.zeros(batch_size, self.num_neurons, device=dev)
        self.s = torch.zeros(batch_size, self.num_neurons, device=dev)

    def step(self, x_t: torch.Tensor, online_homeostasis: bool = False) -> torch.Tensor:
        x_drop = self.dropout(x_t) if self.training else x_t
        s_prev = self.dropout(self.s) if self.training else self.s

        tau_mem = self.get_tau_mem().unsqueeze(0)
        beta = torch.exp(-self.dt_ms / tau_mem)

        if self.mode == "plif":
            tau_syn = self.get_tau_syn().unsqueeze(0)
            alpha = torch.exp(-self.dt_ms / tau_syn)
            self.i_syn = alpha * self.i_syn + self.w_in(x_drop) + self.w_rec(s_prev)
            # Soft-reset (reset by subtraction)
            self.v = beta * self.v + (1.0 - beta) * self.i_syn - self.v_th * self.s
            out = spike_fn(self.v - self.v_th)
            self.s = out
        elif self.mode == "spiking_gru":
            z = torch.sigmoid(self.w_z(x_drop) + self.u_z(s_prev))
            r = torch.sigmoid(self.w_r(x_drop) + self.u_r(s_prev))
            h_tilde = torch.tanh(self.w_h(x_drop) + self.u_h(r * s_prev))
            self.v = (1.0 - z) * self.v + z * h_tilde - self.v_th * self.s
            out = spike_fn(self.v - self.v_th)
            self.s = out
        else:  # "tau_gru" (sürekli aktivasyonlu kontrol modu)
            z = torch.sigmoid(self.w_z(x_drop) + self.u_z(s_prev))
            r = torch.sigmoid(self.w_r(x_drop) + self.u_r(s_prev))
            h_tilde = torch.tanh(self.w_h(x_drop) + self.u_h(r * s_prev))
            self.v = (1.0 - z) * self.v + z * h_tilde
            out = self.v
            self.s = out

        # Çevrimiçi test aşamasında yerel homeostatik eşik adaptasyonu
        if online_homeostasis and self.mode in ("plif", "spiking_gru") and not self.training:
            with torch.no_grad():
                batch_rate = out.mean(dim=0, keepdim=True)
                self.rate_ema.mul_(0.995).add_(0.005 * batch_rate)
                self.v_th.add_(self.homeo_lr * (self.rate_ema - self.target_rate))
                self.v_th.clamp_(0.3, 2.5)

        return out

    @staticmethod
    def compute_homeostatic_loss(spike_seq: torch.Tensor, f_min: float = 0.02, f_max: float = 0.08) -> torch.Tensor:
        # spike_seq shape: [B, T, N]
        mean_rate = spike_seq.mean(dim=(0, 1))
        low_viol = torch.relu(f_min - mean_rate) ** 2
        high_viol = torch.relu(mean_rate - f_max) ** 2
        return (low_viol + high_viol).mean()
```

### 3.3. `src/cerebellum.py` — Efference Copy ve Yumuşatılmış Duyusal Tahmin

Serebellum bloğu, rezervuarın ortalama okuma izi $\bar{r}[t] \in \mathbb{R}^{N}$ ile bir önceki adımda tahmin edilen motor hız komutunu (**Efference Copy**, $\hat{v}[t-1] \in \mathbb{R}^2$) birleştirerek bir sonraki yumuşatılmış kortikal durumu $\bar{x}[t+1]$ tahmin eder:

$$\phi[t] = \left[\bar{r}[t] \; ; \; \hat{v}[t-1]\right] \in \mathbb{R}^{N+2}, \quad \hat{x}_{\text{smooth}}[t+1] = W_{\text{cer}}[t] \phi[t]$$
$$e_{\text{cer}}[t] = \bar{x}[t+1] - \hat{x}_{\text{smooth}}[t+1]$$

```python
import torch
import torch.nn as nn

class CerebellumBlock(nn.Module):
    def __init__(self, reservoir_dim: int, input_dim: int, vel_dim: int = 2, lr: float = 1e-3, eps: float = 1e-4):
        super().__init__()
        self.feat_dim = reservoir_dim + vel_dim
        self.input_dim = input_dim
        self.lr = lr
        self.eps = eps
        self.register_buffer("w_cer", torch.zeros(input_dim, self.feat_dim))
        self.register_buffer("last_phi", torch.zeros(1, self.feat_dim))
        self.register_buffer("last_pred", torch.zeros(1, input_dim))

    def reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        dev = device if device is not None else self.w_cer.device
        self.last_phi = torch.zeros(batch_size, self.feat_dim, device=dev)
        self.last_pred = torch.zeros(batch_size, self.input_dim, device=dev)

    @torch.no_grad()
    def observe_and_predict(
        self,
        x_smooth_current: torch.Tensor,
        r_trace: torch.Tensor,
        v_pred_prev: torch.Tensor,
        adapt: bool = True,
    ) -> torch.Tensor:
        # 1. Önceki adımın tahmin hatasını hesapla
        err = x_smooth_current - self.last_pred  # [B, C]
        if adapt:
            # yerel NLMS güncellemesi
            phi = self.last_phi  # [B, N+2]
            norm_sq = (phi * phi).sum(dim=-1, keepdim=True) + self.eps
            grad = torch.matmul(err.T, phi / norm_sq) / err.shape[0]
            self.w_cer.add_(self.lr * grad)

        # 2. Bir sonraki adım için Efference Copy destekli tahmin üret
        phi_now = torch.cat([r_trace, v_pred_prev], dim=-1)
        self.last_phi = phi_now.detach()
        self.last_pred = torch.matmul(phi_now, self.w_cer.T)
        return err
```

### 3.4. `src/affect.py` — Yu & Dayan (2005) Beklenmedik Sürpriz ve Alt-Popülasyon Kapılama

**Kritik Kural:** `AffectSpace` artık rezervuarın $\tau[t]$ zaman sabitini çevrimiçi değiştirmez (faz bozulmasını önler). Bunun yerine:
1. Kısa vadeli hata enerjisi $E_{\text{fast}}[t]$ ile uzun vadeli beklenen gürültü enerjisi $E_{\text{slow}}[t]$ oranından **Beklenmedik Sürpriz (Noradrenalin / `arousal` $a[t]$)** üretir:
   $$a[t] = \tanh\left(\frac{E_{\text{fast}}[t] - E_{\text{slow}}[t]}{E_{\text{slow}}[t] + 10^{-5}}\right)$$
2. Hızlı ve yavaş nöron alt-popülasyonlarının okuma kazançlarını ($g_{\text{fast}}[t], g_{\text{slow}}[t]$) ve çevrimiçi NLMS adım boyunu ($\eta[t]$) modüle eder:
   $$g_{\text{fast}}[t] = 1.0 + \alpha_{\text{gate}} \cdot a[t], \quad g_{\text{slow}}[t] = 1.0 - \alpha_{\text{gate}} \cdot a[t]$$
   $$\eta[t] = \eta_0 \cdot \left(1.0 + \alpha_{\eta} \cdot \max(0, a[t])\right)$$

```python
import torch
import torch.nn as nn

class AffectSpace(nn.Module):
    def __init__(
        self,
        alpha_fast: float = 0.05,
        alpha_slow: float = 0.001,
        alpha_gate: float = 0.25,
        alpha_eta: float = 2.0,
        base_eta: float = 5e-4,
    ):
        super().__init__()
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.alpha_gate = alpha_gate
        self.alpha_eta = alpha_eta
        self.base_eta = base_eta

        self.register_buffer("e_fast", torch.tensor(1.0))
        self.register_buffer("e_slow", torch.tensor(1.0))
        self.register_buffer("valence", torch.tensor(0.0))
        self.register_buffer("arousal", torch.tensor(0.0))
        self.register_buffer("dominance", torch.tensor(0.0))

    def reset_state(self, init_error: float = 1.0) -> None:
        self.e_fast.fill_(init_error)
        self.e_slow.fill_(init_error)
        self.valence.zero_()
        self.arousal.zero_()
        self.dominance.zero_()

    @torch.no_grad()
    def update(self, cer_error: torch.Tensor) -> dict[str, float]:
        # cer_error shape: [B, C]
        inst_err_sq = (cer_error ** 2).mean()
        prev_fast = self.e_fast.clone()

        self.e_fast.mul_(1.0 - self.alpha_fast).add_(self.alpha_fast * inst_err_sq)
        self.e_slow.mul_(1.0 - self.alpha_slow).add_(self.alpha_slow * inst_err_sq)

        # Beklenmedik Sürpriz (Noradrenalin / Arousal)
        unexpected_ratio = (self.e_fast - self.e_slow) / (self.e_slow + 1e-5)
        self.arousal = torch.tanh(unexpected_ratio)

        # Valence (Performans memnuniyeti: beklentiden düşük hata -> pozitif)
        self.valence = -self.arousal

        # Dominance (Hata azalma kararlılığı)
        err_improvement = (prev_fast - self.e_fast) / (self.e_slow + 1e-5)
        self.dominance = 0.95 * self.dominance + 0.05 * torch.tanh(10.0 * err_improvement)

        a_val = float(self.arousal.item())
        g_fast = 1.0 + self.alpha_gate * a_val
        g_slow = 1.0 - self.alpha_gate * a_val
        eta_t = self.base_eta * (1.0 + self.alpha_eta * max(0.0, a_val))

        return {
            "valence": float(self.valence.item()),
            "arousal": a_val,
            "dominance": float(self.dominance.item()),
            "g_fast": g_fast,
            "g_slow": g_slow,
            "eta": eta_t,
        }
```

### 3.5. `src/decision.py` — 5 Başlıklı Sızıntılı Entegratör (5-Head LI) ve Çapalı NLMS

Rezervuardan çıkan $s[t] \in \mathbb{R}^N$ vektörü 5 farklı zaman sabitli ($\tau_{\text{read}, k} \in \{16, 32, 64, 100, 160\}\text{ ms}$) sızıntılı entegratörden geçirilerek yoğun (dense) uygunluk izleri $r^{(k)}[t]$ elde edilir:

$$r^{(k)}[t] = \kappa_k r^{(k)}[t-1] + (1 - \kappa_k) s[t], \quad \kappa_k = \exp\left(-\frac{\Delta t}{\tau_{\text{read}, k}}\right)$$

Ardından `AffectSpace` kazançlarıyla hızlı ve yavaş yarılar kapılanır ($\tilde{r}^{(k)}[t]$) ve 5 başlığın ortalaması alınır. Çevrimiçi uyum sırasında her başlık, Day-0'da öğrenilmiş çapa ağırlığına ($W_0^{(k)}$) doğru elastik bir geri çekme ($\alpha_{\text{anchor}}$) ile güncellenir:

$$W^{(k)}[t+1] = (1 - \alpha_{\text{anchor}}) W^{(k)}[t] + \alpha_{\text{anchor}} W_0^{(k)} + \eta[t] \frac{e_v[t] \left(\tilde{r}^{(k)}[t]\right)^T}{\|\tilde{r}^{(k)}[t]\|^2 + \epsilon}$$

```python
import torch
import torch.nn as nn

class MultiHeadDecisionBlock(nn.Module):
    def __init__(
        self,
        num_neurons: int = 1000,
        out_dim: int = 2,
        dt_ms: float = 4.0,
        readout_taus_ms: tuple[float, ...] = (16.0, 32.0, 64.0, 100.0, 160.0),
        anchor_decay: float = 1e-4,
        nlms_eps: float = 1e-4,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.half_n = num_neurons // 2
        self.out_dim = out_dim
        self.num_heads = len(readout_taus_ms)
        self.anchor_decay = anchor_decay
        self.nlms_eps = nlms_eps

        kappas = [float(torch.exp(torch.tensor(-dt_ms / tau))) for tau in readout_taus_ms]
        self.register_buffer("kappas", torch.tensor(kappas).view(self.num_heads, 1, 1))

        # 5 okuma başlığı
        self.heads = nn.ModuleList([
            nn.Linear(num_neurons, out_dim, bias=True) for _ in range(self.num_heads)
        ])

        # Day-0 Çapa (Anchor) ağırlıkları
        self.register_buffer("w_anchor", torch.zeros(self.num_heads, out_dim, num_neurons))
        self.register_buffer("b_anchor", torch.zeros(self.num_heads, out_dim))

        # Sızıntılı entegratör durum izleri: [K, B, N]
        self.register_buffer("r_traces", torch.zeros(self.num_heads, 1, num_neurons))

    def reset_state(self, batch_size: int = 1, device: torch.device = None) -> None:
        dev = device if device is not None else self.kappas.device
        self.r_traces = torch.zeros(self.num_heads, batch_size, self.num_neurons, device=dev)

    @torch.no_grad()
    def save_anchor_weights(self) -> None:
        for k, head in enumerate(self.heads):
            self.w_anchor[k].copy_(head.weight.data)
            self.b_anchor[k].copy_(head.bias.data)

    def forward_step(
        self,
        s_t: torch.Tensor,
        g_fast: float = 1.0,
        g_slow: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # s_t shape: [B, N]
        s_expanded = s_t.unsqueeze(0)  # [1, B, N]
        self.r_traces = self.kappas * self.r_traces + (1.0 - self.kappas) * s_expanded

        # Alt-popülasyon kapılama (Hızlı ilk 500, Yavaş son 500)
        gate_vec = torch.cat([
            torch.full((1, 1, self.half_n), g_fast, device=s_t.device),
            torch.full((1, 1, self.half_n), g_slow, device=s_t.device),
        ], dim=-1)
        gated_traces = self.r_traces * gate_vec  # [K, B, N]

        preds = []
        for k, head in enumerate(self.heads):
            preds.append(head(gated_traces[k]))
        v_pred = torch.stack(preds, dim=0).mean(dim=0)  # [B, 2]
        mean_trace = gated_traces.mean(dim=0)  # [B, N] (Cerebellum için özet iz)
        return v_pred, mean_trace

    @torch.no_grad()
    def online_anchored_nlms_update(
        self,
        v_target: torch.Tensor,
        v_pred: torch.Tensor,
        eta: float,
        g_fast: float = 1.0,
        g_slow: float = 1.0,
    ) -> None:
        # v_target, v_pred shape: [B, 2]
        err_v = v_target - v_pred  # [B, 2]
        gate_vec = torch.cat([
            torch.full((1, self.half_n), g_fast, device=v_target.device),
            torch.full((1, self.half_n), g_slow, device=v_target.device),
        ], dim=-1)

        for k, head in enumerate(self.heads):
            r_k = self.r_traces[k] * gate_vec  # [B, N]
            norm_sq = (r_k * r_k).sum(dim=-1, keepdim=True) + self.nlms_eps
            delta_w = torch.matmul(err_v.T, r_k / norm_sq) / err_v.shape[0]

            # Çapalı (Anchored) ağırlık güncellemesi
            head.weight.data.mul_(1.0 - self.anchor_decay).add_(
                self.anchor_decay * self.w_anchor[k] + eta * delta_w
            )
```

## 4. Eğitim, Çevrimiçi Uyum ve Değerlendirme Protokolü (`experiments/nhp_full.py`)

Kodlama asistanı deney akışını şu 3 aşamalı boru hattı (pipeline) olarak kurmalıdır:

1. **Aşama 1 — Çoklu Oturum Surrogate-Gradient Ön-Eğitimi (Offline Pre-training):**
   * Her primat için (Indy: I1, I2, I3; Loco: L1, L2, L3) her oturumun ilk %65'lik eğitim verisi $2\text{ saniyelik}$ ($\Delta t = 4\text{ ms}$ için $T=500$ adım, yarı-örtüşmeli `stride=250`) pencerelere bölünür.
   * Tüm eğitim pencereleri birleştirilip `batch_size=32`, `AdamW(lr=1e-3, weight_decay=1e-4)` ve `CosineAnnealingLR` ile **100 epoch** boyunca eğitilir.
   * Kayıp fonksiyonu:
     $$\mathcal{L}_{\text{total}} = \sqrt{\frac{1}{T}\sum_{t=1}^T \|v[t] - \hat{v}[t]\|^2 + 10^{-6}} + 10.0 \cdot \mathcal{L}_{\text{homeo}}$$
2. **Aşama 2 — Oturum Özgü İnce Ayar ve Çapa Kaydı (Session Fine-Tuning & Anchor Saving):**
   * Ön-eğitilmiş model, hedef oturumun (veya günler arası protokolde **Day-0 kaynak oturumunun**) %65 eğitim diliminde **150 epoch** ince ayara tabi tutulur (%10 doğrulama diliminde en yüksek $R^2$ veren epoch seçilir).
   * Eğitim tamamlandığında `decision_block.save_anchor_weights()` çağrılarak Day-0 çapa ağırlıkları ($W_0^{(k)}$) dondurulur.
3. **Aşama 3 — Kesintisiz Akış (Streaming) Çevrimiçi Uyum ve Ablasyon Tablosu:**
   * Test dilimi (%25 oturum içi test veya **Day-k yeni gün oturumu**) pencerelere bölünmeden $t = 1, \dots, T_{\text{test}}$ boyunca adım adım akıtılır (`model.eval()`, gradyan kapalı, `online_homeostasis=True`).
   * Aşağıdaki **5 ablasyon ve kontrol koşulu** aynı test akışı üzerinde çalıştırılarak `scripts/make_tables.py` ile raporlanır:

| Koşul Kodu | Açıklama | Beklenen Davranış |
| :--- | :--- | :--- |
| **A (Frozen Baseline)** | Ön-eğitimli + ince ayarlı model, çevrimiçi NLMS kapalı ($\eta = 0$). | Oturum içinde yüksek ($R^2 \approx 0.65+$), günler arasında (cross-day) ciddi düşüş. |
| **B (Fixed-$\bar{\eta}$ Anchored NLMS)** | Çapalı NLMS açık, ancak öğrenme hızı D koşulunun ortalama $\bar{\eta}$ değerine sabitlenmiş, kapılama kapalı ($g_{\text{fast}}=g_{\text{slow}}=1$). | Temel çevrimiçi uyum kazancını ölçen birincil adil kontrol. |
| **C (Shuffled-$\eta[t]$ Control)** | D koşulunda üretilen $\eta[t]$ dizisi zamanda rastgele karıştırılarak uygulanır. | Kazancın yalnızca $\eta$ varyansından değil, sürpriz zamanlamasından geldiğini kanıtlar. |
| **D (Full Spike v2 — Proposed)** | Beklenmedik sürpriz modülasyonlu $\eta[t]$ + Hızlı/Yavaş alt-popülasyon kapılama + Çapalı NLMS + Homeostatik eşik. | Özellikle günler arası (cross-day) ve ani kanal kaybı (perturbation) senaryolarında en yüksek $R^2$ ve en hızlı toparlanma. |
| **E (Perturbation Stress Test)** | Test akışının ortasında rastgele 20 giriş kanalı sıfırlanır (electrode dropout). | D koşulunun B ve A koşullarına kıyasla kaç saniyede yeniden yakınsadığını (recovery time) ölçer. |
