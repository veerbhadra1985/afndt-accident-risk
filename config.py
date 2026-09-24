"""All experimental settings in one place. Every value here is reported in the manuscript."""
from dataclasses import dataclass, field

# Six regions over the 49 contiguous jurisdictions (48 states + DC), adapted from
# U.S. Census Bureau regions/divisions so that each region contains adjacent states.
REGIONS = {
    "Northeast":     ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Southeast":     ["DE", "DC", "MD", "VA", "WV", "NC", "SC", "GA", "FL"],
    "Midwest":       ["OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO"],
    "South Central": ["KY", "TN", "AL", "MS", "AR", "LA", "OK", "TX"],
    "Plains-Mountain": ["ND", "SD", "NE", "KS", "MT", "WY", "CO", "NM"],
    "West":          ["ID", "UT", "AZ", "NV", "WA", "OR", "CA"],
}
STATES = sorted(s for v in REGIONS.values() for s in v)
STATE_REGION = {s: r for r, v in REGIONS.items() for s in v}
TIME_BINS = [(0, 6), (6, 12), (12, 18), (18, 24)]   # night, morning, afternoon, evening
FEATURES = ["month_sin", "month_cos", "hour_sin", "hour_cos", "temp_z", "temp_std_z"]
TEMP_COLS = [4, 5]                                   # indices of temperature features

@dataclass
class Config:
    csv: str = "US_Accidents_March23.csv"
    out: str = "results"
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    label_top: float = 0.40          # top 40% of training-context counts -> high risk
    val_frac: float = 0.15           # stratified hold-out of training contexts
    # federated
    rounds: int = 15
    local_steps: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # model
    alpha: float = 0.6               # residual mixing, Eq. (2)
    beta: float = 0.95               # LIF decay, Eq. (5)
    theta_spike: float = 0.5         # firing threshold, Eq. (6)
    t_spike: int = 8                 # LIF time steps
    surrogate_k: float = 10.0        # fast-sigmoid surrogate slope
    d_model: int = 64
    heads: int = 4
    # RAMAL
    sigma_r: float = 0.2             # perturbation std for L_rob
    t_w: float = 2.0                 # DWA temperature
    # baselines
    central_epochs: int = 300
    patience: int = 30
    mu_prox: float = 0.01            # FedProx
    scaffold_lr: float = 0.05
    # analyses
    noise_sigmas: list = field(default_factory=lambda: [0.1, 0.2, 0.5])
    noise_draws: int = 10
    alpha_grid: list = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.6, 0.7, 0.9])
    beta_grid: list = field(default_factory=lambda: [0.85, 0.90, 0.95, 0.98])
    rounds_grid: list = field(default_factory=lambda: [5, 10, 15, 20, 30])
    sens_seeds: list = field(default_factory=lambda: [0, 1, 2])
    shap_nodes: int = 200
    shap_nsamples: int = 200
