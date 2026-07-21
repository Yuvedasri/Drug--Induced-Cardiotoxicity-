"""
Lightweight Graph Attention Network for hERG cardiotoxicity classification.
Baseline (non-causal) architecture -- this is the foundation we'll extend
with the causal component next.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
import re


QT_RISK_KEYWORDS = (
    "amiodarone",
    "sotalol",
    "dofetilide",
    "quinidine",
    "procainamide",
    "azithromycin",
    "clarithromycin",
    "erythromycin",
    "haloperidol",
    "ondansetron",
    "methadone",
    "fluconazole",
    "moxifloxacin",
)


def build_clinical_feature_vector(metadata, device=None):
    """Convert optional clinical context into a fixed feature vector.

    The model in this repository was originally trained on molecular graphs only.
    This helper adds a lightweight clinical signal at inference time so the final
    prediction can reflect the patient's context as well as the molecule.
    """
    metadata = metadata or {}
    history = metadata.get("medical_history", {}) or {}
    labs = metadata.get("lab_report", {}) or {}
    medication_text = str(metadata.get("current_medication") or "").lower()

    age = metadata.get("age")
    age_value = float(age) if age is not None else 0.0
    age_norm = min(max(age_value, 0.0), 100.0) / 100.0
    age_over_65 = 1.0 if age_value >= 65.0 else 0.0

    sex = str(metadata.get("sex") or "").strip().lower()
    sex_male = 1.0 if sex == "male" else 0.0
    sex_female = 1.0 if sex == "female" else 0.0
    sex_other = 1.0 if sex == "other" else 0.0

    diabetes = 1.0 if history.get("diabetes") else 0.0
    hypertension = 1.0 if history.get("hypertension") else 0.0
    heart_failure = 1.0 if history.get("heart_failure") else 0.0

    medication_present = 1.0 if medication_text else 0.0
    qt_risk_medication = 1.0 if any(keyword in medication_text for keyword in QT_RISK_KEYWORDS) else 0.0

    potassium = labs.get("potassium")
    sodium = labs.get("sodium")
    magnesium = labs.get("magnesium")
    troponin = labs.get("troponin")

    potassium_low = 0.0
    potassium_high = 0.0
    if potassium is not None:
        potassium = float(potassium)
        potassium_low = max(0.0, (3.5 - potassium) / 3.5)
        potassium_high = max(0.0, (potassium - 5.0) / 5.0)

    sodium_low = 0.0
    sodium_high = 0.0
    if sodium is not None:
        sodium = float(sodium)
        sodium_low = max(0.0, (135.0 - sodium) / 135.0)
        sodium_high = max(0.0, (sodium - 145.0) / 145.0)

    magnesium_low = 0.0
    if magnesium is not None:
        magnesium = float(magnesium)
        magnesium_low = max(0.0, (1.7 - magnesium) / 1.7)

    troponin_high = 0.0
    troponin_very_high = 0.0
    if troponin is not None:
        troponin = max(0.0, float(troponin))
        troponin_high = min(1.0, troponin / 0.04)
        troponin_very_high = 1.0 if troponin >= 0.1 else 0.0

    features = torch.tensor(
        [
            age_norm,
            age_over_65,
            sex_male,
            sex_female,
            sex_other,
            diabetes,
            hypertension,
            heart_failure,
            medication_present,
            qt_risk_medication,
            potassium_low,
            potassium_high,
            sodium_low,
            sodium_high,
            magnesium_low,
            troponin_high,
            troponin_very_high,
        ],
        dtype=torch.float32,
        device=device,
    )
    return features


class LightweightGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            concat = True
            self.convs.append(
                GATv2Conv(in_dim, out_dim, heads=heads, edge_dim=edge_dim,
                          concat=concat, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(out_dim * heads))
            in_dim = out_dim * heads

        pooled_dim = in_dim * 2  # mean + max pooling concatenated

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._last_attention = None  # populated on forward when return_attention=True

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False, clinical_features=None, clinical_weight=0.35):
        attn_weights = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                x, (ei, alpha) = conv(x, edge_index, edge_attr, return_attention_weights=True)
                attn_weights.append((ei, alpha))
            else:
                x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        out = self.classifier(pooled).squeeze(-1)

        if clinical_features is not None:
            if isinstance(clinical_features, dict):
                clinical_features = build_clinical_feature_vector(clinical_features, device=out.device)
            elif not torch.is_tensor(clinical_features):
                clinical_features = torch.tensor(clinical_features, dtype=torch.float32, device=out.device)
            else:
                clinical_features = clinical_features.to(out.device, dtype=torch.float32)

            if clinical_features.dim() == 1:
                clinical_features = clinical_features.unsqueeze(0)

            clinical_coeffs = torch.tensor(
                [
                    0.10,  # age_norm
                    0.15,  # age_over_65
                    0.03,  # sex_male
                    0.02,  # sex_female
                    0.01,  # sex_other
                    0.18,  # diabetes
                    0.15,  # hypertension
                    0.28,  # heart_failure
                    0.08,  # medication_present
                    0.22,  # qt_risk_medication
                    0.20,  # potassium_low
                    0.08,  # potassium_high
                    0.12,  # sodium_low
                    0.04,  # sodium_high
                    0.18,  # magnesium_low
                    0.10,  # troponin_high
                    0.20,  # troponin_very_high
                ],
                dtype=torch.float32,
                device=out.device,
            )
            clinical_delta = torch.matmul(clinical_features, clinical_coeffs)
            if clinical_delta.dim() > 1:
                clinical_delta = clinical_delta.squeeze(-1)
            out = out + clinical_weight * torch.tanh(clinical_delta)

        if return_attention:
            return out, attn_weights
        return out