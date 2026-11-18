"""Lightweight provenance checks shared by NTK measurement and offline plotting."""

import ast
import hashlib
from pathlib import Path

COMPARABLE_FIELDS = (
    "paligemma_variant",
    "action_expert_variant",
    "num_vlm_prompt_tokens",
    "num_prompt_tokens",
    "chunk_size",
    "max_action_dim",
    "max_state_dim",
    "image_resolution",
    "input_features",
    "output_features",
    "normalization_mapping",
    "tokenizer_max_length",
    "min_period",
    "max_period",
    "time_sampling_beta_alpha",
    "time_sampling_beta_beta",
    "time_sampling_scale",
    "time_sampling_offset",
)

MEASUREMENT_FIELDS = (
    "protocol",
    "scope",
    "dtype",
    "seeds",
    "output_probes",
    "probe_seed",
    "sketch_dim",
    "sketch_seed",
    "dataset_repo_id",
    "sample_indices",
    "sample_digest",
    "torch_version",
)

CORE_FUNCTIONS = (
    "load_samples",
    "load_policy",
    "load_pi05_config",
    "fingerprint_samples",
    "analyze_seed",
)
CORE_FILES = (
    "utils/ntk.py",
    "policies/pi05/modeling_pi05.py",
    "policies/pi_gemma.py",
    "policies/pi05/attention_pi05.py",
)

# Audited at commit 0f81da7349143dee566d2c17ffd93f07a854f132. These are hashes
# of the original whole-script provenance and its unchanged numerical path,
# respectively. Unknown legacy revisions are deliberately NOT assumed compatible.
LEGACY_MEASUREMENT_DIGESTS = {
    "e020f780496d97b5cee01dfcd62861fb70235ee0ae743578de84d2b921a2676e":
        "fc3f3b51608ff5d7e73d30506fd80df96f35bc1ff4ef57427052ba0d4429bad6",
}


def measurement_digest(analyzer_path=None):
    """Hash numerical functions and dependencies, excluding CLI/layout changes."""
    root = Path(__file__).parents[1]
    path = Path(analyzer_path) if analyzer_path else root / "scripts/analyze_pi05_ntk_stages.py"
    source = path.read_text()
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    payload = b"\n\n".join(functions[name].encode() for name in CORE_FUNCTIONS)
    payload += b"".join((root / name).read_bytes() for name in CORE_FILES)
    return hashlib.sha256(payload).hexdigest()


def measurement_identity(manifest):
    """Resolve new manifests or the explicitly audited legacy numerical path."""
    return manifest.get("measurement_digest") or LEGACY_MEASUREMENT_DIGESTS.get(
        manifest.get("source_digest")
    )


def validate_measurement_pair(reference, current):
    """Reject different probes, processed inputs, precision or numerical paths."""
    identity = measurement_identity(reference)
    if not identity or identity != measurement_identity(current):
        raise ValueError(
            "NTK measurement source differs or is an unaudited legacy revision; "
            "remeasure both methods with the same analysis code"
        )
    for field in MEASUREMENT_FIELDS:
        if field not in reference or field not in current or reference[field] != current[field]:
            raise ValueError(f"Incomparable NTK measurement setting: {field}")
    if str(reference.get("device", "")).split(":")[0] != str(current.get("device", "")).split(":")[0]:
        raise ValueError("Incomparable NTK device families")
