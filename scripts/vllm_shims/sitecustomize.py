"""Import-compatibility shim for serving Mistral-format Ministral-3 on vllm/vllm-openai:v0.30.0.

vLLM 0.30.0 (model_executor/models/pixtral.py) imports `PixtralRotaryEmbedding` and
`position_ids_in_meshgrid` from transformers; the transformers 5.17 bundled in that image renamed
the first to `PixtralVisionRotaryEmbedding` and removed the second, so every Pixtral/Mistral3
architecture fails at import. Both names are used only by the Hugging Face-format Pixtral vision
encoder, which a Mistral-format, text-only deployment (--config-format mistral --load-format mistral
--language-model-only) never builds. This file restores the old class name and installs a helper
that raises if it is ever called, so no vision computation can run on a substitute. It is mounted
through PYTHONPATH into that one serving container only (scripts/serve_model.sh, VLLM_SHIM);
engine, transformers and weights are the image's own.
"""

try:
    from transformers.models.pixtral import modeling_pixtral as _mp
except Exception:  # transformers absent or re-organised: leave every import untouched
    _mp = None

if _mp is not None:
    if not hasattr(_mp, "PixtralRotaryEmbedding") and hasattr(_mp, "PixtralVisionRotaryEmbedding"):
        _mp.PixtralRotaryEmbedding = _mp.PixtralVisionRotaryEmbedding
    if not hasattr(_mp, "position_ids_in_meshgrid"):

        def position_ids_in_meshgrid(*args, **kwargs):
            raise NotImplementedError("the Pixtral vision encoder is not served in this deployment")

        _mp.position_ids_in_meshgrid = position_ids_in_meshgrid
