import sys
from functools import cached_property

from aenum import extend_enum
from vllm.config import ModelConfig as _OriginalModelConfig
from vllm.inputs import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus
from vllm.v1.request import StreamingUpdate as _OriginalStreamingUpdate

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding
from vllm_omni.request import OmniRequest, OmniStreamingUpdate

# =============================================================================
# Patch ModelConfig.is_mm_prefix_lm to support omni-specific models
# =============================================================================
# WHY: HunyuanImage-3.0 requires bidirectional attention for image tokens
# (cond_token_attn_type: "joint_full" in config.json). vLLM gates this on
# is_mm_prefix_lm, which checks an internal MM_PREFIX_LM_MODELS list that
# does not include "hunyuan_image_3_moe" (the upstream HF model_type).
#
# WHY NOT model-level: is_mm_prefix_lm is checked in vLLM core (scheduler,
# attention backend selection) before model code runs — no model-level hook.
#
# SCOPE: Only affects model_type in _OMNI_MM_PREFIX_LM_MODELS (currently
# just "hunyuan_image_3_moe"). All other models fall through to the
# original vLLM implementation unchanged.
#
# FRAGILITY: Relies on is_mm_prefix_lm being a cached_property on
# ModelConfig. The __dict__ access + __set_name__ dance works around a
# pydantic dataclass issue in vllm 0.19.0+. If vLLM changes
# is_mm_prefix_lm to a regular method or removes it, this will break.
#
# TODO: Upstream a configurable MM_PREFIX_LM_MODELS or a model_config flag
# so this patch can be removed.
_OMNI_MM_PREFIX_LM_MODELS = ("hunyuan_image_3_moe",)
# Access via __dict__ to avoid triggering cached_property.__get__ which fails
# with "Cannot use cached_property instance without calling __set_name__" in
# pydantic dataclasses (vllm 0.19.0+).
_cp = _OriginalModelConfig.__dict__["is_mm_prefix_lm"]
_original_is_mm_prefix_lm = _cp.func if hasattr(_cp, "func") else _cp.fget


def _patched_is_mm_prefix_lm(self):
    if _original_is_mm_prefix_lm(self):
        return True
    model_type = getattr(self.hf_config, "model_type", "")
    return model_type in _OMNI_MM_PREFIX_LM_MODELS


_patched_cp = cached_property(_patched_is_mm_prefix_lm)
_patched_cp.__set_name__(_OriginalModelConfig, "is_mm_prefix_lm")
_OriginalModelConfig.is_mm_prefix_lm = _patched_cp

# Sanity check: verify the patch is active. If vLLM changes the descriptor
# type or __set_name__ semantics, this will fail loudly at import time
# rather than silently falling back to unpatched behavior.
_installed = _OriginalModelConfig.__dict__.get("is_mm_prefix_lm")
assert _installed is _patched_cp, (
    "is_mm_prefix_lm patch failed to install — bidirectional attention "
    "for HunyuanImage3 will not work. Check vLLM ModelConfig changes."
)

# =============================================================================
# Patch ModelOptNvFp4FusedMoE to handle W4A16_NVFP4 MoE checkpoints
# =============================================================================
# WHY: vLLM 0.21.0's ModelOptNvFp4FusedMoE.__init__ unconditionally passes
# activation_key=kNvfp4Dynamic to select_nvfp4_moe_backend, which forces a
# W4A4 backend (CUTLASS / FlashInfer / TRT-LLM). When the on-disk checkpoint
# is W4A16_NVFP4 (no input_scale tensors, BF16 activations), the chosen
# backend's create_weights allocates W4A4-shaped params, and the W4A16
# packed weights from the safetensors file fail the shape assertion in
# vllm.model_executor.layers.linear.weight_loader with messages like
# "Tried to load weights of size [128, 2048] to a parameter of size
# [128, 1024]". Loading any Qwen3-MoE-style W4A16_NVFP4 checkpoint dies at
# worker init.
#
# Upstream fix landed as vllm PR #42566 (merged 2026-05-22, after v0.21.0):
# pass activation_key=None when quant_method=="W4A16_NVFP4", forcing the
# only backend (Marlin) whose _supports_quant_scheme tolerates absent
# activation quantization. Marlin's MoE path drops activation scales in
# convert_to_nvfp4_moe_kernel_format, so the rest of the init is unchanged.
#
# This patch backports that one behavioral change to whatever pre-fix vllm
# version vllm-omni is pinned against.
#
# WHY NOT model-level: ModelOptNvFp4FusedMoE is selected by vLLM's
# ModelOptNvFp4Config.get_quant_method when the layer is a FusedMoE — model
# code never instantiates it directly and has no hook to override.
#
# SCOPE: Affects only ModelOptNvFp4FusedMoE init. Dense W4A16 (via
# ModelOptNvFp4W4A16LinearMethod) is unaffected — it already works in
# v0.21.0. Mixed-precision FP8+W4A16 dispatch (the other half of PR #42566)
# is not backported — vllm-omni's quantization factory does not currently
# accept MIXED_PRECISION checkpoints that span both FP8 and W4A16 layers.
#
# FRAGILITY: Relies on (a) select_nvfp4_moe_backend continuing to accept
# activation_key=None and select Marlin in that case, and (b) Marlin's MoE
# kernel continuing to ignore the absent activation scales. Both are stable
# on the relevant code paths in v0.21.0–main. If a future vLLM minor
# refactors the backend-selection contract (e.g. renames the activation_key
# kwarg or returns a different backend tuple), the failure surfaces at the
# first W4A16_NVFP4 MoE layer instantiation (worker init), not at module
# import — see the install-time `_already_patched_upstream` probe below as
# the only import-time signal.
#
# TODO: Remove once vllm-omni bumps its vllm pin to a release that contains
# #42566 (expected vllm 0.22+). The `_already_patched_upstream` check below
# is intended to detect that case at install time and skip our backport so
# we don't double-apply the fix.
try:
    # kNvfp4Static moved between vllm versions: in v0.21.0 it lives in
    # `quantization.utils.quant_utils`; in some later versions it's re-exported
    # from `fused_moe.oracle.nvfp4`. Try the new path first, fall back.
    try:
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            kNvfp4Static,
        )
    except ImportError:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kNvfp4Static,
        )
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE as _OriginalModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        is_global_sf_supported_for_nvfp4_backend,
    )
except ImportError as _w4a16_patch_import_err:
    # Patch surface not importable in this build — log loudly. Silent passing
    # bit us 2026-05-25 when an import path differed between vllm versions
    # and the W4A16 MoE checkpoint silently fell through to the broken W4A4
    # backend selection. Use vllm-omni's logger so the warning lands in the
    # standard worker stderr, not a separate channel.
    import logging as _logging

    _logging.getLogger("vllm_omni.patch").warning(
        "W4A16_NVFP4 MoE backport patch could NOT install: %s. "
        "Loading W4A16_NVFP4 MoE checkpoints will likely fail with a "
        "weight-loader shape mismatch at worker init. Check vllm version "
        "and import paths.",
        _w4a16_patch_import_err,
    )
else:
    _original_nvfp4_fused_moe_init = _OriginalModelOptNvFp4FusedMoE.__init__
    # If upstream already shipped the use_a16 flag (vllm post-#42566), the
    # backport is redundant. `self.use_a16 = ...` is compiled as STORE_ATTR
    # which stores the attribute name in __code__.co_names (NOT co_consts
    # or co_varnames). Match the exact name to avoid substring false
    # positives from unrelated docstrings/messages mentioning "use_a16".
    _already_patched_upstream = "use_a16" in (
        _original_nvfp4_fused_moe_init.__code__.co_names or ()
    )

    def _patched_nvfp4_fused_moe_init(self, quant_config, moe_config, *args, **kwargs):
        _original_nvfp4_fused_moe_init(self, quant_config, moe_config, *args, **kwargs)
        # If upstream is patched, the original __init__ already set use_a16
        # correctly — don't double-apply.
        if _already_patched_upstream:
            return
        self.use_a16 = getattr(quant_config, "quant_method", None) == "W4A16_NVFP4"
        if not self.use_a16:
            return
        # Re-run backend selection with activation_key=None. The W4A4
        # backends all reject this; only Marlin survives. Marlin handles
        # W4A16 MoE correctly.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=None,
        )
        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
        )

    _OriginalModelOptNvFp4FusedMoE.__init__ = _patched_nvfp4_fused_moe_init

# =============================================================================
# Patch GlmImageTextConfig to expose mrope_section in rope_parameters
# =============================================================================
# GLM-Image uses M-RoPE with mrope_section: [8, 12, 12], but transformers'
# implementation doesn't expose it in rope_parameters. vLLM's uses_mrope
# detection relies on "mrope_section" being present in rope_parameters.
# This patch ensures proper M-RoPE detection for GLM-Image.
try:
    from transformers.models.glm_image.configuration_glm_image import GlmImageTextConfig

    _original_glm_image_text_config_init = GlmImageTextConfig.__init__

    def _patched_glm_image_text_config_init(self, *args, **kwargs):
        _original_glm_image_text_config_init(self, *args, **kwargs)
        # Ensure rope_parameters exists and contains mrope_section
        if self.rope_parameters is None:
            self.rope_parameters = {}
        if isinstance(self.rope_parameters, dict) and "mrope_section" not in self.rope_parameters:
            # GLM-Image uses mrope_section: [8, 12, 12] for T/H/W dimensions
            self.rope_parameters["mrope_section"] = [8, 12, 12]

    GlmImageTextConfig.__init__ = _patched_glm_image_text_config_init
except ImportError:
    # GlmImageTextConfig not available, skip patching
    pass

# Extend RequestStatus enum with omni-specific statuses
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    # The value - 1 is intentionally chosen to ensure it is treated
    # as a non-finished state and remains compatible with existing comparisons.
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)

if not hasattr(RequestStatus, "WAITING_FOR_INPUT"):
    # Full-payload stage handoff uses a distinct waiting state so the
    # scheduler can restore the request once non-stage-0 inputs arrive.
    extend_enum(RequestStatus, "WAITING_FOR_INPUT", -2)

# Snapshot sys.modules: `hasattr` below can trigger lazy submodule imports
# (e.g. transformers' `_LazyModule.__getattr__`), which mutate sys.modules
# during iteration and raise `dictionary changed size during iteration`.
for module_name, module in list(sys.modules.items()):
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = OmniMRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "StreamingUpdate") and module.StreamingUpdate == _OriginalStreamingUpdate:
        module.StreamingUpdate = OmniStreamingUpdate
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest
