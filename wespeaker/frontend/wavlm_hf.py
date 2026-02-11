# wespeaker/frontend/wavlm_hf.py
from __future__ import annotations

import torch
from torch import nn

try:
    from transformers import WavLMModel
except Exception as e:
    WavLMModel = None
    _IMPORT_ERR = e


class WavLMHFFrontend(nn.Module):
    """
    HuggingFace WavLM frontend that returns frame-level features:
      input:  wavs (B, T), wav_lens (B,)
      output: feats (B, T', D), feats_lens (B,)  (feats_lens may be None if not computable)
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
        output_layer: int = -1,          # -1 = last hidden state
        freeze: bool = False,            # freeze all wavlm params if True
        use_attention_mask: bool = True, # mask padding using wav_lens
    ):
        super().__init__()
        if WavLMModel is None:
            raise ImportError(f"Failed to import transformers.WavLMModel: {_IMPORT_ERR}")

        self.model_name = model_name
        self.output_layer = int(output_layer)
        self.use_attention_mask = bool(use_attention_mask)

        self.wavlm = WavLMModel.from_pretrained(model_name)

        if freeze:
            for p in self.wavlm.parameters():
                p.requires_grad = False
            self.wavlm.eval()

        self._hidden = int(self.wavlm.config.hidden_size)

    def output_size(self) -> int:
        return self._hidden

    def _make_attention_mask(self, wav_lens: torch.Tensor, T: int) -> torch.Tensor:
        # wav_lens: (B,) in samples
        # returns attention_mask: (B, T) boolean/int
        ar = torch.arange(T, device=wav_lens.device).unsqueeze(0)  # (1,T)
        mask = ar < wav_lens.unsqueeze(1)                          # (B,T)
        return mask

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor | None = None):
        """
        wavs: (B, T) float32/float16 in [-1, 1] (typical audio)
        wav_lens: (B,) lengths in samples
        """
        if wavs.dim() != 2:
            raise ValueError(f"Expected wavs shape (B,T). Got {tuple(wavs.shape)}")

        attn = None
        if self.use_attention_mask and (wav_lens is not None):
            attn = self._make_attention_mask(wav_lens.long(), wavs.size(1)).to(torch.long)

        # Ask HF to return hidden states if we want an intermediate layer
        need_all = (self.output_layer != -1)

        out = self.wavlm(
            input_values=wavs,
            attention_mask=attn,
            output_hidden_states=need_all,
            return_dict=True,
        )

        if self.output_layer == -1:
            feats = out.last_hidden_state  # (B, T', D)
        else:
            feats = out.hidden_states[self.output_layer]  # (B, T', D)

        # Optional: estimate output lengths if method exists (safe best-effort)
        feats_lens = None
        if wav_lens is not None:
            if hasattr(self.wavlm, "_get_feat_extract_output_lengths"):
                with torch.no_grad():
                    feats_lens = self.wavlm._get_feat_extract_output_lengths(wav_lens.long())
            # else: leave None

        return feats, feats_lens
