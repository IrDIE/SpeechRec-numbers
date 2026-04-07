"""CTC decoders: greedy, beam search, and LM-fused beam search."""
import tempfile

import torch


class GreedyDecoder:
    """Argmax + collapse repeats + remove blanks. Base class for all decoders."""

    def __init__(self, tokenizer, blank_id: int | None = None):
        self.tokenizer = tokenizer
        self.blank_id = tokenizer.pad_id if blank_id is None else blank_id

    def decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[str]]:
        preds = log_probs.argmax(dim=-1)
        out: list[list[str]] = []
        for seq, length in zip(preds, lengths):
            ids, collapsed, prev = seq[: int(length)].tolist(), [], None
            for tok in ids:
                if tok != prev and tok != self.blank_id:
                    collapsed.append(tok)
                prev = tok
            out.append([self.tokenizer.id2token[i] for i in collapsed])
        return out


class BeamSearchDecoder(GreedyDecoder):
    """Lexicon-free beam search via torchaudio (no LM)."""

    def __init__(self, tokenizer, beam_size: int = 50):
        super().__init__(tokenizer)
        from torchaudio.models.decoder import ctc_decoder

        self._decoder = ctc_decoder(
            lexicon=None,
            tokens=list(tokenizer.vocab),
            lm=None,
            beam_size=beam_size,
            blank_token=tokenizer.id2token[self.blank_id],
            sil_token=tokenizer.id2token[self.blank_id],
            unk_word="<unk>",
        )

    def decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[str]]:
        results = self._decoder(log_probs.cpu(), lengths.cpu())
        return [
            [
                self.tokenizer.id2token[int(t)]
                for t in hyps[0].tokens
                if int(t) != self.blank_id
            ]
            for hyps in results
        ]


class LMBeamSearchDecoder(BeamSearchDecoder):
    """Lexicon + KenLM-fused beam search via torchaudio."""

    def __init__(
        self,
        tokenizer,
        lm_path: str,
        beam_size: int = 50,
        lm_weight: float = 0.5,
        word_score: float = 0.0,
    ):
        # Skip BeamSearchDecoder.__init__; build a different ctc_decoder.
        GreedyDecoder.__init__(self, tokenizer)
        from torchaudio.models.decoder import ctc_decoder

        lex = tempfile.NamedTemporaryFile("w", suffix=".lex", delete=False)
        for w in tokenizer.vocab:
            if not w.startswith("<"):
                lex.write(f"{w} {w}\n")
        lex.close()

        self._decoder = ctc_decoder(
            lexicon=lex.name,
            tokens=list(tokenizer.vocab),
            lm=lm_path,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            blank_token=tokenizer.id2token[self.blank_id],
            sil_token=tokenizer.id2token[self.blank_id],
            unk_word="<unk>",
        )

    def decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[str]]:
        results = self._decoder(log_probs.cpu(), lengths.cpu())
        return [list(hyps[0].words) for hyps in results]
