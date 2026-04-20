"""CTC decoders: greedy, beam search, and LM-fused beam search."""
import math
import tempfile

import torch

NEG_INF = float("-inf")


def _lse(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


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


class ConstrainedBeamDecoder:
    """CTC beam search restricted to valid Russian number sequences (1–max_num).

    Builds a trie by encoding every valid phrase with the tokenizer, so it works
    with both RussianCharTokenizer (trie over char IDs) and RussianWordTokenizer
    (trie over word IDs).
    """

    def __init__(self, tokenizer, beam_size: int = 50, max_num: int = 999_999):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.blank_id = tokenizer.pad_id
        self._trie = self._build_trie(tokenizer, max_num)

    @staticmethod
    def _build_trie(tokenizer, max_num: int) -> dict:
        from data_processor.postprocessor import DigitToRussian
        converter = DigitToRussian()
        trie: dict = {}
        for n in range(1, max_num + 1):
            node = trie
            for tok in tokenizer.encode(converter.convert(n)):
                node = node.setdefault(tok, {})
            node[None] = True  # complete valid sequence marker
        return trie

    def decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[str]]:
        return [
            self._decode_one(log_probs[b, : int(lengths[b])].tolist())
            for b in range(log_probs.size(0))
        ]

    def _decode_one(self, log_probs: list[list[float]]) -> list[str]:
        # beam: prefix (token IDs) -> (trie_node, log_p_blank, log_p_nonblank)
        beams: dict[tuple, tuple] = {(): (self._trie, 0.0, NEG_INF)}

        for lp in log_probs:
            next_beams: dict[tuple, list] = {}

            for prefix, (node, p_b, p_nb) in beams.items():
                p_total = _lse(p_b, p_nb)
                last = prefix[-1] if prefix else None

                # Blank: prefix stays the same
                e = next_beams.setdefault(prefix, [node, NEG_INF, NEG_INF])
                e[1] = _lse(e[1], p_total + lp[self.blank_id])

                # Non-blank: only tokens allowed by trie
                for tok, child in node.items():
                    if tok is None:
                        continue
                    log_p = lp[tok]
                    if log_p == NEG_INF:
                        continue
                    # CTC: repeating last token only extends via blank path
                    add = (p_b if tok == last else p_total) + log_p
                    e = next_beams.setdefault(prefix + (tok,), [child, NEG_INF, NEG_INF])
                    e[2] = _lse(e[2], add)

            # Prune to beam_size
            beams = {
                k: tuple(v)
                for k, v in sorted(
                    next_beams.items(),
                    key=lambda kv: _lse(kv[1][1], kv[1][2]),
                    reverse=True,
                )[: self.beam_size]
            }

        # Prefer complete valid sequences (trie leaf), else fall back to best beam
        candidates = [
            (p, _lse(pb, pnb))
            for p, (node, pb, pnb) in beams.items()
            if None in node
        ] or [
            (p, _lse(pb, pnb)) for p, (node, pb, pnb) in beams.items()
        ]
        best = max(candidates, key=lambda x: x[1])[0]
        return [self.tokenizer.id2token[i] for i in best]
