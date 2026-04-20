import re

import Levenshtein

# Single source of truth for Russian word → numeric value
WORD_TO_VAL: dict[str, int] = {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2,
    "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
}

_UNITS    = {1:"один",2:"два",3:"три",4:"четыре",5:"пять",6:"шесть",7:"семь",8:"восемь",9:"девять"}
_UNITS_F  = {1:"одна",2:"две",3:"три",4:"четыре",5:"пять",6:"шесть",7:"семь",8:"восемь",9:"девять"}
_TEENS    = {10:"десять",11:"одиннадцать",12:"двенадцать",13:"тринадцать",14:"четырнадцать",
             15:"пятнадцать",16:"шестнадцать",17:"семнадцать",18:"восемнадцать",19:"девятнадцать"}
_TENS     = {2:"двадцать",3:"тридцать",4:"сорок",5:"пятьдесят",6:"шестьдесят",7:"семьдесят",8:"восемьдесят",9:"девяносто"}
_HUNDREDS = {1:"сто",2:"двести",3:"триста",4:"четыреста",5:"пятьсот",6:"шестьсот",7:"семьсот",8:"восемьсот",9:"девятьсот"}


def _three_digits(n: int, female: bool = False) -> list[str]:
    if n == 0:
        return []
    parts = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if 10 <= rem <= 19:
        parts.append(_TEENS[rem])
    else:
        t, u = divmod(rem, 10)
        if t >= 2:
            parts.append(_TENS[t])
        if u:
            parts.append((_UNITS_F if female else _UNITS)[u])
    return parts


def digit_to_russian(digit_str: str) -> str:
    """Convert digit string to Russian number words (nominative case)."""
    n = int(digit_str)
    if n == 0:
        return "ноль"
    parts = []
    if n >= 1_000_000:
        m, n = divmod(n, 1_000_000)
        parts.extend(_three_digits(m))
        parts.append("миллион" if m == 1 else "миллиона" if 2 <= m <= 4 else "миллионов")
    if n >= 1000:
        k, n = divmod(n, 1000)
        parts.extend(_three_digits(k, female=True))
        parts.append("тысяча" if k == 1 else "тысячи" if 2 <= k <= 4 else "тысяч")
    if n:
        parts.extend(_three_digits(n))
    return " ".join(parts)


class BaseTokenizer:
    """Common interface for all tokenizers. pad_id=0 doubles as CTC blank."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.token2id = {w: i for i, w in enumerate(vocab)}
        self.id2token = {i: w for w, i in self.token2id.items()}
        self.pad_id = self.token2id["<pad>"]
        self.unk_id = self.token2id["<unk>"]

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids, skip_special=True) -> str:
        raise NotImplementedError

    def join(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def label_from_digits(self, digit_str: str) -> list[int]:
        """Convert a digit string to token IDs for training labels."""
        return self.encode(digit_to_russian(digit_str))

    def __len__(self):
        return len(self.vocab)


class NumericTokenizer(BaseTokenizer):
    """Semantic numeric tokens: units 1-9 | teens 10-19 | tens | hundreds | 1000.

    Vocab size: 2 special + 9 + 10 + 8 + 9 + 1 = 39.
    encode() takes digit strings directly — no Russian intermediate.
    join() returns a digit string directly — no Russian-to-digit conversion needed.
    """

    _VOCAB_NUMS: list[int] = (
        list(range(1, 10)) +
        list(range(10, 20)) +
        list(range(20, 100, 10)) +
        list(range(100, 1000, 100)) +
        [1000]
    )

    def __init__(self):
        super().__init__(["<pad>", "<unk>"] + [str(n) for n in self._VOCAB_NUMS])

    def _decompose(self, n: int) -> list[int]:
        """Break 1-999 into additive numeric tokens."""
        parts = []
        h = (n // 100) * 100
        rem = n % 100
        if h:
            parts.append(h)
        if 10 <= rem <= 19:
            parts.append(rem)
        else:
            t = (rem // 10) * 10
            u = rem % 10
            if t:
                parts.append(t)
            if u:
                parts.append(u)
        return parts

    def encode(self, digit_str: str) -> list[int]:
        n = int(digit_str.strip())
        nums: list[int] = []
        thousands = n // 1000
        if thousands:
            nums.extend(self._decompose(thousands))
            nums.append(1000)
        remainder = n % 1000
        if remainder:
            nums.extend(self._decompose(remainder))
        return [self.token2id.get(str(x), self.unk_id) for x in nums]

    def label_from_digits(self, digit_str: str) -> list[int]:
        return self.encode(digit_str)

    def decode(self, ids, skip_special=True) -> str:
        return self.join([
            self.id2token[i] for i in ids
            if not skip_special or i not in (self.pad_id, self.unk_id)
        ])

    def join(self, tokens: list[str]) -> str:
        """Sum numeric tokens → digit string."""
        total = current = 0
        for t in tokens:
            if t in ("<pad>", "<unk>"):
                continue
            try:
                n = int(t)
            except ValueError:
                continue
            if n == 1000:
                total += (current or 1) * 1000
                current = 0
            else:
                current += n
        return str(total + current)


class RussianWordTokenizer(BaseTokenizer):
    def __init__(self):
        vocab = [
            "<pad>", "<sos>", "<eos>", "<unk>",
            "ноль", "один", "два", "три", "четыре", "пять",
            "шесть", "семь", "восемь", "девять", "одна", "две",
            "десять", "одиннадцать", "двенадцать", "тринадцать",
            "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
            "восемнадцать", "девятнадцать",
            "двадцать", "тридцать", "сорок", "пятьдесят",
            "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
            "сто", "двести", "триста", "четыреста",
            "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот",
            "тысяча", "тысячи", "тысяч",
        ]
        super().__init__(vocab)
        self.sos_id = self.token2id["<sos>"]
        self.eos_id = self.token2id["<eos>"]

    def encode(self, text: str) -> list[int]:
        return [self.token2id.get(w, self.unk_id) for w in text.lower().split()]

    def decode(self, ids, skip_special=True) -> str:
        special = {self.pad_id, self.sos_id, self.eos_id}
        return " ".join(
            self.id2token.get(i, "<unk>")
            for i in ids
            if not skip_special or i not in special
        )

    def join(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class RussianCharTokenizer(BaseTokenizer):
    """Character-level tokenizer over Russian letters + space."""

    SPACE_TOKEN = "<space>"

    def __init__(self):
        vocab = ["<pad>", "<unk>", self.SPACE_TOKEN]
        vocab.extend(chr(c) for c in range(ord("а"), ord("я") + 1))
        super().__init__(vocab)
        self.space_id = self.token2id[self.SPACE_TOKEN]

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text.lower():
            if ch == " ":
                ids.append(self.space_id)
            else:
                ids.append(self.token2id.get(ch, self.unk_id))
        return ids

    def decode(self, ids, skip_special=True) -> str:
        chars = []
        for i in ids:
            if skip_special and i == self.pad_id:
                continue
            chars.append(" " if i == self.space_id else self.id2token.get(i, ""))
        return "".join(chars)

    def join(self, tokens: list[str]) -> str:
        return "".join(" " if t == self.SPACE_TOKEN else t for t in tokens)


_CONSONANTS = "бвгджзйклмнпрстфхцчшщ"


def normalize_for_ctc(text: str) -> str:
    """Acoustic normalization for CTC training on Russian numbers.

    Rules (applied in order):
      1. тысячи / тысяч → тысяча   (acoustically identical forms)
      2. дцать → цат               (двадцать → двацат, пятнадцать → пятнацат)
      3. Remove ь and ъ
      4. Collapse double consonants (нн → н, тт → т, …)
    """
    # 1. Normalize thousand forms at word boundaries
    words = text.split()
    words = ["тысяча" if w in ("тысячи", "тысяч") else w for w in words]
    text = " ".join(words)
    # 2. Simplify -дцать suffix
    text = text.replace("дцать", "цат")
    # 3. Remove soft and hard signs
    text = text.replace("ь", "").replace("ъ", "")
    # 4. Collapse double consonants
    text = re.sub(rf"([{_CONSONANTS}])\1+", r"\1", text)
    return text


def _build_normalized_word_to_val() -> dict[str, int]:
    """WORD_TO_VAL extended with normalized aliases. Used only by NormalizedRussianToDigit."""
    extended = dict(WORD_TO_VAL)
    for w, v in WORD_TO_VAL.items():
        nw = normalize_for_ctc(w)
        if nw not in extended:
            extended[nw] = v
    return extended


class NormalizedCharTokenizer(BaseTokenizer):
    """Char-level CTC tokenizer with acoustic normalization for Russian numbers (1 000–999 999).

    Applies normalize_for_ctc() before encoding, shrinking the label alphabet from
    ~32 Cyrillic chars down to ~18 acoustically distinct chars.
    Vocab is built from the normalized forms of all Russian number words.
    """

    SPACE_TOKEN = "<space>"

    def __init__(self):
        chars = sorted({
            ch
            for word in WORD_TO_VAL
            for ch in normalize_for_ctc(word)
            if ch != " "
        })
        vocab = ["<pad>", "<unk>", self.SPACE_TOKEN] + chars
        super().__init__(vocab)
        self.space_id = self.token2id[self.SPACE_TOKEN]

    def label_from_digits(self, digit_str: str) -> list[int]:
        return self.encode(digit_to_russian(digit_str))

    def encode(self, text: str) -> list[int]:
        text = normalize_for_ctc(text.lower())
        ids = []
        for ch in text:
            if ch == " ":
                ids.append(self.space_id)
            else:
                ids.append(self.token2id.get(ch, self.unk_id))
        return ids

    def decode(self, ids, skip_special=True) -> str:
        chars = []
        for i in ids:
            if skip_special and i == self.pad_id:
                continue
            chars.append(" " if i == self.space_id else self.id2token.get(i, ""))
        return "".join(chars)

    def join(self, tokens: list[str]) -> str:
        return "".join(" " if t == self.SPACE_TOKEN else t for t in tokens)


class RussianToDigitLevenshtein:
    """Convert Russian number words to digit string, with Levenshtein word correction."""

    def __init__(self, correction_threshold: int = 2):
        self.valid_words = set(WORD_TO_VAL.keys())
        self.threshold = correction_threshold

    def _correct_word(self, w: str) -> str:
        if w in self.valid_words:
            return w
        best = min(self.valid_words, key=lambda v: Levenshtein.distance(w, v))
        return best if Levenshtein.distance(w, best) <= self.threshold else w

    def convert(self, text: str) -> str:
        text = text.strip()
        if text.isdigit():
            return text
        words = [self._correct_word(w) for w in text.lower().split()]
        total = current = 0
        for w in words:
            if w in ("тысяча", "тысячи", "тысяч"):
                total += (current or 1) * 1000
                current = 0
            elif w in ("миллион", "миллиона", "миллионов"):
                total += (current or 1) * 1_000_000
                current = 0
            else:
                current += WORD_TO_VAL.get(w, 0)
        return str(total + current)


class NormalizedRussianToDigit:
    """Converter for NormalizedCharTokenizer output.

    Uses an extended lookup that includes normalized word forms (e.g. 'двацат' → 20).
    Levenshtein correction operates over normalized valid words only.
    """

    def __init__(self, correction_threshold: int = 2):
        self._word_to_val = _build_normalized_word_to_val()
        self.valid_words = set(self._word_to_val.keys())
        self.threshold = correction_threshold

    def _correct_word(self, w: str) -> str:
        if w in self.valid_words:
            return w
        best = min(self.valid_words, key=lambda v: Levenshtein.distance(w, v))
        return best if Levenshtein.distance(w, best) <= self.threshold else w

    def convert(self, text: str) -> str:
        text = text.strip()
        if text.isdigit():
            return text
        words = [self._correct_word(w) for w in text.lower().split()]
        total = current = 0
        for w in words:
            if w == "тысяча":
                total += (current or 1) * 1000
                current = 0
            else:
                current += self._word_to_val.get(w, 0)
        return str(total + current)
