import re
import Levenshtein  # pip install python-Levenshtein

VALID_WORDS = {
    "ноль", "один", "два", "три", "четыре", "пять",
    "шесть", "семь", "восемь", "девять", "десять",
    "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
    "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
    "сто", "двести", "триста", "четыреста",
    "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот",
    "тысяча", "тысячи", "тысяч",
    "миллион", "миллиона", "миллионов"
}

def correct_with_dict(pred: str) -> str:
    words = pred.split()
    corrected = []
    for w in words:
        if w in VALID_WORDS:
            corrected.append(w)
        else:
            # Find closest valid word
            best = min(VALID_WORDS, key=lambda v: Levenshtein.distance(w, v))
            if Levenshtein.distance(w, best) <= 2:   # threshold
                corrected.append(best)
            else:
                corrected.append(w)   # keep original if far
    return " ".join(corrected)


class DigitToRussian:
    """Convert digit string to Russian number words (nominative case, no "and")"""

    def __init__(self):
        self.units = [
            "",
            "один",
            "два",
            "три",
            "четыре",
            "пять",
            "шесть",
            "семь",
            "восемь",
            "девять",
        ]
        self.units_female = [
            "",
            "одна",
            "две",
            "три",
            "четыре",
            "пять",
            "шесть",
            "семь",
            "восемь",
            "девять",
        ]
        self.teens = [
            "десять",
            "одиннадцать",
            "двенадцать",
            "тринадцать",
            "четырнадцать",
            "пятнадцать",
            "шестнадцать",
            "семнадцать",
            "восемнадцать",
            "девятнадцать",
        ]
        self.tens = [
            "",
            "",
            "двадцать",
            "тридцать",
            "сорок",
            "пятьдесят",
            "шестьдесят",
            "семьдесят",
            "восемьдесят",
            "девяносто",
        ]
        self.hundreds = [
            "",
            "сто",
            "двести",
            "триста",
            "четыреста",
            "пятьсот",
            "шестьсот",
            "семьсот",
            "восемьсот",
            "девятьсот",
        ]

    def _convert_three_digits(self, num, female=False):
        """Convert 0-999 to words. If female=True, use female forms for 1,2."""
        if num == 0:
            return ""

        h = num // 100
        t = (num % 100) // 10
        u = num % 10

        parts = []
        if h > 0:
            parts.append(self.hundreds[h])

        if t == 1:
            # Teens
            parts.append(self.teens[u])
        else:
            if t > 1:
                parts.append(self.tens[t])
            if u > 0:
                # Choose unit form: female for thousands (тысяча) or default
                if female and u <= 2:
                    parts.append(self.units_female[u])
                else:
                    parts.append(self.units[u])
        return " ".join(parts)

    def convert(self, digit_str):
        """Convert digit string to Russian words."""
        num = int(digit_str)
        if num == 0:
            return "ноль"

        parts = []

        # Millions (if needed)
        millions = num // 1000000
        if millions > 0:
            parts.append(self._convert_three_digits(millions))
            if millions == 1:
                parts.append("миллион")
            elif 2 <= millions <= 4:
                parts.append("миллиона")
            else:
                parts.append("миллионов")
            num %= 1000000

        # Thousands
        thousands = num // 1000
        if thousands > 0:
            # For thousands, use female forms for 1,2 (одна/две)
            parts.append(self._convert_three_digits(thousands, female=True))
            if thousands == 1:
                parts.append("тысяча")
            elif 2 <= thousands <= 4:
                parts.append("тысячи")
            else:
                parts.append("тысяч")
            num %= 1000

        # Hundreds/tens/units
        if num > 0:
            parts.append(self._convert_three_digits(num, female=False))

        return " ".join(parts)


class BaseTokenizer:
    """Common interface for all tokenizers. pad_id is always 0 (doubles as CTC blank)."""

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
        """Join a list of decoded token strings back into text."""
        raise NotImplementedError

    def __len__(self):
        return len(self.vocab)


class NumericTokenizer(BaseTokenizer):
    """Semantic numeric token vocabulary.

    Each token represents one atomic numeric concept:
      units 1-9 | teens 10-19 | tens 20,30,...,90 | hundreds 100,...,900 | 1000

    Vocab size: 2 special + 9 + 10 + 8 + 9 + 1 = 39 tokens.

    encode() accepts EITHER a digit string ("331401") OR Russian text — no external
    DigitToRussian needed. join() returns a digit string directly — no external
    RussianToDigitLevenshtein needed (RussianToDigitLevenshtein.convert() passes
    digit strings through unchanged).
    """

    _VOCAB_NUMS: list[int] = (
        list(range(1, 10)) +            # units  1-9
        list(range(10, 20)) +           # teens  10-19
        list(range(20, 100, 10)) +      # tens   20,30,...,90
        list(range(100, 1000, 100)) +   # hundreds 100,...,900
        [1000]
    )

    _WORD_TO_NUM: dict[str, int] = {
        "один": 1, "одна": 1, "два": 2, "две": 2,
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

    _NUM_TO_WORD: dict[int, str] = {
        1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
        6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
        10: "десять", 11: "одиннадцать", 12: "двенадцать", 13: "тринадцать",
        14: "четырнадцать", 15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать",
        18: "восемнадцать", 19: "девятнадцать",
        20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят",
        60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто",
        100: "сто", 200: "двести", 300: "триста", 400: "четыреста",
        500: "пятьсот", 600: "шестьсот", 700: "семьсот", 800: "восемьсот", 900: "девятьсот",
        1000: "тысяча",
    }

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

    def encode(self, text: str) -> list[int]:
        """Accept either a digit string ("331401") or Russian text."""
        text = text.strip()
        if text.isdigit():
            # Direct encoding — no Russian intermediate needed
            n = int(text)
            nums: list[int] = []
            thousands = n // 1000
            if thousands:
                nums.extend(self._decompose(thousands))
                nums.append(1000)
            remainder = n % 1000
            if remainder:
                nums.extend(self._decompose(remainder))
            return [self.token2id.get(str(x), self.unk_id) for x in nums]
        # Fallback: Russian text (e.g. from RussianSpeechDataset via DigitToRussian)
        ids = []
        for w in text.lower().split():
            num = self._WORD_TO_NUM.get(w)
            ids.append(self.token2id.get(str(num), self.unk_id) if num else self.unk_id)
        return ids

    def decode(self, ids, skip_special=True) -> str:
        """Return the digit string directly."""
        return self.join([
            self.id2token[i] for i in ids
            if not skip_special or i not in (self.pad_id, self.unk_id)
        ])

    def join(self, tokens: list[str]) -> str:
        """Sum numeric tokens → digit string. No Russian text intermediate."""
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
    def __init__(self, word_vocab=None):
        base_vocab = [
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
        if word_vocab:
            for w in word_vocab:
                if w not in base_vocab:
                    base_vocab.append(w)
        super().__init__(base_vocab)
        self.sos_id = self.token2id["<sos>"]
        self.eos_id = self.token2id["<eos>"]

    def encode(self, text):
        return [self.token2id.get(w, self.unk_id) for w in text.lower().split()]

    def decode(self, ids, skip_special=True):
        special = {self.pad_id, self.sos_id, self.eos_id}
        tokens = []
        for i in ids:
            if skip_special and i in special:
                continue
            tokens.append(self.id2token.get(i, "<unk>"))
        return " ".join(tokens)

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

    def encode(self, text):
        ids = []
        for ch in text.lower():
            if ch == " ":
                ids.append(self.space_id)
            else:
                ids.append(self.token2id.get(ch, self.unk_id))
        return ids

    def decode(self, ids, skip_special=True):
        chars = []
        for i in ids:
            if skip_special and i == self.pad_id:
                continue
            if i == self.space_id:
                chars.append(" ")
            else:
                chars.append(self.id2token.get(i, ""))
        return "".join(chars)

    def join(self, tokens: list[str]) -> str:
        return "".join(" " if t == self.SPACE_TOKEN else t for t in tokens)


class RussianToDigit:
    """Convert Russian number words to digit string."""

    def __init__(self):
        self.word_to_val = {
            "ноль": 0,
            "один": 1,
            "одна": 1,
            "два": 2,
            "две": 2,
            "три": 3,
            "четыре": 4,
            "пять": 5,
            "шесть": 6,
            "семь": 7,
            "восемь": 8,
            "девять": 9,
            "десять": 10,
            "одиннадцать": 11,
            "двенадцать": 12,
            "тринадцать": 13,
            "четырнадцать": 14,
            "пятнадцать": 15,
            "шестнадцать": 16,
            "семнадцать": 17,
            "восемнадцать": 18,
            "девятнадцать": 19,
            "двадцать": 20,
            "тридцать": 30,
            "сорок": 40,
            "пятьдесят": 50,
            "шестьдесят": 60,
            "семьдесят": 70,
            "восемьдесят": 80,
            "девяносто": 90,
            "сто": 100,
            "двести": 200,
            "триста": 300,
            "четыреста": 400,
            "пятьсот": 500,
            "шестьсот": 600,
            "семьсот": 700,
            "восемьсот": 800,
            "девятьсот": 900,
            "тысяча": 1000,
            "тысячи": 1000,
            "тысяч": 1000,
            "миллион": 1000000,
            "миллиона": 1000000,
            "миллионов": 1000000,
        }

    def convert(self, text):
        words = text.lower().split()
        total = 0
        current = 0
        multiplier = 1
        i = 0
        while i < len(words):
            w = words[i]
            if w in ("тысяча", "тысячи", "тысяч"):
                if current == 0:
                    current = 1
                total += current * 1000
                current = 0
            elif w in ("миллион", "миллиона", "миллионов"):
                if current == 0:
                    current = 1
                total += current * 1000000
                current = 0
            else:
                val = self.word_to_val.get(w, 0)
                if val >= 100:
                    current += val
                elif val >= 20:
                    current += val
                elif val >= 10:
                    current += val
                else:
                    current += val
            i += 1
        total += current
        return str(total)

class RussianToDigitLevenshtein:
    def __init__(self, correction_threshold=2):
        self.word_to_val = {
            "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2,
            "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
            "восемь": 8, "девять": 9,
            "десять": 10,
            "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
            "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
            "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
            "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
            "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
            "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
            "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
            "девятьсот": 900,
            "тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
            "миллион": 1_000_000, "миллиона": 1_000_000, "миллионов": 1_000_000,
        }
        self.valid_words = set(self.word_to_val.keys())
        self.threshold = correction_threshold

    def _correct_word(self, w: str) -> str:
        if w in self.valid_words:
            return w
        # find closest valid word
        best = min(self.valid_words, key=lambda v: Levenshtein.distance(w, v))
        if Levenshtein.distance(w, best) <= self.threshold:
            return best
        return w  # unchanged, will be rejected later

    def convert(self, text: str) -> str | None:
        text = text.strip()
        if text.isdigit():
            return text   # NumericTokenizer already produced a digit string
        words = text.lower().split()
        if not words:
            return ""
        # Correct each word
        corrected = [self._correct_word(w) for w in words]
        total = 0
        current = 0
        i = 0
        while i < len(corrected):
            w = corrected[i]
            if w in ("тысяча", "тысячи", "тысяч"):
                if current == 0:
                    current = 1
                total += current * 1000
                current = 0
                i += 1
                continue
            if w in ("миллион", "миллиона", "миллионов"):
                if current == 0:
                    current = 1
                total += current * 1_000_000
                current = 0
                i += 1
                continue
            val = self.word_to_val.get(w)
            if val is None:
                val = 0
                # return None
            current += val
            i += 1
        total += current
        return str(total)

def prepare_targets():
    import pandas as pd

    converter = DigitToRussian()

    df = pd.read_csv("train.csv", sep=",")
    df["spoken"] = df["transcription"].apply(lambda x: converter.convert(str(x)))
    # Now df has columns: filename, transcription (digits), spoken (Russian words)
    pd.to_csv(df, "train_prepared.csv", sep=",", index=False)
