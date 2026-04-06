import re


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


class RussianWordTokenizer:
    def __init__(self, word_vocab=None):
        # Base vocabulary (special tokens + all possible number words)
        base_vocab = [
            "<pad>",
            "<sos>",
            "<eos>",
            "<unk>",
            "ноль",
            "один",
            "два",
            "три",
            "четыре",
            "пять",
            "шесть",
            "семь",
            "восемь",
            "девять",
            "одна",
            "две",
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
            "двадцать",
            "тридцать",
            "сорок",
            "пятьдесят",
            "шестьдесят",
            "семьдесят",
            "восемьдесят",
            "девяносто",
            "сто",
            "двести",
            "триста",
            "четыреста",
            "пятьсот",
            "шестьсот",
            "семьсот",
            "восемьсот",
            "девятьсот",
            "тысяча",
            "тысячи",
            "тысяч",
            "миллион",
            "миллиона",
            "миллионов",
        ]
        if word_vocab:
            # Add any extra words from the training data
            for w in word_vocab:
                if w not in base_vocab:
                    base_vocab.append(w)
        self.vocab = base_vocab
        self.token2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2token = {i: w for w, i in self.token2id.items()}
        self.pad_id = self.token2id["<pad>"]
        self.sos_id = self.token2id["<sos>"]
        self.eos_id = self.token2id["<eos>"]
        self.unk_id = self.token2id["<unk>"]

    def encode(self, text):
        words = text.lower().split()
        return [self.token2id.get(w, self.unk_id) for w in words]

    def decode(self, ids, skip_special=True):
        tokens = []
        for i in ids:
            if skip_special and i in (self.pad_id, self.sos_id, self.eos_id):
                continue
            tokens.append(self.id2token.get(i, "<unk>"))
        return " ".join(tokens)

    def __len__(self):
        return len(self.vocab)


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


def prepare_targets():
    import pandas as pd

    converter = DigitToRussian()

    df = pd.read_csv("train.csv", sep=",")
    df["spoken"] = df["transcription"].apply(lambda x: converter.convert(str(x)))
    # Now df has columns: filename, transcription (digits), spoken (Russian words)
    pd.to_csv(df, "train_prepared.csv", sep=",", index=False)
