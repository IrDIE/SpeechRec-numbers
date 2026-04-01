import re

class RussianNumberTokenizer:
    """
    Tokenizer for Russian numbers built from scratch.
    No pretrained components - vocabulary defined based on your dataset.
    """
    
    def __init__(self, transcripts=None):
        """
        Build vocabulary from training transcripts if provided,
        or use standard Russian number vocabulary.
        """
        # Standard Russian number vocabulary
        self.base_vocab = [
            "<pad>", "<sos>", "<eos>", "<unk>",
            # Units
            "ноль", "один", "одна", "одно", "два", "две", "три", "четыре",
            "пять", "шесть", "семь", "восемь", "девять",
            # Teens
            "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
            "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
            # Tens
            "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
            "семьдесят", "восемьдесят", "девяносто",
            # Hundreds
            "сто", "двести", "триста", "четыреста", "пятьсот",
            "шестьсот", "семьсот", "восемьсот", "девятьсот",
            # Thousands
            "тысяча", "тысячи", "тысяч",
            # Millions
            "миллион", "миллиона", "миллионов"
        ]
        
        # If transcripts provided, expand vocabulary with any missing words
        if transcripts is not None:
            all_words = set()
            for text in transcripts:
                words = text.lower().split()
                all_words.update(words)
            
            # Add any new words not in base vocabulary
            for word in all_words:
                if word not in self.base_vocab:
                    self.base_vocab.append(word)
        
        # Create mappings
        self.vocab = self.base_vocab
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
        self.pad_id = self.token2id["<pad>"]
        self.sos_id = self.token2id["<sos>"]
        self.eos_id = self.token2id["<eos>"]
        self.unk_id = self.token2id["<unk>"]
    
    def encode(self, text):
        """Convert text to token IDs"""
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            if word in self.token2id:
                token_ids.append(self.token2id[word])
            else:
                token_ids.append(self.unk_id)
        
        return token_ids
    
    def decode(self, token_ids, skip_special=True):
        """Convert token IDs back to text"""
        tokens = []
        for idx in token_ids:
            if skip_special and idx in [self.pad_id, self.sos_id, self.eos_id]:
                continue
            tokens.append(self.id2token.get(idx, "<unk>"))
        
        return " ".join(tokens)
    
    def __len__(self):
        return len(self.vocab)

  
class RussianNumberNormalizer:
    """Convert Russian spoken numbers to digit format"""
    
    def __init__(self):
        # Mapping dictionaries
        self.digits = {
            "ноль": "0", "один": "1", "одна": "1", "одно": "1",
            "два": "2", "две": "2", "три": "3", "четыре": "4",
            "пять": "5", "шесть": "6", "семь": "7", "восемь": "8",
            "девять": "9"
        }
        
        self.teens = {
            "десять": "10", "одиннадцать": "11", "двенадцать": "12",
            "тринадцать": "13", "четырнадцать": "14", "пятнадцать": "15",
            "шестнадцать": "16", "семнадцать": "17", "восемнадцать": "18",
            "девятнадцать": "19"
        }
        
        self.tens = {
            "двадцать": "20", "тридцать": "30", "сорок": "40",
            "пятьдесят": "50", "шестьдесят": "60", "семьдесят": "70",
            "восемьдесят": "80", "девяносто": "90"
        }
        
        self.hundreds = {
            "сто": "100", "двести": "200", "триста": "300",
            "четыреста": "400", "пятьсот": "500", "шестьсот": "600",
            "семьсот": "700", "восемьсот": "800", "девятьсот": "900"
        }
        
        # Combine all for lookup
        self.word_to_num = {}
        self.word_to_num.update(self.digits)
        self.word_to_num.update(self.teens)
        self.word_to_num.update(self.tens)
        self.word_to_num.update(self.hundreds)
    
    def parse_number(self, words):
        """Parse a sequence of words into a number"""
        result = 0
        current = 0
        multiplier = 1
        
        i = 0
        while i < len(words):
            word = words[i].lower()
            
            # Handle thousands
            if word in ["тысяча", "тысячи", "тысяч"]:
                if current == 0:
                    current = 1
                result += current * 1000
                current = 0
                multiplier = 1
                i += 1
                continue
            
            # Handle hundreds
            elif word in self.hundreds:
                current += int(self.hundreds[word])
                i += 1
                continue
            
            # Handle tens
            elif word in self.tens:
                current += int(self.tens[word])
                i += 1
                continue
            
            # Handle teens
            elif word in self.teens:
                current += int(self.teens[word])
                i += 1
                continue
            
            # Handle digits
            elif word in self.digits:
                # Check if next word is a teen or tens?
                if i + 1 < len(words) and words[i + 1] in self.tens:
                    # This is a compound like "двадцать один"
                    tens_val = int(self.tens[words[i + 1]])
                    digit_val = int(self.digits[word])
                    current += tens_val + digit_val
                    i += 2
                elif i + 1 < len(words) and words[i + 1] in self.teens:
                    # Handle like "пятнадцать" already covered
                    pass
                else:
                    current += int(self.digits[word])
                    i += 1
                continue
            
            else:
                # Unknown word, treat as separator
                if current > 0:
                    result += current
                    current = 0
                i += 1
        
        result += current
        return str(result)
    
    def normalize(self, text):
        """Convert full text with Russian numbers to digit format"""
        # Split into words
        words = text.lower().split()
        
        # Find number sequences
        result_words = []
        i = 0
        while i < len(words):
            # Check if current word could be part of a number
            if words[i] in self.word_to_num or words[i] in ["тысяча", "тысячи", "тысяч", "миллион", "миллиона"]:
                # Collect all consecutive number-related words
                number_words = []
                while i < len(words) and (
                    words[i] in self.word_to_num or 
                    words[i] in ["тысяча", "тысячи", "тысяч", "миллион", "миллиона", "миллионов"]
                ):
                    number_words.append(words[i])
                    i += 1
                
                # Parse the number sequence
                parsed_number = self.parse_number(number_words)
                result_words.append(parsed_number)
            else:
                result_words.append(words[i])
                i += 1
        
        return " ".join(result_words)