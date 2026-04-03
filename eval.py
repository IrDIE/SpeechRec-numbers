
import torch
from data_processor.postprocessor import RussianToDigit

def eval(model):
    predicted_words = model.decode(...)
    digit_normalizer = RussianToDigit()
    digits = digit_normalizer.convert(predicted_words)
    print(digits)  # "13003"