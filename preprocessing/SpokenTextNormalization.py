from typing import List
import pandas as pd
import regex as re


# Dict các từ viết tắt và thay thế của chúng
CLEAN_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "I'm": "I am",
    "isn't": "is not",
    "aren't": "are not",
    "it's": "it is",
    "I'll": "I will",
    "we'll": "we will",
    "don't": "do not",
    "didn't": "did not",
    "ca n't": "can not",
    "wo not": "will not",
    "n't": "not",
    "could've": "could have",
    "i'm": "i am",
    "i've": "i have",
    "might've": "might have",
    "must've": "must have",
    "shan't": "shall not",
    "should've": "should have",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we're": "we are",
    "we've": "we have",
    "what're": "what are",
    "what've": "what have",
    "who're": "who are",
    "who've": "who have",
    "would've": "would have",
    "you're": "you are",
    "you've": "you have",
    "gonna": "going to",
    "gon'na": "going to",
    "gon na": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "wan'na": "want to",
    "wan na": "want to",
    "hafta": "have to",
    "hadta": "had to",
    "shoulda": "should have",
    "woulda": "would have",
    "coulda": "could have",
    "mighta": "might have",
    "musta": "must have",
    "oughta": "ought to",
    "dont": "do not",
    "doesnt": "does not"
}

# List các từ ấp úng (hesitation words)
FILLER_WORDS = ['uh', 'um', 'er', 'ah', 'well', 'you know', 'I mean', 'like', 'hmm']


# Viết laại các từ viết tắt ở dạng đầy đủ của nó
def write_out_words(text: str, contractions: dict) -> str:
    for k, v in contractions.items():
        #text = text.replace(k, v)
        text = re.sub(r'\b' + k + r'\b', v, text)
    return text


# Loại bỏ các từ ấp úng
def remove_filler_words(text: str, fillers: List[str]) -> str:
    for filler in fillers:
        text = re.sub(r'\b' + filler + r'\b', '', text) #\buhm\b: Matches the whole word "uhm".
    return text


# Loại bỏ các từ trùng lặp liên tiếp nhau dùng regex
# Giải thích chi tiết công thức regex:
# r'\b: \b là ký hiệu biên giới từ, nghĩa là nó sẽ khớp với vị trí giữa một ký tự từ (như chữ cái, số) và một ký tự không phải từ (như khoảng trắng, dấu câu).
#
# (\w+): \w khớp với bất kỳ ký tự từ nào (chữ cái, chữ số, hoặc dấu gạch dưới). Dấu + cho biết khớp với một hoặc nhiều ký tự từ. Toàn bộ phần này được đặt trong ngoặc đơn để tạo thành một nhóm bắt (capturing group), cho phép chúng ta tham chiếu đến từ này sau này trong công thức regex. Nhóm bắt này được đánh số 1 (vì là nhóm đầu tiên).
#
# (?:\W+\1\b)+:
#
# (?: ... ): Đây là nhóm không bắt (non-capturing group), nghĩa là nó không lưu lại kết quả khớp như nhóm bắt.
# \W+: \W khớp với bất kỳ ký tự không phải từ nào (khoảng trắng, dấu câu, vv.). Dấu + cho biết khớp với một hoặc nhiều ký tự không phải từ.
# \1: Tham chiếu lại đến nhóm bắt đầu tiên (từ đầu tiên).
# \b: Ký hiệu biên giới từ.
# +: Cho biết rằng phần không bắt này có thể lặp lại một hoặc nhiều lần. Điều này có nghĩa là nếu từ đầu tiên lặp lại nhiều lần liên tiếp, toàn bộ cụm từ sẽ khớp.
#
# Ví dụ minh họa:
# Giả sử chuỗi text là "hello hello world world world". Công thức regex sẽ hoạt động như sau:
#
# \b(\w+): Nhóm bắt đầu tiên sẽ bắt từ "hello".
# (?:\W+\1\b)+: Nhóm không bắt sẽ khớp với khoảng trắng và sau đó là từ "hello" một lần nữa.
# Sau đó nó lại khớp với từ "world world world" bằng cách bắt từ "world" và sau đó là khoảng trắng và "world" nhiều lần.
# Kết quả là, tất cả các từ lặp lại liên tiếp sẽ được thay thế bằng từ duy nhất đầu tiên.

def remove_consecutive_duplicates(text: str) -> str:
    return re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', text)


# Hàm xử lý một đoạn hội thoại trong 1 dòng DataFrame (1 dòng hội thoại trong DataFrame)
def spoken_text_normalization_string(text: str, contractions: dict, fillers: List[str]) -> str:
    text = write_out_words(text, contractions)
    text = remove_filler_words(text, fillers)
    text = remove_consecutive_duplicates(text)
    return text


# Hàm xử lý toàn bộ DataFrame
def spoken_text_normalization(_df: pd.DataFrame, dialogue_column: str, contractions: dict, fillers: List[str]) -> pd.DataFrame:
    assert dialogue_column in _df.columns, f"Expected column '{dialogue_column}' in the dataframe"

    # Hàm xử lý từng dòng trong DataFrame
    def spoken_text_normalization_for_each_row(row):
        row['clean_dialogue'] = spoken_text_normalization_string(row[dialogue_column], contractions, fillers)
        return row

    _df = _df.apply(spoken_text_normalization_for_each_row, axis=1)
    return _df


# Ví dụ
if __name__ == "__main__":
    input_file = "../preprocessing/input.csv"  # Adjust the path as necessary
    df = pd.read_csv(input_file, quotechar='"', skipinitialspace=True, on_bad_lines='skip')

    print("before:")
    print(df[['dialogue']])

    df_cleaned = spoken_text_normalization(df, "dialogue", CLEAN_CONTRACTIONS, FILLER_WORDS)

    print("after:")
    print(df_cleaned[['clean_dialogue']])

    output_file = 'cleaned_output.csv'
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
