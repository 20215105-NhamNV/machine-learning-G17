import string
import numpy as np
import pandas as pd
from time import time
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Ghi nhận thời gian bắt đầu
start_time = time()

# Đọc danh sách từ từ file CSV
df_words = pd.read_csv('wordslist1.csv', header=0)
words = df_words['word']

# Tạo lemmatizer
lmtzr = WordNetLemmatizer()

# Đọc file clean.csv
df_emails = pd.read_csv('cleaned_file.csv', header=0)

# Tạo file frequency1.csv và ghi header
with open("frequency1.csv", "w") as f:
    for i in words:
        f.write(str(i) + ',')
    f.write('output\n')

# Duyệt qua từng email trong file clean.csv
for index, row in df_emails.iterrows():
    category = row['Category']
    message = row['Message']
    
    # Khởi tạo mảng tần suất từ
    words_list_array = np.zeros(words.size)
    
    # Xử lý nội dung email
    for word in message.split():
        word = lmtzr.lemmatize(word.lower())
        if (word in stopwords.words('english') or word in string.punctuation 
                or len(word) <= 2 or word.isdigit()):
            continue
        for i in range(words.size):
            if words[i] == word:
                words_list_array[i] += 1
                break

    # Ghi tần suất từ và nhãn vào file frequency1.csv
    with open("frequency1.csv", "a") as f:
        for count in words_list_array:
            f.write(str(int(count)) + ',')
        f.write("-1\n" if category == 0 else "1\n")

    # Theo dõi tiến trình xử lý
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} emails")

# In tổng thời gian xử lý
print(f"Time (in seconds) to process the dataset: {round(time() - start_time, 2)}")
