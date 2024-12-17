import os
import nltk
import time
import string
import operator
import numpy as np
import csv
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def text_cleanup(text):
    # Loại bỏ dấu câu
    text_without_punctuation = [c for c in text if c not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)
    
    # Loại bỏ stopwords
    text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
    text_without_stopwords = ' '.join(text_without_stopwords)
    
    # Chuyển tất cả các từ về chữ thường
    cleaned_text = [word.lower() for word in text_without_stopwords.split()]
    return cleaned_text

start_time = time.time()

lmtzr = WordNetLemmatizer()
k = 0
count = {}

# Đọc dữ liệu từ cleaned.csv
with open("cleaned_file.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # Lấy văn bản từ cột 'Message'
        message = row['Message']
        
        # Làm sạch và xử lý văn bản
        words = text_cleanup(message)
        
        # Đếm số lần xuất hiện của từ đã lemmatize
        for word in words:
            if not word.isdigit() and len(word) > 2:  # Loại bỏ từ số và từ quá ngắn
                word = lmtzr.lemmatize(word)
                if word in count:
                    count[word] += 1
                else:
                    count[word] = 1
        
        k += 1
        if k % 100 == 0:
            print("Đã xử lý " + str(k) + " email(s)")

# Sắp xếp các từ theo số lần xuất hiện giảm dần
sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
sorted_count = dict(sorted_count)

# Ghi kết quả vào file wordslist.csv
with open("wordslist1.csv", mode="w+", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    
    for word, times in sorted_count.items():
        if times < 30:  # Dừng lại khi số lần xuất hiện của từ nhỏ hơn 100
            break
        writer.writerow([word, times])

print('Thời gian xử lý các email (tính bằng giây): ' + str(round(time.time() - start_time, 2)))