import itertools
import numpy as np
import pandas as pd
from time import time
import cvxopt.solvers
import numpy.linalg as la
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import json
import string
from time import time
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class Kernel(object):

    @staticmethod
    def linear():
        return lambda x, y: np.dot(x.T, y)

    @staticmethod
    def polykernel(dimension, offset):
        return lambda x, y: ((offset + np.dot(x.T, y)) ** dimension)

    @staticmethod
    def radial_basis(gamma):
        return lambda x, y: np.exp(-gamma * la.norm(np.subtract(x, y)))

class SVMTrainer(object):

    def __init__(self, kernel, c):
        self.kernel = kernel
        self.c = c

    def train(self, X, y):
        lagrange_multipliers = self.compute_multipliers(X, y)
        return self.construct_predictor(X, y, lagrange_multipliers)

    def kernel_matrix(self, X, n_samples):
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
        return K

    def construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self.kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])
        return SVMPredictor(
            kernel=self.kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def compute_multipliers(self, X, y):
        n_samples, n_features = X.shape
        K = self.kernel_matrix(X, n_samples)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])

class SVMPredictor(object):

    def __init__(self, kernel, bias, weights, support_vectors, support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)

    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights, self._support_vectors, self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()

def calculate(true_positive, false_positive, false_negative, true_negative):
    result = {}
    result['precision'] = true_positive / (true_positive + false_positive)
    result['recall'] = true_positive / (true_positive + false_negative)
    result['accuracy'] = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    return result

def confusion_matrix(true_positive, false_positive, false_negative, true_negative):
    matrix = PrettyTable([' ', 'Ham', 'Spam'])
    matrix.add_row(['Ham', true_positive, false_positive])
    matrix.add_row(['Spam', false_negative, true_negative])
    return matrix, calculate(true_positive, false_positive, false_negative, true_negative)

def implementSVM(X_train, Y_train, X_test, Y_test, parameters, type):
    ham_spam = 0
    spam_spam = 0
    ham_ham = 0
    spam_ham = 0
    if (type == "polykernel"):
        dimension = parameters['dimension']
        offset = parameters['offset']
        trainer = SVMTrainer(Kernel.polykernel(dimension, offset), 1)
        predictor = trainer.train(X_train, Y_train)
        save_model_to_json(predictor, "polykernel_model.json")
    elif (type == "linear"):
        trainer = SVMTrainer(Kernel.linear(), 1)
        predictor = trainer.train(X_train, Y_train)
        save_model_to_json(predictor, "linear_model.json")
    for i in range(X_test.shape[0]):
        ans = predictor.predict(X_test[i])
        if(ans==1 and Y_test[i]==1):
            spam_spam+=1
        elif(ans==-1 and Y_test[i]==1):
            spam_ham+=1
        elif(ans==-1 and Y_test[i]==-1):
            ham_ham+=1
        elif(ans==1 and Y_test[i]==-1):
            ham_spam+=1
    return confusion_matrix(ham_ham,ham_spam,spam_ham,spam_spam)

def write_to_file(matrix, result, parameters, type, start_time):
    f = open("results1.txt", "a")
    if (type == "polykernel"):
        f.write("Polykernel model parameters")
        f.write("\n")
        f.write("Dimension : " + str(parameters['dimension']))
        f.write("\n")
        f.write("Offset : " + str(parameters['offset']))
    elif (type == "linear"):
        f.write("Linear model")
    f.write("\n")
    f.write(matrix.get_string())
    f.write("\n")
    f.write("Precision : " + str(round(result['precision'], 2)))
    f.write("\n")
    f.write("Recall : " + str(round(result['recall'], 2)))
    f.write("\n")
    f.write("Accuracy : " + str(round(result['accuracy'], 2)))
    f.write("\n")
    f.write("Time spent for model : " + str(round(time() - start_time, 2)))
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()

def save_model_to_json(predictor, filename):
    model_data = {
        "bias": predictor._bias,
        "weights": predictor._weights.tolist(),
        "support_vectors": predictor._support_vectors.tolist(),
        "support_vector_labels": predictor._support_vector_labels.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(model_data, f)

def load_model_from_json(filename, kernel):
    with open(filename, 'r') as f:
        model_data = json.load(f)
    return SVMPredictor(
        kernel=kernel,
        bias=model_data["bias"],
        weights=np.array(model_data["weights"]),
        support_vectors=np.array(model_data["support_vectors"]),
        support_vector_labels=np.array(model_data["support_vector_labels"])
    )

global_start_time = time()

cvxopt.solvers.options['show_progress'] = False

df1 = pd.read_csv('frequency1.csv', header=0)
input_output = df1.to_numpy()
X = input_output[:, :-1]
Y = input_output[:, -1:]

total = X.shape[0]
train = int(X.shape[0] * 70 / 100)

X_train = X[:train, :]
Y_train = Y[:train, :]
X_test = X[train:, :]
Y_test = Y[train:, :]

f = open("results1.txt", "w+")
f.close()
k = 0

type = {}
parameters = {}

type['1'] = "polykernel"
type['2'] = "linear"

theta = 2
d = 1

parameters['dimension'] = theta
parameters['offset'] = d

print("Menu")
print("1. svm tuyến tính")
print("2. svm phi tuyến")
print("3. svm tuyến tính với với model sau khi train và tập test.csv tự tạo")
print("4. svm phi tuyến với model sau khi train và tập test.csv tự tạo")
print("5.svm tuyến tính với với model sau khi train với email người dùng nhập vào")
print("6.svm phi với với model sau khi train với email người dùng nhập vào")
choose = int(input("Nhập lựa chọn (1, 2, 3, 4, 5 hoặc 6): "))
if (choose == 1):
    print("Đang xử lí...")
    start_time = time()
    matrix, result = implementSVM(X_train, Y_train, X_test, Y_test, parameters, str(type['2']))
    write_to_file(matrix, result, parameters, type, start_time)
    k += 1
    print("Done : kết quả với tập test mặc định được lưu vào file result1.txt")

if (choose == 2):
    print("Đang xử lí...")
    start_time = time()
    matrix, result = implementSVM(X_train, Y_train, X_test, Y_test, parameters, str(type['1']))
    write_to_file(matrix, result, parameters, type, start_time)
    k += 1
    print("Done : kết quả với tập test mặc định được lưu vào file result1.txt")

if (choose == 3):
    print("Đang xử lí...")
    predictor = load_model_from_json("linear_model.json", Kernel.linear())

    test_df = pd.read_csv("test.csv", header=0)
    test_data = test_df.to_numpy()
    X_test = test_data[:, :-1]  # Lấy các cột đặc trưng
    Y_test = test_data[:, -1:]  # Lấy cột nhãn
    ham_spam = 0
    spam_spam = 0
    ham_ham = 0
    spam_ham = 0

    for i in range(X_test.shape[0]):
        ans = predictor.predict(X_test[i])
        if ans == 1 and Y_test[i] == 1:
            spam_spam += 1
        elif ans == -1 and Y_test[i] == 1:
            spam_ham += 1
        elif ans == -1 and Y_test[i] == -1:
            ham_ham += 1
        elif ans == 1 and Y_test[i] == -1:
            ham_spam += 1

    # Bước 4: Tính toán các chỉ số
    matrix, result = confusion_matrix(ham_ham, ham_spam, spam_ham, spam_spam)

    # Bước 5: Ghi kết quả vào file `result.txt`
    with open("result.txt", "w",encoding="utf-8") as f:
        f.write("Kết quả kiểm tra với tập test.csv\n")
        f.write(matrix.get_string())
        f.write("\n")
        f.write("Precision : " + str(round(result['precision'], 2)) + "\n")
        f.write("Recall : " + str(round(result['recall'], 2)) + "\n")
        f.write("Accuracy : " + str(round(result['accuracy'], 2)) + "\n")
    print("Hoàn thành! Kết quả được lưu vào result.txt.")

if (choose == 4):
    print("Đang xử lí...")
    predictor = load_model_from_json("polykernel_model.json", Kernel.polykernel(parameters['dimension'], parameters['offset']))

    # Đọc dữ liệu test từ test.csv
    test_df = pd.read_csv("test.csv", header=0)
    test_data = test_df.to_numpy()
    X_test = test_data[:, :-1]  # Lấy các cột đặc trưng
    Y_test = test_data[:, -1:]  # Lấy cột nhãn

    # Các biến đếm
    ham_spam = 0
    spam_spam = 0
    ham_ham = 0
    spam_ham = 0

    # Dự đoán từng mẫu trong tập test
    for i in range(X_test.shape[0]):
        ans = predictor.predict(X_test[i])
        if ans == 1 and Y_test[i] == 1:
            spam_spam += 1
        elif ans == -1 and Y_test[i] == 1:
            spam_ham += 1
        elif ans == -1 and Y_test[i] == -1:
            ham_ham += 1
        elif ans == 1 and Y_test[i] == -1:
            ham_spam += 1

    # Tính toán ma trận nhầm lẫn và các chỉ số
    matrix, result = confusion_matrix(ham_ham, ham_spam, spam_ham, spam_spam)

    # Ghi kết quả vào file result.txt
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write("Kết quả kiểm tra với tập test.csv\n")
        f.write(matrix.get_string())
        f.write("\n")
        f.write("Precision : " + str(round(result['precision'], 2)) + "\n")
        f.write("Recall : " + str(round(result['recall'], 2)) + "\n")
        f.write("Accuracy : " + str(round(result['accuracy'], 2)) + "\n")
    print("Hoàn thành! Kết quả được lưu vào result.txt.")

if (choose == 5):
    email = input("enter your email:")
    email = email.encode('ascii', 'ignore').decode()  # Bỏ ký tự không ASCII
    # Đọc danh sách từ từ file CSV
    df_words = pd.read_csv('wordslist1.csv', header=0)
    words = df_words['word']

    # Tạo lemmatizer
    lmtzr = WordNetLemmatizer()

    # Khởi tạo mảng tần suất từ
    words_list_array = np.zeros(words.size,dtype=int)

    for word in email.split():
        word = lmtzr.lemmatize(word.lower())
        if (word in stopwords.words('english') or word in string.punctuation 
                or len(word) <= 2 or word.isdigit()):
            continue
        for i, w in enumerate(words):
            if w == word:
                words_list_array[i] += 1
                break

    predictor = load_model_from_json("linear_model.json", Kernel.linear())
    ans = predictor.predict(words_list_array)
    if ans == 1:
        print("this email is spam")
    if ans == -1:
        print("this email is not spam")

if (choose == 6):
    email = input("enter your email:")
    email = email.encode('ascii', 'ignore').decode()  # Bỏ ký tự không ASCII
    # Đọc danh sách từ từ file CSV
    df_words = pd.read_csv('wordslist1.csv', header=0)
    words = df_words['word']

    # Tạo lemmatizer
    lmtzr = WordNetLemmatizer()

    # Khởi tạo mảng tần suất từ
    words_list_array = np.zeros(words.size,dtype=int)

    for word in email.split():
        word = lmtzr.lemmatize(word.lower())
        if (word in stopwords.words('english') or word in string.punctuation 
                or len(word) <= 2 or word.isdigit()):
            continue
        for i, w in enumerate(words):
            if w == word:
                words_list_array[i] += 1
                break
    
    predictor = load_model_from_json("polykernel_model.json", Kernel.polykernel(parameters['dimension'], parameters['offset']))
    ans = predictor.predict(words_list_array)
    if ans == 1:
        print("this email is spam")
    if ans == -1:
        print("this email is not spam")
        
f = open("results1.txt", "a")
f.write("Time spent for entire code : " + str(round(time() - global_start_time, 2)))
f.close()
