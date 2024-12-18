# machine-learning-G17

chạy mô hình:

download các thư viện cần thiết

cd svm

py createWordsList.py

py createFrequency.py

py svmModel.py

Trong file svmModel.py có thể lựa chọn hàm nhân tuyến tính hoặc hàm nhân đa thức

    -trong hàm nhân đa thức,có thể thay đổi 2 chỉ số theta và d (đọc báo cáo mục 3.1 trang 11)
    
    -trong class SVMTrainer:

        def __init__(self, kernel, c)
        
        c chính là mức độ phạt (penalty degree) đối với các lỗi (đọc báo cáo mục 3.1 trang 9)

    -chạy svmModel.py với menu 1 hoặc 2 sẽ tạo ra file result(thời gian chạy từ 3 đến 5 phút) chứa thông tin: precision, recall ,accuracy , đồng thời lưu model đã train vào file json.

    -test model đã train bằng menu 3 hoặc 4

