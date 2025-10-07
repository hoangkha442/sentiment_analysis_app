# Sentiment Analysis App (EN + VI)

## Giới thiệu

Ứng dụng web phân tích cảm xúc văn bản hỗ trợ tiếng Anh và tiếng Việt. Sử dụng mô hình PhoBERT cho tiếng Việt và DistilBERT (SST-2) cho tiếng Anh. Người dùng nhập nội dung, chọn ngôn ngữ (hoặc tự động nhận diện), hệ thống trả về kết quả cảm xúc (tích cực, tiêu cực, trung tính) cùng độ tin cậy.

## Thành viên nhóm

| MSSV    | Họ và Tên        |
| ------- | ---------------- |
| 2591306 | Châu Hoàng Kha   |
| 2591314 | Trần Thị Bảo My  |
| 2591320 | Nguyễn Thành Quí |

## Cấu trúc thư mục

```
main.py
render.yaml
requirements.txt
static/
    main.js
    style.css
templates/
    index.html
```

## Hướng dẫn chạy

### 1. clone website

```sh
git clone https://github.com/hoangkha442/sentiment_analysis_app.git
```

```sh
cd sentiment_analysis_app
```

### 2. Cài đặt môi trường

- Yêu cầu Python 3.10+
- Cài đặt các thư viện cần thiết:

```sh
pip install -r requirements.txt
```

### 3. Chạy ứng dụng Flask

```sh
python main.py
```

Ứng dụng sẽ chạy tại địa chỉ: [http://localhost:7860](http://localhost:7860)

### 4. Sử dụng

- Truy cập trang web.
- Nhập nội dung cần phân tích.
- Chọn ngôn ngữ hoặc để chế độ tự động.
- Nhấn "Phân tích" để xem kết quả.
