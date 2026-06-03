# Sử dụng Python 3.10 làm môi trường gốc
FROM python:3.10-slim

# Tạo thư mục làm việc
WORKDIR /app

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào Docker
COPY . .

# Chạy server FastAPI ở cổng 7860 (Cổng bắt buộc của Hugging Face)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]