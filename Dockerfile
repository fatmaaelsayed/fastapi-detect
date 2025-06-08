# صورة Python 3.12 slim
FROM python:3.12-slim

# تثبيت المكتبات المطلوبة لتشغيل OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# تحديد مجلد العمل داخل الـ container
WORKDIR /app

# نسخ ملفات المشروع
COPY . /app

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# فتح البورت اللي Uvicorn بيشتغل عليه
EXPOSE 8000

# أمر التشغيل
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
