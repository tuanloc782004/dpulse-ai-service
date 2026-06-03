import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Load biến môi trường từ file .env
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")

# Khởi tạo kết nối
try:
    client = MongoClient(MONGO_URI)
    db = client.get_default_database() # Tự động lấy tên DB từ chuỗi URI
    print("✅ Đã kết nối MongoDB thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối MongoDB: {e}")

def get_all_approved_services():
    """Lấy tất cả dịch vụ đã được duyệt để train model"""
    return list(db.services.find({"approvalStatus": "APPROVED"}))

def get_user_data(user_id: str):
    """Lấy thông tin User (preferences)"""
    return db.users.find_one({"_id": ObjectId(user_id)})

def get_user_bookings(user_id: str):
    """Lấy danh sách ID dịch vụ user đã đặt thành công"""
    bookings = list(db.bookings.find({
        "userId": ObjectId(user_id),
        "status": {"$in": ["PAID", "COMPLETED"]}
    }))
    return [str(b["serviceId"]) for b in bookings]

def get_user_wishlist(user_id: str):
    """Lấy danh sách ID dịch vụ user đã lưu"""
    wishlists = list(db.wishlists.find({"userId": ObjectId(user_id)}))
    return [str(w["serviceId"]) for w in wishlists]