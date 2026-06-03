from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from database import get_all_approved_services, get_user_data, get_user_bookings, get_user_wishlist
from recommender import train_tfidf_model, get_recommendations

# --- HÀM TỰ ĐỘNG CẬP NHẬT MA TRẬN ---
def refresh_ai_matrix():
    print("🔄 Bắt đầu kéo dữ liệu mới từ MongoDB và nạp lại Ma trận...")
    services = get_all_approved_services()
    train_tfidf_model(services)

# --- QUẢN LÝ VÒNG ĐỜI SERVER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Chạy ngay khi Server vừa bật (Khởi tạo In-Memory)
    refresh_ai_matrix()
    
    # 2. Cài đặt Background Job (Mỗi 60 phút chạy lại 1 lần)
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_ai_matrix, 'interval', minutes=60)
    scheduler.start()
    
    yield # Server bắt đầu nhận request
    
    # 3. Chạy khi Server tắt
    scheduler.shutdown()

# --- API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Recommendation Engine is running!"}

@app.get("/recommend/{user_id}")
def recommend_services(user_id: str, limit: int = 6):
    try:
        # 1. Lấy dữ liệu User
        user = get_user_data(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        preferences = user.get("preferences", [])
        booked_ids = get_user_bookings(user_id)
        wishlist_ids = get_user_wishlist(user_id)

        # 2. Đưa vào hàm tính toán
        results = get_recommendations(
            user_id=user_id,
            preferences=preferences,
            booked_ids=booked_ids,
            wishlist_ids=wishlist_ids,
            top_k=limit
        )

        return {
            "success": True,
            "data": results,
            "user_stats": {
                "preferences_count": len(preferences),
                "bookings_count": len(booked_ids),
                "wishlist_count": len(wishlist_ids)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))