from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import pandas as pd

# BIẾN TOÀN CỤC (IN-MEMORY) - Giúp tính toán phản hồi trong 0.01 giây
tfidf_vectorizer = None
tfidf_matrix = None
services_df = None # Dùng Pandas DataFrame để tra cứu nhanh

def preprocess_text(text: str):
    """Tiền xử lý tiếng Việt: Tách từ ghép (VD: 'khách sạn' -> 'khách_sạn')"""
    if not text or not isinstance(text, str):
        return ""
    # Underthesea giúp AI hiểu từ ghép tiếng Việt cực tốt
    tokens = word_tokenize(text.lower(), format="text")
    return tokens

def train_tfidf_model(services_data):
    """Huấn luyện Ma trận TF-IDF từ danh sách Services"""
    global tfidf_vectorizer, tfidf_matrix, services_df

    if not services_data:
        print("⚠️ Không có dữ liệu Service để train.")
        return

    corpus = []
    ids = []

    for svc in services_data:
        # Ghép tất cả văn bản có giá trị của Dịch vụ
        features_text = " ".join(svc.get("features", []))
        combined_text = f"{svc.get('name', '')} {svc.get('type', '')} {svc.get('description', '')} {features_text} {svc.get('address', '')}"
        
        corpus.append(preprocess_text(combined_text))
        ids.append(str(svc["_id"]))

    # Lưu DataFrame để sau này lấy lại đoạn text tra cứu cho User
    services_df = pd.DataFrame({"service_id": ids, "text": corpus})

    # Khởi tạo và Train TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    print(f"🤖 Đã nạp thành công {len(ids)} Dịch vụ vào Ma trận In-Memory TF-IDF.")

def get_recommendations(user_id, preferences, booked_ids, wishlist_ids, top_k=6):
    global tfidf_vectorizer, tfidf_matrix, services_df

    if tfidf_matrix is None or services_df is None:
        print("❌ Lỗi: Ma trận TF-IDF chưa được nạp!")
        return []

    # 1. TẠO PROFILE TEXT CHO USER
    user_text_parts = []
    
    if preferences:
        prefs_str = " ".join(preferences)
        user_text_parts.append(preprocess_text(prefs_str))

    wishlist_texts = services_df[services_df['service_id'].isin(wishlist_ids)]['text'].tolist()
    if wishlist_texts:
        user_text_parts.extend(wishlist_texts * 2) 

    booked_texts = services_df[services_df['service_id'].isin(booked_ids)]['text'].tolist()
    if booked_texts:
        user_text_parts.extend(booked_texts * 3)

    user_profile_text = " ".join(user_text_parts)
    
    # ---- DÒNG IN LOG ĐỂ DEBUG ----
    print(f"\n🧠 [AI DEBUG] Khách hàng ID: {user_id}")
    print(f"👉 Preferences của khách: {preferences}")
    print(f"👉 Văn bản tổng hợp để tính toán: '{user_profile_text}'")
    print(f"👉 Tổng số dịch vụ đang nằm trong RAM: {len(services_df)}")
    # ------------------------------

    if not user_profile_text.strip():
        print("⚠️ Khách hàng chưa có đủ dữ liệu sở thích (Cold Start).")
        return [] 

    # 2. VECTOR HÓA VÀ TÍNH COSINE
    user_vector = tfidf_vectorizer.transform([user_profile_text])
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_k:][::-1]

    recommendations = []
    print("📊 [AI DEBUG] BẢNG ĐIỂM CHI TIẾT (TỪ CAO XUỐNG THẤP):")
    for idx in top_indices:
        svc_id = services_df.iloc[idx]['service_id']
        score = float(cosine_sim[idx])
        
        # In tất cả điểm ra màn hình để tra cứu
        print(f"   - Service ID: {svc_id} | Điểm Cosine: {score:.4f}")
        
        if score > 0 and svc_id not in booked_ids:
            recommendations.append({
                "service_id": svc_id,
                "score": round(score, 4)
            })

    return recommendations