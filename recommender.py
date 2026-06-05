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
    """Tính Cosine Similarity để đưa ra gợi ý"""
    global tfidf_vectorizer, tfidf_matrix, services_df

    if tfidf_matrix is None or services_df is None:
        return []

    # 1. TẠO PROFILE TEXT CHO USER (Gia trọng số)
    user_text_parts = []
    
    # Cộng điểm Preferences (Trọng số x1)
    if preferences:
        prefs_str = " ".join(preferences)
        user_text_parts.append(preprocess_text(prefs_str))

    # Cộng điểm Wishlist (Trọng số x2)
    wishlist_texts = services_df[services_df['service_id'].isin(wishlist_ids)]['text'].tolist()
    if wishlist_texts:
        user_text_parts.extend(wishlist_texts * 2) 

    # Cộng điểm Booking (Trọng số x3 - Tín hiệu mạnh nhất)
    booked_texts = services_df[services_df['service_id'].isin(booked_ids)]['text'].tolist()
    if booked_texts:
        user_text_parts.extend(booked_texts * 3)

    user_profile_text = " ".join(user_text_parts)

    # NẾU USER MỚI TINH (Cold Start) -> Chả có thông tin gì
    if not user_profile_text.strip():
        return [] # Trả về mảng rỗng (Node.js sẽ lấy các dịch vụ Trending bù vào)

    # 2. VECTOR HÓA USER & TÍNH COSINE SIMILARITY
    user_vector = tfidf_vectorizer.transform([user_profile_text])
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Lấy top K index có độ tương đồng cao nhất
    top_indices = cosine_sim.argsort()[-top_k:][::-1]

    # Loại bỏ những dịch vụ mà user ĐÃ ĐẶT RỒI (Không gợi ý lại chỗ vừa đi xong)
    recommendations = []
    for idx in top_indices:
        svc_id = services_df.iloc[idx]['service_id']
        score = float(cosine_sim[idx])
        
        # Chỉ lấy nếu độ tương đồng > 0 và chưa từng book
        if score > 0 and svc_id not in booked_ids:
            recommendations.append({
                "service_id": svc_id,
                "score": round(score, 4)
            })

    return recommendations