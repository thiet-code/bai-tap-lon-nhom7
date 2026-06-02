# Hybrid Recommender System on MovieLens 100K

Kết hợp Collaborative Filtering (BPR-MF, NeuMF) và Content-Based Filtering, giải quyết bài toán cold-start
## 📖 1. Giới thiệu & Mục tiêu
Hệ thống là sự kết hợp tối ưu giữa ba phương pháp tiếp cận cốt lõi:
* **Bayesian Personalized Ranking Matrix Factorization (BPR-MF)**
* **Neural Collaborative Filtering (NeuMF)** (bao gồm GMF và MLP)
* **Content-Based Filtering** dựa trên kỹ thuật trích xuất đặc trưng văn bản TF-IDF.

Đồng thời, hệ thống tích hợp cơ chế **blend với độ phổ biến (popularity-based blending)** nhằm xử lý hiệu quả bài toán khởi đầu lạnh (cold-start) đối với các thực thể mới tham gia vào hệ thống.

## 🛠️ 2. Công nghệ & Thư viện sử dụng
* Python 3.10+ là ngôn ngữ chính.
* PyTorch dùng để cài đặt mô hình mạng nơ-ron BPR-MF, GMF, MLP, NeuMF.
* scikit-learn cung cấp TF-IDF Vectorizer, Truncated SVD và metric phụ trợ.
* NumPy, Pandas, SciPy cho xử lý dữ liệu và ma trận thưa.
* Matplotlib để xuất biểu đồ so sánh mô hình.
* pytest-style cho 16 unit test kiểm tra hành vi mô hình.

## 📊 3. Kết quả thử nghiệm nổi bật
* Hybrid đứng đầu cả 7 chỉ số đánh giá so với 5 baseline.
* NDCG@10 đạt 0.409, vượt NeuMF đơn lẻ +4.4%.
* MRR đạt 0.340, tăng +5.3% so với NeuMF.
* Trên kịch bản cold-start nhân tạo, Hybrid sát NeuMF (gap < 1%) và vượt xa ItemPop (+47% NDCG), Content-Based (+170%).
* Toàn bộ pipeline (6 mô hình + tuning) chạy ~60 giây trên GPU RTX 4060.


## 👥 4. Danh sách thành viên nhóm 7


| STT | Họ và tên | Mã sinh viên | Vai trò / Nhiệm vụ chính trong dự án |
| :---: | :--- | :---: | :--- |
| 1 | Hoàng Văn Thiết | 21002239 | Khảo sát dữ liệu MovieLens 100K, chia tập theo phương pháp Leave-One-Out, sinh mẫu negative và cài đặt các thước đo HR, NDCG, MRR. |
| 2 | Ngô Thế Hướng | 21002209 | Xây dựng mô hình ItemPop, ItemKNN và mô hình Content-Based. |
| 3 | Nguyễn Trung Hiếu | 21002204 | Xây dựng bộ chấm điểm lai Hybrid Engine chuẩn hóa Z-score, làm giao diện CLI demo và Unit Test. |
| 4 | Phạm Hoàng Long | 21002218 | Cài đặt và huấn luyện hai kiến trúc mạng nơ-ron BPR-MF và NeuMF. |

