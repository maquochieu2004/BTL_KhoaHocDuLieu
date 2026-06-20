# =====================================================
# PHÂN TÍCH TAI NẠN GIAO THÔNG VIỆT NAM 2020-2024
# Tác giả: Ma Quốc Hiếu
# MSSV: K225480106089
# Môn: Khoa học dữ liệu
# =====================================================

# Thư viện làm việc với file và thư mục
import os

# Thư viện xử lý dữ liệu
import pandas as pd

# Thư viện vẽ biểu đồ
import matplotlib.pyplot as plt

# Thuật toán dự đoán
from sklearn.linear_model import LinearRegression

# Thuật toán phân cụm
from sklearn.cluster import KMeans

# =====================================================
# TẠO THƯ MỤC CHỨA HÌNH ẢNH
# =====================================================

if not os.path.exists("images"):
    os.makedirs("images")

# =====================================================
# ĐỌC FILE EXCEL
# =====================================================

df = pd.read_excel(
    "Traffic_Accident_Vietnam_Realistic_2020_2024.xlsx"
)

print("\n========== THÔNG TIN DỮ LIỆU ==========")
print(df.head())

# =====================================================
# CÂU HỎI 1
# TAI NẠN GIAO THÔNG THAY ĐỔI NHƯ THẾ NÀO QUA CÁC NĂM
# =====================================================

# Tính tổng số vụ tai nạn theo từng năm
yearly = df.groupby("Year")["Accidents"].sum()

plt.figure(figsize=(10, 6))

# Vẽ biểu đồ đường
plt.plot(
    yearly.index,
    yearly.values,
    marker="o",
    linewidth=3
)

# Hiển thị giá trị trên từng điểm
for x, y in zip(yearly.index, yearly.values):
    plt.text(x, y + 200, f"{y:,}")

plt.title("So vu tai nan giao thong qua cac nam")
plt.xlabel("Nam")
plt.ylabel("So vu tai nan (vu)")
plt.grid(True)

plt.figtext(
    0.15,
    0.01,
    "Ghi chu: So vu tai nan co xu huong tang dan tu 2020 den 2024."
)

plt.savefig("images/q1_trend.png")
plt.close()

# =====================================================
# CÂU HỎI 2
# TOP 10 TỈNH CÓ NHIỀU TAI NẠN NHẤT
# =====================================================

top10 = (
    df.groupby("Province")["Accidents"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    top10.index,
    top10.values
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        int(bar.get_height()),
        ha='center'
    )

plt.xticks(rotation=45)

plt.title("Top 10 tinh co nhieu tai nan nhat")
plt.ylabel("Tong so vu tai nan (vu)")

plt.tight_layout()

plt.savefig("images/q2_top10.png")
plt.close()

# =====================================================
# CÂU HỎI 3
# TỶ LỆ TỬ VONG TRÊN MỖI VỤ TAI NẠN
# =====================================================

rate = df.groupby("Province").sum()

# Công thức:
# Tỷ lệ tử vong = Số người chết / Số vụ tai nạn
rate["DeathRate"] = (
    rate["Deaths"] /
    rate["Accidents"]
)

top20 = (
    rate["DeathRate"]
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(14, 6))

bars = plt.bar(
    top20.index,
    top20.values
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{bar.get_height():.2f}",
        ha="center",
        fontsize=8
    )

plt.xticks(rotation=60)

plt.title("Ty le tu vong tren moi vu tai nan")
plt.ylabel("Ty le")

plt.tight_layout()

plt.savefig("images/q3_death_rate.png")
plt.close()

# =====================================================
# CÂU HỎI 4
# TOP 10 TỈNH CÓ NHIỀU NGƯỜI TỬ VONG NHẤT
# =====================================================

top_deaths = (
    df.groupby("Province")["Deaths"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

plt.bar(
    top_deaths.index,
    top_deaths.values
)

plt.xticks(rotation=45)

plt.title("Top 10 tinh co nhieu nguoi tu vong nhat")
plt.ylabel("So nguoi chet")

plt.tight_layout()

plt.savefig("images/q4_deaths.png")
plt.close()

# =====================================================
# CÂU HỎI 5
# TOP 10 TỈNH CÓ NHIỀU NGƯỜI BỊ THƯƠNG NHẤT
# =====================================================

top_injuries = (
    df.groupby("Province")["Injuries"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

plt.bar(
    top_injuries.index,
    top_injuries.values
)

plt.xticks(rotation=45)

plt.title("Top 10 tinh co nhieu nguoi bi thuong nhat")
plt.ylabel("So nguoi bi thuong")

plt.tight_layout()

plt.savefig("images/q5_injuries.png")
plt.close()

# =====================================================
# CÂU HỎI 6
# DỰ ĐOÁN SỐ VỤ TAI NẠN NĂM 2026
# THUẬT TOÁN: LINEAR REGRESSION
# =====================================================

yearly_df = (
    df.groupby("Year")["Accidents"]
    .sum()
    .reset_index()
)

# X = năm
X = yearly_df[["Year"]]

# y = số vụ tai nạn
y = yearly_df["Accidents"]

# Khởi tạo mô hình
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X, y)

# Dự đoán năm 2026
prediction = model.predict(
    pd.DataFrame({"Year": [2026]})
)

future_value = int(prediction[0])

print("\nDự đoán số vụ tai nạn năm 2026:", future_value)

# =====================================================
# CÂU HỎI 7
# PHÂN CỤM TỈNH THÀNH
# THUẬT TOÁN: K-MEANS
# =====================================================

cluster_data = (
    df.groupby("Province")
    [["Accidents", "Deaths"]]
    .mean()
)

# Chia thành 3 cụm
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Gán nhãn cụm
cluster_data["Cluster"] = (
    kmeans.fit_predict(cluster_data)
)

plt.figure(figsize=(10, 6))

plt.scatter(
    cluster_data["Accidents"],
    cluster_data["Deaths"],
    c=cluster_data["Cluster"]
)

plt.xlabel("So vu tai nan")
plt.ylabel("So nguoi chet")

plt.title("Phan cum cac tinh bang K-Means")

plt.savefig("images/q7_clustering.png")
plt.close()

print("\n=================================")
print("CHƯƠNG TRÌNH CHẠY THÀNH CÔNG")
print("7 ẢNH ĐÃ ĐƯỢC LƯU TRONG THƯ MỤC IMAGES")
print("=================================")