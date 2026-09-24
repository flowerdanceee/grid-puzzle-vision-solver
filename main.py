import cv2
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# 1. 读取并预处理图像
# ============================================================
img = cv2.imread("game2.jpg")

# 缩小图像，降低后续轮廓检测的计算量
img = cv2.pyrDown(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
_, thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

# ============================================================
# 2. 检测所有轮廓
# ============================================================
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

# ============================================================
# 3. 轮廓辅助函数
# ============================================================

def count_children(index, hierarchy):
    """
    统计指定轮廓的直接子轮廓数量。
    """
    count = 0

    child = hierarchy[index][2]

    while child != -1:
        count += 1

        # 下一个同级轮廓
        child = hierarchy[child][0]

    return count


def get_all_children(index, hierarchy):
    """
    获取指定轮廓的所有直接子轮廓索引。
    """
    children = []
    child = hierarchy[index][2]
    while child != -1:
        children.append(child)
        # 下一个同级轮廓
        child = hierarchy[child][0]

    return children


# ============================================================
# 4. 筛选棋盘外围轮廓
# ============================================================
candidates = []
for index, contour in enumerate(contours):

    # --------------------------------------------------------
    # 条件 1：面积不能太小
    # --------------------------------------------------------
    area = cv2.contourArea(contour)

    if area < 10000:
        continue

    # --------------------------------------------------------
    # 条件 2：外围轮廓应该接近正方形
    # --------------------------------------------------------
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if not 0.8 < aspect_ratio < 1.2:
        continue

    # --------------------------------------------------------
    # 条件 3：轮廓近似应该是四边形
    # --------------------------------------------------------
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approx) != 4:
        continue

    # --------------------------------------------------------
    # 条件 4：应该包含若干内部格子
    #
    # 例如：
    # 16 = 4 × 4
    # 25 = 5 × 5
    # 64 = 8 × 8
    # --------------------------------------------------------
    child_count = count_children(index,hierarchy)
    grid_size = int(np.sqrt(child_count))

    if child_count < 16:
        continue

    if grid_size * grid_size != child_count:
        continue

    # 保存候选轮廓
    candidates.append(
        {
            "index": index,
            "area": area,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "child_count": child_count,
        }
    )

# ============================================================
# 5. 选择棋盘外围轮廓
# ============================================================
if not candidates:
    raise RuntimeError("没有找到符合条件的棋盘轮廓")

target_index = candidates[0]["index"]

# ============================================================
# 6. 获取棋盘内部格子轮廓
# ============================================================
inside_contours = get_all_children(target_index,hierarchy)


# ============================================================
# 7. 提取每个格子的特征
# ============================================================
def get_contour_features(contours,contour_indices,img):
    """
    提取每个格子的：

    - 中心坐标
    - 平均颜色
    """

    centers = []
    mean_colors = []

    for index in contour_indices:

        contour = contours[index]

        # ----------------------------------------------------
        # 计算轮廓中心
        # ----------------------------------------------------
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        # ----------------------------------------------------
        # 创建当前轮廓的 mask
        # ----------------------------------------------------
        mask = np.zeros(img.shape[:2],dtype=np.uint8)
        cv2.drawContours(mask,[contour],-1,255,-1)

        # ----------------------------------------------------
        # 计算轮廓内部的平均颜色
        # ----------------------------------------------------
        mean_color = cv2.mean(img,mask=mask)[:3]
        centers.append([center_x, center_y])
        mean_colors.append(mean_color)

    return (np.array(centers, dtype=np.int32),np.array(mean_colors, dtype=np.float32))

centers, mean_colors = get_contour_features(contours,inside_contours,img)

# ============================================================
# 8. 根据格子数量确定棋盘大小
# ============================================================
cell_count = len(mean_colors)
n = int(np.sqrt(cell_count))
if n * n != cell_count:
    raise RuntimeError(
        f"检测到 {cell_count} 个格子，"
        f"无法组成正方形棋盘"
    )

print(f"检测到棋盘大小：{n} × {n}")
print(f"检测到格子数量：{cell_count}")

# ============================================================
# 9. 使用 K-Means 对格子颜色进行分类
# ============================================================
kmeans = KMeans(n_clusters=n,random_state=0,n_init="auto")
labels = kmeans.fit_predict(mean_colors)


# ============================================================
# 10. 将检测结果转换成棋盘矩阵
# ============================================================
def make_matrix(centers, labels, n):
    """
    根据格子中心坐标，将一维 labels 转换成 n × n 棋盘。
    排序方式：
        1. 按 y 从上到下确定行
        2. 每一行内部按 x 从左到右排序
    注意：
    这里先按照 y 排序，再每 n 个格子划分成一行，
    可以避免中心点存在 1~2 像素检测误差导致的错行。
    """

    centers = np.asarray(centers)
    labels = np.asarray(labels)

    # --------------------------------------------------------
    # 第一步：按照 y 从上到下排序
    # --------------------------------------------------------
    y_order = np.argsort(centers[:, 1])
    centers = centers[y_order]
    labels = labels[y_order]
    matrix = []

    # --------------------------------------------------------
    # 第二步：每 n 个格子作为一行
    # --------------------------------------------------------
    for row in range(n):
        start = row * n
        end = (row + 1) * n
        row_centers = centers[start:end]
        row_labels = labels[start:end]

        # ----------------------------------------------------
        # 第三步：当前行按照 x 从左到右排序
        # ----------------------------------------------------
        x_order = np.argsort(row_centers[:, 0])

        matrix.append(row_labels[x_order])

    return np.array(matrix)


matrix = make_matrix(centers,labels,n)

print("\n棋盘矩阵：")
print(matrix)


# ============================================================
# 11. 求解旗子位置
# ============================================================
def solve_flags(board):
    """
    求解旗子放置问题。

    约束：

    1. 每一行只能放一个旗子
    2. 每一列只能放一个旗子
    3. 每一种颜色只能放一个旗子
    4. 相邻行的旗子不能位于相邻列
       （8 邻域不能相邻）

    返回：
        solution[row] = 该行旗子所在的列

    例如：

        [2, 5, 0, 7, ...]

    表示：

        第 0 行 → 第 2 列
        第 1 行 → 第 5 列
        第 2 行 → 第 0 列
        第 3 行 → 第 7 列
    """

    n = len(board)

    # --------------------------------------------------------
    # 状态记录
    # --------------------------------------------------------

    # 某一列是否已经放置旗子
    used_cols = [False] * n

    # 某一种颜色是否已经使用
    used_colors = [False] * n

    # solution[row] = 当前行旗子的列
    solution = [-1] * n

    # --------------------------------------------------------
    # 回溯搜索
    # --------------------------------------------------------

    def backtrack(row):

        # ----------------------------------------------------
        # 所有行都已经放置完成
        # ----------------------------------------------------
        if row == n:
            return True

        # ----------------------------------------------------
        # 尝试当前行的每一列
        # ----------------------------------------------------
        for col in range(n):

            # ------------------------------------------------
            # 条件 1：当前列不能已经有旗子
            # ------------------------------------------------
            if used_cols[col]:
                continue

            # ------------------------------------------------
            # 条件 2：当前颜色不能已经使用
            # ------------------------------------------------
            color = board[row][col]

            if used_colors[color]:
                continue

            # ------------------------------------------------
            # 条件 3：不能与上一行的旗子处于 8 邻域
            # 因为我们从上往下搜索，所以只需要检查上一行。
            # ------------------------------------------------
            if row > 0:

                previous_col = solution[row - 1]

                if abs(col - previous_col) <= 1:
                    continue

            # ------------------------------------------------
            # 选择当前位置
            # ------------------------------------------------
            solution[row] = col
            used_cols[col] = True
            used_colors[color] = True

            # ------------------------------------------------
            # 继续搜索下一行
            # ------------------------------------------------
            if backtrack(row + 1):
                return True

            # ------------------------------------------------
            # 回溯
            # ------------------------------------------------
            solution[row] = -1
            used_cols[col] = False
            used_colors[color] = False

        return False

    # ========================================================
    # 开始搜索
    # ========================================================

    if backtrack(0):
        return solution

    return None


# ============================================================
# 12. 求解
# ============================================================

result = solve_flags(matrix)

print("\n鸽子位置：")
print(result)
