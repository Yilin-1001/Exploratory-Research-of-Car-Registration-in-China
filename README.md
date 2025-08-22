# Exploratory-Research-of-Car-Registration-in-China
This data analysis project explores the potential pattern of the number of car registration in one month in China by clustering, Association Rule Mining and Outlier Detection.

中国汽车2025年5月上牌量聚类分析 – 技术说明 (中英文双语)
项目方法整体概述 | Project Methodology Overview
中国汽车2025年5月上牌量聚类分析项目旨在利用无监督学习方法从大量车辆上牌数据中发现潜在模式和类别。数据集包含了多种类型的特征，包括数值特征（如上牌量、价格等）和分类特征（如车辆类型、燃料类别、城市等）。由于特征类型多样，我们采用了两种聚类方法：基于 Gower 距离的层次聚类 和 K-Prototypes 聚类算法，分别针对混合数据类型的距离计算和直接聚类进行优化。项目流程包括数据预处理、使用两种方法分别进行聚类分析、选择适当的簇数，以及结果输出与简单可视化。通过这两种方法的对比，能够从不同角度刻画2025年5月汽车市场上牌数据的聚类模式，为进一步的市场细分和业务决策提供支持。
This project analyzes the China vehicle registration data of May 2025 using unsupervised clustering methods to uncover latent patterns and groupings in the data. The dataset contains diverse feature types, including numerical features (e.g. registration counts, prices) and categorical features (e.g. vehicle segment, fuel type, city). To accommodate this mix of data types, two clustering approaches are employed: hierarchical clustering based on Gower distance and the K-Prototypes clustering algorithm, each tailored for mixed data but with different strategies (distance matrix vs. direct iterative clustering). The overall workflow includes data preprocessing, applying each clustering method, determining the optimal number of clusters, and outputting results with basic visualization. By using both methods, the analysis captures clustering patterns in the May 2025 vehicle market from different perspectives, providing insights for market segmentation and informed decision-making.
聚类方法原理解释 | Clustering Methods and Principles
基于 Gower 距离的层次聚类原理 | Hierarchical Clustering with Gower Distance
Gower 距离是一种针对混合类型数据的相似度度量，适用于包含数值、分类等不同类型特征的数据。对于任意两个样本 
i
i 和 
j
j，Gower 距离首先计算每个特征上的差异，再综合得到总体差异。具体而言，对每个特征 
k
k 定义距离 
d
(
k
)
(
i
,
j
)
d 
(k)
 (i,j)：若特征 
k
k 为数值型，则采用归一化差值 
d
(
k
)
(
i
,
j
)
=
∣
x
i
k
−
x
j
k
∣
R
k
d 
(k)
 (i,j)= 
R 
k
​	
 
∣x 
ik
​	
 −x 
jk
​	
 ∣
​	
 ，其中 
R
k
R 
k
​	
  是特征 
k
k 在数据中的范围（最大值减去最小值）；若特征 
k
k 为分类型，则 
d
(
k
)
(
i
,
j
)
=
0
d 
(k)
 (i,j)=0 （当 
x
i
k
=
x
j
k
x 
ik
​	
 =x 
jk
​	
 ）或 
1
1（当 
x
i
k
≠
x
j
k
x 
ik
​	
 

=x 
jk
​	
 ）。如果某特征在某个样本中缺失，则该特征对该对样本的不相似度不予计算。对于所有特征，Gower 不相似度定义为各特征距离的加权平均：
d
i
j
(Gower)
=
∑
k
=
1
p
w
k
(
i
j
)
 
d
(
k
)
(
i
,
j
)
∑
k
=
1
p
w
k
(
i
j
)
 
,
d 
ij
(Gower)
​	
 = 
∑ 
k=1
p
​	
 w 
k
(ij)
​	
 
∑ 
k=1
p
​	
 w 
k
(ij)
​	
 d 
(k)
 (i,j)
​	
 ,
其中 
p
p 为特征总数，权重 
w
k
(
i
j
)
=
1
w 
k
(ij)
​	
 =1 表示样本 
i
,
j
i,j 在第 
k
k 个特征上均有值（非缺失），若任一缺失则 
w
k
(
i
j
)
=
0
w 
k
(ij)
​	
 =0。通过上述公式得到的 Gower 距离介于0和1之间，数值越大表示样本差异越大。在本项目中，我们使用 Gower 距离矩阵作为层次聚类算法的输入，对混合型数据计算严谨的样本间距离。
层次聚类采用凝聚式（自底向上）策略：开始时每个样本各自成为一类，不断合并最近的簇直至达到预定的簇数 
K
K。我们选择“平均链接（average linkage）”作为聚类准则，即定义两个簇 
A
A 和 
B
B 之间的距离为簇间所有成对样本距离的平均值：
d
(
A
,
B
)
=
1
∣
A
∣
⋅
∣
B
∣
∑
i
∈
A
∑
j
∈
B
d
(
i
,
j
)
 
,
d(A,B)= 
∣A∣⋅∣B∣
1
​	
  
i∈A
∑
​	
  
j∈B
∑
​	
 d(i,j),
其中 
d
(
i
,
j
)
d(i,j) 是样本 
i
i 和 
j
j 之间的距离（在本项目中为 Gower 距离）。平均链接方法能够综合考虑簇间所有样本的距离，从而生成相对均衡的簇。利用预先计算的 Gower 距离矩阵，本项目通过 AgglomerativeClustering 实现层次聚类，将 metric="precomputed" 与平均链接结合，以直接使用自定义距离矩阵。我们针对簇数 
K
K 进行尝试（例如 
K
=
2
K=2 到 
9
9），使用轮廓系数(silhouette score)评估每种簇划分的质量，从中选取轮廓系数最高的 
K
K 作为最佳聚类数量。层次聚类的结果包括每个样本的簇标签，以及基于簇的特征概况分析，如数值特征均值和主要分类特征分布等。此外，我们对距离矩阵采样应用经典多维尺度法 (MDS) 将高维距离映射到二维平面，以可视化聚类结果的整体结构。
Hierarchical clustering with Gower distance is utilized to handle the mixed data types in our dataset. Gower’s distance provides a way to compute dissimilarity between two data points with numeric and categorical features. For any two samples 
i
i and 
j
j, the Gower computation considers each feature 
k
k individually and then aggregates the differences. Specifically, for feature 
k
k, the pairwise distance 
d
(
k
)
(
i
,
j
)
d 
(k)
 (i,j) is defined as follows: if feature 
k
k is numeric, we use the normalized difference 
d
(
k
)
(
i
,
j
)
=
∣
x
i
k
−
x
j
k
∣
R
k
d 
(k)
 (i,j)= 
R 
k
​	
 
∣x 
ik
​	
 −x 
jk
​	
 ∣
​	
 , where 
R
k
R 
k
​	
  is the range (max minus min) of feature 
k
k in the dataset. If feature 
k
k is categorical, then 
d
(
k
)
(
i
,
j
)
=
0
d 
(k)
 (i,j)=0 if 
x
i
k
=
x
j
k
x 
ik
​	
 =x 
jk
​	
  (no difference) or 
1
1 if 
x
i
k
≠
x
j
k
x 
ik
​	
 

=x 
jk
​	
  (different category). Features with missing values are skipped for that pair. The overall Gower dissimilarity is the weighted average of feature-wise distances:
d
i
j
(Gower)
=
∑
k
=
1
p
w
k
(
i
j
)
 
d
(
k
)
(
i
,
j
)
∑
k
=
1
p
w
k
(
i
j
)
 
,
d 
ij
(Gower)
​	
 = 
∑ 
k=1
p
​	
 w 
k
(ij)
​	
 
∑ 
k=1
p
​	
 w 
k
(ij)
​	
 d 
(k)
 (i,j)
​	
 ,
where 
p
p is the total number of features and 
w
k
(
i
j
)
=
1
w 
k
(ij)
​	
 =1 if both samples 
i
i and 
j
j have valid values on feature 
k
k (otherwise 
w
k
(
i
j
)
=
0
w 
k
(ij)
​	
 =0). The resulting Gower distance lies between 0 and 1, with larger values indicating more dissimilar samples. In this project, we compute a full distance matrix using Gower distance for all sample pairs, providing a rigorous measure of dissimilarity for mixed data.
The hierarchical clustering algorithm then proceeds in an agglomerative (bottom-up) fashion: each sample starts in its own cluster, and clusters are iteratively merged until the desired number of clusters 
K
K is reached. We use average linkage as the linkage criterion, meaning the distance between two clusters 
A
A and 
B
B is defined as the average of all pairwise distances between samples in 
A
A and those in 
B
B:
d
(
A
,
B
)
=
1
∣
A
∣
⋅
∣
B
∣
∑
i
∈
A
∑
j
∈
B
d
(
i
,
j
)
 
,
d(A,B)= 
∣A∣⋅∣B∣
1
​	
  
i∈A
∑
​	
  
j∈B
∑
​	
 d(i,j),
where 
d
(
i
,
j
)
d(i,j) is the distance between sample 
i
i and 
j
j (in our case, the Gower distance). This average linkage method accounts for all members of each cluster when merging, producing more balanced clusters. In implementation, we use AgglomerativeClustering with metric="precomputed" and average linkage to directly supply our custom Gower distance matrix. We explore different cluster counts 
K
K (for example, 2 through 9) and evaluate each clustering result with the silhouette coefficient, which measures clustering quality. The number of clusters with the highest silhouette score is chosen as the optimal 
K
K. The hierarchical clustering yields a cluster label for each sample and allows further profiling of clusters (such as computing mean values of numerical features and the distribution of top categorical features in each cluster). Additionally, we apply Classical Multi-Dimensional Scaling (MDS) on a subset of the distance matrix to project the high-dimensional relationships into 2D for visualization, giving an overall picture of cluster structure in the data.
K-Prototypes 聚类算法原理 | K-Prototypes Clustering Algorithm
K-Prototypes 算法是针对混合数据（同时含有数值和类别特征）的一种划分式聚类方法，由 Huang 在1998年提出。它综合了 K-Means 和 K-Modes 算法的思想：针对数值型特征，使用与 K-Means 相同的距离度量（通常为欧氏距离的平方）；针对分类特征，使用简单匹配不相似度（类别不同则距离为1，相同则为0），类似 K-Modes 中对类别变量的处理。为了将两种不同类型特征的差异合成为一个统一的距离度量，K-Prototypes 引入了一个可调权重参数 
γ
γ，用于平衡数值距离和类别距离的影响。对于给定数据点 
x
x 和簇中心（原型） 
c
c，定义 K-Prototypes 距离为：
d
kproto
(
x
,
c
)
=
∑
j
=
1
m
(
x
j
−
c
j
)
2
  
+
  
γ
∑
l
=
1
r
1
(
x
l
′
≠
c
l
′
)
 
,
d 
kproto
​	
 (x,c)= 
j=1
∑
m
​	
 (x 
j
​	
 −c 
j
​	
 ) 
2
 +γ 
l=1
∑
r
​	
 1(x 
l
′
​	
 

=c 
l
′
​	
 ),
其中前一项求和覆盖所有数值特征 
j
j（共有 
m
m 个，使用数值差的平方），后一项覆盖所有分类特征 
l
l（共有 
r
r 个，
1
(
⋅
)
1(⋅) 为指示函数，当分类值不相同时取1，相同时取0），
c
j
c 
j
​	
  和 
c
l
′
c 
l
′
​	
  分别是簇中心在对应数值和分类特征上的值。参数 
γ
γ 权衡了类别不匹配的惩罚幅度，防止类别特征对距离的贡献被数值特征掩盖（用户可根据数据特征尺度调整 
γ
γ，也可使用默认估计值）。聚类的目标是最小化数据点到其所属簇原型的上述距离之和，即最小化总的簇内距离代价。
K-Prototypes 算法的迭代过程与 K-Means 类似：
初始化 – 随机选择 
K
K 个初始原型（包括数值均值部分和分类模式部分），或使用 Huang 提出的改进初始化方法。
分配簇 – 将每个数据点分配到与其距离 
d
kproto
d 
kproto
​	
  最近的原型所属的簇。
更新原型 – 对每个簇，重新计算新的原型：数值特征取簇内所有点的均值，分类特征取簇内出现频率最高的类别（模式）。
重复执行“分配簇”和“更新原型”步骤，直到簇分配不再变化或达到预定的迭代次数。算法收敛后得到稳定的簇划分结果。
在本项目中，我们使用 kmodes 库提供的实现 (KPrototypes 类) 来对数据执行 K-Prototypes 聚类。在预处理阶段，我们将所有数值特征归一化到 [0,1] 区间，这样可以减少选择 
γ
γ 时的难度，因为所有特征的数值尺度相近。模型训练过程中，我们针对一组候选簇数 
K
K 值（如 6 至 15）反复运行聚类，将 成本 (cost)（即簇内距离的总和）作为评价指标。随着 
K
K 增大，聚类成本通常会下降，我们选择其中成本最小的簇数作为最佳 
K
K。聚类完成后，每个样本获得一个簇标签，我们可以进一步提取每个簇的特征画像，包括数值特征的均值（本项目中特别保留“上牌量”以原始单位呈现簇均值）和每个类别特征中占比最高的几种类别。K-Prototypes 聚类方法能够高效地处理大规模数据，其直接迭代优化的机制使其在混合数据聚类中表现良好。
The K-Prototypes algorithm is a partition-based clustering method designed for mixed-type data (containing both numeric and categorical features), introduced by Huang in 1998. It integrates the ideas of K-Means and K-Modes: for numeric features, it uses the same distance measure as K-Means (typically the squared Euclidean distance), and for categorical features, it uses a simple matching dissimilarity (distance is 1 if categories differ, 0 if they are the same), as employed in K-Modes. To combine the contributions of numeric and categorical differences into one unified distance metric, K-Prototypes introduces a weighting parameter 
γ
γ that balances the influence of categorical mismatch relative to numeric distance. For a given data point 
x
x and a cluster prototype 
c
c, the K-Prototypes distance is defined as:
d
kproto
(
x
,
c
)
=
∑
j
=
1
m
(
x
j
−
c
j
)
2
  
+
  
γ
∑
l
=
1
r
1
(
x
l
′
≠
c
l
′
)
 
,
d 
kproto
​	
 (x,c)= 
j=1
∑
m
​	
 (x 
j
​	
 −c 
j
​	
 ) 
2
 +γ 
l=1
∑
r
​	
 1(x 
l
′
​	
 

=c 
l
′
​	
 ),
where the first summation runs over all 
m
m numeric features (using the squared difference for each), and the second summation runs over all 
r
r categorical features (with 
1
(
⋅
)
1(⋅) being an indicator that equals 1 if feature values differ and 0 if they are the same). Here, 
c
j
c 
j
​	
  and 
c
l
′
c 
l
′
​	
  represent the prototype’s values for the numeric feature 
j
j and categorical feature 
l
l, respectively. The parameter 
γ
γ controls the relative weight of a categorical mismatch in the distance calculation, to ensure categorical features are not overshadowed by numeric features; 
γ
γ can be tuned based on the scale of the data or set to a default heuristic. The clustering objective is to minimize the total within-cluster distance cost, i.e., the sum of 
d
kproto
(
x
i
,
c
 
c
l
u
s
t
e
r
(
i
)
)
d 
kproto
​	
 (x 
i
​	
 ,c 
cluster(i)
​	
 ) for all points 
x
i
x 
i
​	
  and their assigned cluster prototypes.
The iterative procedure of K-Prototypes mirrors that of K-Means:
Initialization: Choose 
K
K initial prototypes (each prototype consists of a numeric centroid and categorical mode for the cluster), either randomly or via improved methods suggested by Huang.
Cluster Assignment: Assign each data point to the cluster whose prototype has the smallest 
d
kproto
d 
kproto
​	
  distance to the point.
Prototype Update: For each cluster, update the prototype based on current members: compute the new numeric centroid (mean of each numeric feature in the cluster) and the new categorical mode (most frequent category for each categorical feature in the cluster).
Repeat the assignment and update steps until cluster memberships stabilize or a maximum number of iterations is reached. Upon convergence, the clusters and their prototypes are finalized.
In this project, we leverage the KPrototypes class from the kmodes Python library to perform K-Prototypes clustering. During preprocessing, all numeric features are scaled to [0,1] using min-max normalization, which helps in choosing an appropriate 
γ
γ since numeric ranges are standardized. We experiment with a range of candidate cluster counts 
K
K (e.g., 6 through 15) and run the clustering for each, evaluating using the cost (the sum of within-cluster distances) returned by the model. Typically, as 
K
K increases, the cost decreases; we select the number of clusters that yields the lowest cost among the tried values as the optimal 
K
K. After clustering, each data record is assigned a cluster label. We then derive cluster profiles, including the mean of each numeric feature for each cluster (with a special handling to present the “registration count” in its original unit rather than scaled) and the top categories by frequency for each categorical feature in each cluster. The K-Prototypes method is efficient for large datasets, and its direct iterative optimization makes it well-suited for clustering mixed-type data in this project.
cluster_gower_pipeline.py 脚本函数详解 | Explanation of Functions in cluster_gower_pipeline.py
该脚本实现了基于 Gower 距离和层次聚类的完整分析流程，从数据预处理到结果输出。下面对脚本中各主要函数的作用、输入输出、关键参数及在项目中的用途进行逐一说明。
This script implements the full analysis pipeline using Gower distance and hierarchical clustering, covering data preprocessing through result output. The following is a detailed explanation of each major function in the script, including its purpose, inputs/outputs, key parameters, and role in the project.
extract_price_mean(s)
功能： 提取价格区间字符串中的数值并计算均值。如果输入字符串包含形如 “A-B” 的价格区间，则返回区间中两个数值的平均值；如果只包含单一数值，则直接返回该数值；如果无法提取数字则返回 NaN。该函数的实现通过正则表达式搜索字符串中的数值，并进行简单的字符串替换来处理可能存在的不同破折号字符。
输入： s：表示价格的字符串（可能为区间或单值），或缺失值 NaN。
输出： float 类型的平均价格值。如果输入为区间字符串，输出为区间两端数值的平均；如果输入为单个数字字符串，则输出该数字转换的浮点值；无法提取数字时输出 NaN。
关键参数： 无其他参数；函数内部定义了对中文格式破折号的替换和对字符串中数字的提取逻辑。
作用： 本项目数据包含车辆指导价区间等信息，该函数用于将价格区间转换为可用于分析的数值型均价。例如，对于 “9.98-12.98万” 这样的价格区间字符串，函数会返回 11.48（单位为万），从而为后续聚类提供一个代表性价格数值。在脚本中，extract_price_mean 被应用于检测数据列中是否存在价格区间格式，并自动生成一个新的数值列“price_mean_auto”，以丰富数值特征集合。
Function: Extract the numeric values from a price range string and compute their average. If the input string contains a range in the form “A-B”, the function returns the average of the two numbers; if it contains a single number, it returns that number as a float; if no numeric value can be parsed, it returns NaN. The implementation uses regular expressions to find numeric substrings and handles different dash characters that might appear in the input.
Input: s: a string representing a price range or a single price (possibly including Chinese characters or symbols), or it can be NaN.
Output: A float value representing the average price. For a range string, it is the mean of the two endpoint numbers; for a single numeric string, it is the numeric value; if parsing fails, the result is NaN.
Key Parameters: There are no additional parameters besides the input string. The function internally standardizes the string (replacing various dash symbols) and uses a regex to extract numbers.
Role in Project: The dataset includes vehicle price information, sometimes given as a price range (e.g., “9.98-12.98万”). This function converts such textual price ranges into a single numeric value (e.g., 11.48, in ten-thousand yuan units) which can be used as a numerical feature for clustering. In the script, extract_price_mean is used to automatically create a new numeric column “price_mean_auto” when a price range column is detected in the input data, thereby augmenting the feature set for clustering.
looks_like_city_or_level(name)
功能： 判断给定的列名是否可能代表“城市、级别、车型、类别”等分类属性。实现方式是将列名转换为小写字符串，并检查其中是否包含特定关键词（“城市”，“级别”，“车型”，“类别”）。如果包含任一关键词则返回 True，否则返回 False。
输入： name：字符串，表示数据列的名称。
输出： 布尔值。True 表示列名看起来属于城市/级别/车型/类别这类特征，False 表示否。
关键参数： 函数内部定义了关键词列表 keys = ["城市", "级别", "车型", "类别"] 作为识别标志。比较时不区分大小写。
作用： 在项目的数据预处理阶段，需要筛选哪些列作为分类特征参与聚类。一般而言，高基数（unique 值很多）的分类变量在聚类中作用有限且计算代价高，脚本中采用了阈值（如唯一值超过200则不考虑）的策略。然而，一些重要的分类信息如城市、车辆级别等可能具有较多唯一值但仍有分析价值。looks_like_city_or_level 函数用于识别这类列名，以便在选择分类特征时进行例外处理——即使唯一值数量较多，只要列名包含上述关键词之一，仍将其视作分类特征纳入聚类分析。这确保了关键业务维度（地域、车型类别等）不会因取值种类多而被忽略。
Function: Determine whether a given column name likely represents a categorical attribute such as “city”, “level/segment”, “vehicle model/type”, or “category”. The function converts the name to lowercase and checks for the presence of specific keywords (“城市” meaning city, “级别” meaning level/segment, “车型” meaning vehicle type/model, “类别” meaning category). It returns True if any of the keywords are found, otherwise False.
Input: name: a string representing a column name in the dataset.
Output: Boolean. True if the column name appears to be related to city/level/model/category, False otherwise.
Key Parameters: The function uses an internal list of keywords (keys = ["城市", "级别", "车型", "类别"]) for the check. The comparison is case-insensitive.
Role in Project: During data preprocessing, we need to decide which columns should be treated as categorical features for clustering. Generally, categorical variables with very high cardinality (many unique values) are avoided due to computational cost and potentially diluted significance; the script uses a threshold (e.g., exclude categorical columns with more than 200 unique values). However, certain key categorical attributes like city or vehicle segment may have many unique values yet remain important for analysis. The looks_like_city_or_level function helps identify such columns by name so that they can be included as categorical features despite having a high number of unique values. This ensures that important business dimensions (such as geographic location or vehicle category) are not inadvertently dropped from the clustering due to high cardinality.
gower_distance(X_num, X_cat, num_ranges)
功能： 基于前述的 Gower 公式计算一组混合类型数据的距离矩阵。该函数接受数值特征矩阵和分类特征矩阵，逐对计算所有样本之间的 Gower 距离。实现细节包括：对每个数值特征列，按列范围归一化差值累加距离，并记录有效比较的权重计数；对每个分类特征列，比较类别是否相同，不同则距离计为1，相同为0，并相应累加距离和权重。最后将距离累加矩阵按权重平均，得到样本两两之间的平均距离。函数返回一个大小为 
n
×
n
n×n 的对称距离矩阵（numpy 数组）。
输入：
X_num: NumPy数组，形状为 
(
n
,
p
)
(n,p)，包含 
n
n 个样本的 
p
p 个数值特征。如果没有数值特征则传入 None。
X_cat: NumPy数组，形状为 
(
n
,
q
)
(n,q)，包含 
n
n 个样本的 
q
q 个分类特征（类型为 object 或字符串）。如果没有分类特征则传入 None。
num_ranges: NumPy数组，长度为 
p
p，对应每个数值特征的取值范围（max-min）。用于归一化数值距离。如果某特征范围为0（所有值相等），函数内部会将其视为1以避免除零。
输出： NumPy二维数组 D，形状为 
(
n
,
n
)
(n,n)，表示 
n
n 个样本之间的两两距离。矩阵为对称矩阵，且对角线为0。
关键参数： 函数没有显式其他参数，但内部使用了分块计算 (block 的大小512) 来处理大型矩阵以兼顾效率和内存。另外，函数在计算过程中分别累积了数值部分的距离和权重矩阵 W_num，以及分类部分的距离和权重矩阵 W_cat，最后将两者相加得到总距离矩阵和总权重矩阵。在合并结果时，对那些在所有特征上均无有效比较的样本对（权重和为0）设定权重为1，以避免除以0 的情况。
作用： 本函数是实现 Gower 距离聚类的核心。它将混合型特征的数据转化为可用于聚类算法的距离度量。在项目中，我们对抽样后的数据调用 gower_distance 得到距离矩阵 D。随后层次聚类算法以此矩阵为依据进行聚类，因此 gower_distance 的准确性和效率直接影响聚类结果。由于 Gower 距离计算代价与 
O
(
n
2
)
O(n 
2
 ) 级别的样本对数相关，为兼顾速度，函数实现中采用了向量化和分块处理策略。结果矩阵 D 被保存为 .npy 文件以备查。总而言之，该函数将预处理后的数据转换为距离矩阵，是后续基于距离的聚类和可视化分析（如 MDS）的基础。
Function: Compute the distance matrix for a dataset with mixed data types using the Gower distance formula. This function takes a matrix of numeric features and a matrix of categorical features, and calculates pairwise Gower distances between all samples. Key implementation steps include: for each numeric feature column, compute the absolute difference between all pairs of samples, normalize by the feature’s range, and accumulate these differences into the distance matrix while tracking a weight matrix for valid comparisons; for each categorical feature, determine whether pairs of samples match or not (distance 0 if same, 1 if different), and accumulate these into the distance matrix and weight matrix accordingly. In the end, the accumulated distance values are divided by the accumulated weights to yield the average distance for each pair. The result is an 
n
×
n
n×n symmetric distance matrix (as a NumPy array) for the 
n
n samples, with zeros on the diagonal.
Input:
X_num: a NumPy array of shape 
(
n
,
p
)
(n,p) containing 
p
p numeric features for 
n
n samples. If there are no numeric features, this can be None.
X_cat: a NumPy array of shape 
(
n
,
q
)
(n,q) containing 
q
q categorical features for the same 
n
n samples (with dtype object or string). Use None if no categorical features.
num_ranges: a NumPy array of length 
p
p providing the range (max minus min) for each numeric feature, used to normalize numeric differences. If a feature’s range is 0 (all values identical), the code treats it as 1 to avoid division by zero.
Output: A NumPy 2D array D of shape 
(
n
,
n
)
(n,n) giving the pairwise distances between all 
n
n samples. This matrix is symmetric and has zeros on the diagonal.
Key Details: The function has no additional external parameters, but internally it uses a block size (512) for processing pairwise differences in batches to balance memory usage and speed. It separately accumulates distance contributions and valid-count weights for numeric features (W_num) and categorical features (W_cat), then sums them to get total distance and weight matrices. When combining results, any sample pair with a total weight of 0 (meaning no feature could be compared, e.g., both samples missing all features) is assigned a weight of 1 to avoid division by zero. After computing the averaged distances, the matrix is symmetrized and the diagonal is set to 0.
Role in Project: This function is central to the Gower-based clustering pipeline. It transforms the preprocessed dataset into a distance matrix D that serves as the input for distance-based clustering. In our workflow, after sampling the data, we call gower_distance to obtain D, which is then used by the hierarchical clustering algorithm to form clusters. The accuracy and efficiency of gower_distance directly affect the clustering outcome, as an incorrect distance computation could lead to poor clustering, and a slow computation could be a bottleneck. Because computing Gower distance is 
O
(
n
2
)
O(n 
2
 ) in the number of samples, the implementation uses vectorization and blocking for performance. The resulting D matrix is also saved to disk as an .npy file for record or reuse. In summary, this function converts the cleaned and prepared data into a pairwise distance representation, laying the groundwork for the subsequent clustering and visualization (MDS) steps.
classical_mds_from_distance(D, max_points, seed=42)
功能： 对给定的距离矩阵执行经典多维尺度分析（Classical MDS），将高维距离关系映射到二维坐标。该函数先在距离矩阵 D 中随机抽取最多 max_points 个样本（若样本总数不超过该阈值则全部使用），然后对这些子样本的距离矩阵应用经典MDS算法：通过特征值分解将距离矩阵转换为欧氏空间坐标。具体步骤包括：构造中心矩阵 
J
=
I
−
1
m
11
T
J=I− 
m
1
​	
 11 
T
 （
m
m 为选取的样本数），利用 
B
=
−
1
2
J
(
D
(
2
)
)
J
B=− 
2
1
​	
 J(D 
(2)
 )J 将距离矩阵平方形式转换为内积矩阵，然后对 
B
B 进行特征分解，取最大的两个特征值对应的特征向量并按 
λ
λ
​	
  比例缩放，得到二维嵌入坐标。函数返回选取的样本索引和对应的二维坐标数组。
输入：
D: NumPy二维数组，形状为 
(
n
,
n
)
(n,n)，输入的距离矩阵。
max_points: 整数，指定参与MDS降维的最大样本数。如果 
n
>
m
a
x
p
o
i
n
t
s
n>max 
p
​	
 oints，则函数将随机选择 max_points 个样本进行MDS；如果 
n
n 较小则使用全部样本。
seed: 整数，随机种子，用于在 
n
n 很大时重复选择子样本，使每次运行结果可重复（默认为42）。
输出： 二元组 (idx, X2d)：其中 idx 是所选子样本在原始距离矩阵中的索引列表（长度为 m）；X2d 是形状 
(
m
,
2
)
(m,2) 的 NumPy 数组，对应每个选中样本的二维坐标。
关键参数： 该函数实现了经典MDS的标准算法，没有额外可调参数。max_points 和 seed 用于控制抽样过程。当数据规模较大时，通过限制 max_points 可以降低 MDS 的计算量和内存消耗。此外，函数中对特征值进行了截断，只取前两个正特征值及其向量用于构造二维坐标，这是因为我们只关心二维表示。
作用： 虽然聚类是在高维空间（基于多个特征的 Gower 距离）中完成的，但为了直观展示簇间关系，我们需要将样本映射到二维平面。classical_mds_from_distance 函数为此提供了解决方案：通过MDS，我们可以得到每个样本在二维空间的位置，同时尽可能保持原始距离矩阵所表达的样本间距离关系。项目中，该函数被调用以对随机选取的至多 MDS_SUBSET（例如2000）个样本进行降维。生成的二维坐标被用于绘制散点图，每个点用颜色或标记区分簇标签，从而可以直观评估聚类效果，例如各簇是否彼此分离清晰或存在重叠。总之，此函数将距离矩阵转换为可视化友好的形式，在项目报告和结果分析中扮演辅助角色。
Function: Perform Classical Multi-Dimensional Scaling (MDS) on a given distance matrix to project the high-dimensional relationships into 2D coordinates. The function first selects at most max_points samples from the distance matrix (if the total number of samples 
n
n is larger than this threshold, it randomly chooses max_points of them; otherwise it uses all samples). It then applies the classical MDS algorithm to the distance submatrix of those samples: constructing the centering matrix 
J
=
I
−
1
m
11
T
J=I− 
m
1
​	
 11 
T
  (where 
m
m is the number of selected samples), computing 
B
=
−
1
2
J
(
D
(
2
)
)
J
B=− 
2
1
​	
 J(D 
(2)
 )J to convert the squared distance matrix into an inner-product (Gram) matrix, performing eigen-decomposition on 
B
B, and taking the top two eigenvalues and eigenvectors (scaling the eigenvectors by the square roots of the eigenvalues) to obtain coordinates in two dimensions. The function returns the indices of the selected samples and the corresponding array of 2D coordinates.
Input:
D: a NumPy 2D array of shape 
(
n
,
n
)
(n,n) representing the input distance matrix for 
n
n samples.
max_points: an integer specifying the maximum number of points to include in MDS. If 
n
>
m
a
x
p
o
i
n
t
s
n>max 
p
​	
 oints, the function will randomly choose max_points samples for the MDS computation; if 
n
n is small enough, all samples are used.
seed: an integer random seed for reproducibility when sampling points (default 42). This ensures the same subset of points is chosen each time if random sampling is needed.
Output: A tuple (idx, X2d), where idx is an array of indices (length 
m
m) for the samples chosen from the original dataset, and X2d is an 
(
m
,
2
)
(m,2) NumPy array containing the 2D coordinates for each selected sample.
Key Details: This function implements the standard classical MDS procedure without additional parameters. The max_points and seed control the random sampling aspect of the input. By limiting the number of points to max_points when the dataset is large, we reduce the computational load and memory usage of the MDS, since the algorithm involves an eigen-decomposition of an 
m
×
m
m×m matrix. The function specifically takes the top two positive eigenvalues and their eigenvectors to construct the 2D embedding, as we are interested only in a two-dimensional representation.
Role in Project: While clustering is performed in the original high-dimensional feature space (via Gower distances), it is useful to visualize the clusters in a 2D plot to intuitively assess their separation and structure. The classical_mds_from_distance function enables this by producing a two-dimensional embedding of the samples that preserves, as much as possible, the distance relationships given by the Gower matrix. In the project, this function is called to reduce a random subset of up to MDS_SUBSET samples (e.g., 2000 samples) into two dimensions. The resulting coordinates are used to create a scatter plot, coloring or labeling points by their cluster membership. This helps to visually evaluate the clustering result — for example, whether clusters form distinct groups or have significant overlap. In summary, this function transforms the distance matrix into a visualization-friendly form and plays a supporting role in interpreting and presenting the clustering outcomes.
main() (层次聚类流程主函数 | Hierarchical Clustering Pipeline Main)
功能： 脚本的主入口，统筹执行整个 Gower 距离层次聚类分析流程。从读取数据、预处理、计算距离、聚类选取最佳簇数、输出结果到绘制可视化，一系列步骤均由 main() 调用完成。
流程概述 (Inputs/Outputs)：
数据读取： 从预定义路径读取 Excel 数据文件，存入 DataFrame（df）。
价格特征处理： 检测列名中是否有包含“价/price/售价”等字样的列，并判断其中是否存在价格区间格式的数据。如有，则利用前述 extract_price_mean 函数生成平均价格列 price_mean_auto 添加到 DataFrame。
日期特征处理： 查找数据中的日期时间列，将其转换为Ordinal数值（将日期映射为整数序号）以便纳入数值特征参与聚类。新增列名为原日期列名加 _ordinal 后缀。
特征选择： 去除诸如 “id/编号/index” 等无关列。将剩余特征按数据类型拆分为数值型特征列表 numeric_cols 和分类型特征列表 categorical_cols。在此过程中应用 looks_like_city_or_level 函数：即使某分类列唯一值较多，只要列名指示其为城市/级别等，则仍保留在 categorical_cols 中。另外，如果先前生成了 price_mean_auto 列，则确保其包含在数值特征列表中。若最终选取的数值+分类特征总数为0（理论上不会发生），则抛出异常终止。
数值归一化： 计算所有数值特征的最小值和最大值，以便获取每个特征的范围 num_ranges。将 DataFrame 中的数值特征减去各自最小值，使之归一化到 [0, max-min] 区间（注意并非标准 [0,1]，但保留了相对尺度）。若某特征范围为0则后续在计算距离时特殊处理。
下采样 (Sampling)： 确定用于聚类计算的样本子集。如果数据总行数 
N
N 大于预设最大样本数 SAMPLE_MAX（例如10000），则随机无放回抽取 SAMPLE_MAX 条记录作为样本子集，否则使用全体数据。记录下采样索引，以便后续对聚类结果解释对应原数据。下采样的主要目的是控制 Gower 距离矩阵的规模在可处理范围内。
距离矩阵计算： 调用 gower_distance(X_num, X_cat, num_ranges) 计算上述样本子集的距离矩阵 D。计算完成后，将矩阵保存为文件（如 gower_distance_sample.npy）以备将来使用或验证。
簇数选择 (K selection)： 初始化一个空列表收集不同簇数下的轮廓系数。遍历预定义的簇数范围 K_RANGE（如2到9），对于每个 
k
k：
使用 AgglomerativeClustering 模型进行聚类，指定簇数为 
k
k、距离度量为预计算矩阵、链接方式为平均。注意，由于不同版本的 sklearn 参数命名差异，代码中尝试了 metric="precomputed" 或老版本的 affinity="precomputed"。
获取聚类标签 labels，基于距离矩阵计算该聚类结果的轮廓系数 score。将 
k
k 及对应轮廓系数附加到列表。跟踪当前最高的轮廓分和对应的簇数及标签。
确定最佳聚类： 根据上述循环找到轮廓系数最高的 
k
k（记为 best_k）及其对应的簇标签分配 best_labels。将所有尝试的 
k
k 及轮廓系数保存为 CSV 文件（silhouette_scores.csv）供参考。
结果保存： 将样本子集的 DataFrame 增加一列“cluster”表示所属簇，并输出为 CSV 文件（cluster_assignments_sample.csv）。随后，基于簇计算聚类结果的概要：对每个簇计算数值特征均值，保存为 cluster_numeric_profile.csv；对每个分类特征，计算各簇中每个类别取值所占比例，并提取每簇最常见的前三大类别，分别保存为文件（文件名格式如 cluster_top_categories__列名.csv）。这些概要信息方便了解每个簇的特征组成。
MDS 可视化： 调用 classical_mds_from_distance(D, max_points=MDS_SUBSET, seed=RANDOM_SEED) 获取二维坐标。然后绘制散点图，将每个点根据其簇标签着色，实现对聚类结果的二维可视化，并保存图片（例如 gower_clusters_mds_classical_subset.png）和对应坐标数据 CSV。
结束打印： 在控制台打印聚类完成的信息，包括原始总样本数与抽样数、最终采用的数值和分类特征列表、以及最佳簇数和对应的轮廓系数，以供快速查看。
作用： main() 函数串联起上述所有步骤，保证聚类分析流程按正确顺序执行。其作用相当于脚本的“大脑”，协调各子函数完成各自任务并传递数据。通过阅读 main() 的流程，我们可以清晰了解项目方法：预处理数据 -> 计算 Gower 距离矩阵 -> 尝试不同簇数的层次聚类并评估 -> 输出最佳结果及分析概要 -> 降维可视化。这一流程将原始数据成功转换为有意义的聚类划分和信息摘要，为后续业务分析提供了基础。需要注意的是，此脚本主要针对示例或抽样数据进行聚类分析（控制在 SAMPLE_MAX 条以内），如果要处理全量数据需要留意内存和计算时间开销。通过灵活调整参数如 K_RANGE、LINKAGE（链接方法）等，用户也可以进一步尝试不同聚类设定以比较效果。总之，main() 实现了整个聚类分析的自动化执行和结果导出，是连接数据处理、模型训练和结果产出的核心。
Main Function Purpose: The main() function serves as the entry point and orchestrator of the entire hierarchical clustering pipeline using Gower distance. It coordinates all steps from data loading and preprocessing to distance computation, clustering with optimal cluster selection, and finally result output and visualization.
Workflow (Inputs/Outputs):
Data Loading: Reads the Excel data file from the predefined path into a pandas DataFrame (df).
Price Feature Handling: Scans the columns for any that likely contain price information (checking for keywords like “价”, “price”, “售价”). If a price range format is detected in such a column, it applies extract_price_mean to create a new numeric column price_mean_auto with the average price values. This provides a numeric representation of price for clustering.
Date Feature Handling: Identifies any datetime columns in the data and converts them to ordinal integers (days count) by creating new columns with the suffix _ordinal. This allows date fields to be included as numeric features in clustering if needed.
Feature Selection: Removes irrelevant columns such as IDs or indices (e.g., columns containing “id”, “编号”, “index”). The remaining columns are separated into a list of numeric features numeric_cols and a list of categorical features categorical_cols. During this process, the looks_like_city_or_level function is used: normally, categorical columns with too many unique values (e.g., >200) are excluded to avoid high-dimensional sparse effects, but if a column name indicates it’s a city/level/category, it will be kept despite high cardinality. Also, if price_mean_auto was created, it is added to the numeric features list. If no features are selected at all (numeric + categorical lists empty, which should not happen in a valid dataset), the code raises an error and stops.
Numeric Normalization: Computes the minimum and maximum for each numeric feature to determine the range num_ranges. It then subtracts the minimum from each numeric column in the DataFrame, effectively normalizing all numeric features to start at 0 (so values lie in 
[
0
,
range
]
[0,range]). Note: This is a range normalization but not scaling to [0,1]; however, since clustering with Gower uses the range for each feature, this step ensures the data is adjusted accordingly (if a feature has zero range, it’s handled separately in distance computation).
Sampling: Determines the subset of samples to use for clustering computations. If the total number of records 
N
N exceeds a predefined SAMPLE_MAX (e.g., 10000), it randomly selects SAMPLE_MAX rows without replacement as the sample subset; otherwise, it uses the entire dataset. The indices of the sampled rows are stored, and a new DataFrame df_sample is created for the subset. Sampling is done to limit the size of the distance matrix for feasibility.
Distance Matrix Calculation: Calls gower_distance(X_num, X_cat, num_ranges) to compute the distance matrix D for the sampled data. After computation, the distance matrix is saved (e.g., as gower_distance_sample.npy) for potential future use or inspection.
Cluster Number Selection: Initializes a list to record silhouette scores for different cluster counts. Iterates over the range of cluster counts specified by K_RANGE (for example, 2 through 9). For each candidate 
k
k:
Fits an AgglomerativeClustering model with 
k
k clusters, using the precomputed distance matrix and average linkage. (The code accounts for version differences by trying metric="precomputed" or falling back to affinity="precomputed".)
Obtains the cluster labels for the samples and computes the silhouette score for this clustering (using the distance matrix for the “precomputed” metric in silhouette calculation). It appends the 
k
k and silhouette score to the list, and tracks the best score and corresponding 
k
k and labels as best_k and best_labels.
Determine Optimal Clusters: After the loop, identifies the cluster count best_k with the highest silhouette score and retrieves the corresponding labels best_labels. All tried 
k
k and their silhouette scores are saved to a CSV file (silhouette_scores.csv) for reference.
Result Saving: Adds a “cluster” column to the sample DataFrame (df_sample) to store the cluster label of each sample, and writes this to CSV (cluster_assignments_sample.csv). Then, it generates summary statistics for clusters: for numeric features, it computes the mean value of each feature within each cluster and saves this as cluster_numeric_profile.csv; for categorical features, it calculates the proportion of each category value within each cluster and extracts the top 3 most frequent categories per cluster, saving each as a separate CSV file (filename pattern cluster_top_categories__<ColumnName>.csv). These profiles help in interpreting the characteristics of each cluster (e.g., typical feature values and dominant categories).
MDS Visualization: Calls classical_mds_from_distance(D, max_points=MDS_SUBSET, seed=RANDOM_SEED) to obtain 2D coordinates for up to a certain number of points (e.g., 2000). It then plots a scatter diagram of these points, coloring points by their cluster label, to visualize the cluster distribution in two dimensions. The plot is saved as an image (for example, gower_clusters_mds_classical_subset.png) and the underlying 2D coordinates with cluster labels are also saved to a CSV.
Completion Log: Prints out a summary in the console, including total rows and sampled rows, the list of numeric and categorical features used, the chosen best 
K
K and its silhouette score, and the directory where outputs are saved.
Role of main(): The main() function ties together all the steps, ensuring the clustering analysis proceeds in the correct order. It effectively is the “brain” of the script, coordinating the sub-functions and passing data between them. By reading through main(), one can understand the project’s approach: prepare data -> compute Gower distance matrix -> try hierarchical clustering for various K and evaluate -> output the best clustering results and summaries -> perform dimensionality reduction for visualization. This automated sequence transforms raw data into meaningful cluster groupings and summary information, forming the basis for further business analysis. It’s worth noting that the script, by default, focuses on clustering either the entire dataset if it’s small or a random subset if the dataset is large (capped by SAMPLE_MAX for computational reasons). If applying this to the full dataset, one must be mindful of memory and time, but the structure remains the same. The main() function also allows easy adjustments of parameters like K_RANGE or LINKAGE if one wants to experiment with different clustering settings. In summary, main() executes the end-to-end clustering analysis and output generation, serving as the core driver that links data preprocessing, model fitting, and result production.
kprototypes_pycharm_run.py 脚本函数详解 | Explanation of Functions in kprototypes_pycharm_run.py
此脚本用于执行 K-Prototypes 聚类分析，适合在 PyCharm 等环境中直接运行。它涵盖了从数据预处理到模型训练、结果保存的完整流程。下面对脚本中定义的主要函数进行详细说明，包括其功能、输入输出、参数和在项目流程中的作用。
This script performs the K-Prototypes clustering analysis and is designed to be run directly (e.g., in PyCharm). It covers the full process from data preprocessing to model training and result saving. Below, we provide detailed explanations of the main functions defined in this script, including their purpose, inputs/outputs, parameters, and roles in the workflow.
parse_price(df, col)
功能： 将 DataFrame 中指定列 col 的价格区间字符串解析为数值均价。该函数会在原 DataFrame 中新增一列 price_mean_auto，其值为对于每一行 df[col] 的解析结果。解析逻辑与前述 extract_price_mean 类似：如果字符串中包含“–”或“-”表示一个区间，则取划分后的两个数字的平均；如果不包含区间符号则尝试直接转换为浮点数；解析失败则设为 NaN。
输入：
df: pandas DataFrame，包含原始数据。
col: 字符串，要解析的价格区间列名（例如 "指导价区间"）。
输出： 返回带有新列的 DataFrame（原 DataFrame 的副本亦被就地修改）。在输出的 DataFrame 中，多出一列 price_mean_auto，其值为所给列每行价格的均值。
关键参数： 无其他独立参数。函数内部定义了一个嵌套的辅助函数 get_mean(price_str)，用于对单个字符串进行解析。注意该函数会去除字符串中的“万”字等单位影响，并只处理数值部分。
作用： 数据集中如果存在表示价格范围的列，如“指导价区间”，需要将其转换为数值特征以供聚类算法使用。parse_price 封装了这一转换过程，方便重复使用。在本项目中，如果输入数据包含 "指导价区间" 列，preprocess 函数会调用 parse_price 将其转化为均价列。这样，车辆的价格信息就能以连续数值形式加入聚类分析，提高聚类对车型定位的把握。
Function: Parse price range strings in a specified column of a DataFrame into numeric average values. The function adds a new column price_mean_auto to the DataFrame, where each entry is the parsed result of df[col] for that row. The parsing logic is similar to extract_price_mean: if the string contains a “–” or “-” indicating a range, it splits the string and computes the average of the two numbers; if there is no range delimiter, it attempts to convert the string directly to a float; if parsing fails, the result is NaN. This function also handles unit characters like “万” by removing them before conversion.
Input:
df: a pandas DataFrame containing the dataset.
col: string, the column name in df that contains price ranges (for example, "指导价区间" which means "Guidance Price Range").
Output: Returns the DataFrame with a new column added (price_mean_auto). The original DataFrame is modified in place to include this column, which holds the average price for each row based on the range in col.
Key Details: There are no additional parameters besides the DataFrame and column name. Internally, the function defines a helper function get_mean(price_str) that performs the parsing for a single string. This helper removes any occurrences of “万” (ten-thousand, a common unit in Chinese prices) and handles the splitting and conversion logic. All values (including the newly created column) remain aligned with the original DataFrame rows.
Role in Project: If the dataset contains a column for price ranges (such as “指导价区间”), it needs to be converted into a numerical feature for the clustering. The parse_price function accomplishes this conversion in a reusable manner. In our project, the preprocess function will call parse_price when it detects the "指导价区间" column, thereby producing the price_mean_auto column with numeric values. This allows the vehicle price information to be included as a continuous numeric feature in clustering, improving the algorithm’s ability to group vehicles by their price level.
preprocess(df)
功能： 对原始数据 DataFrame 进行全面预处理，生成适合聚类的清洗和变换后数据。具体操作步骤：
数据拷贝： 创建原始 DataFrame 的副本 df_raw 以保留未经缩放的原始数据（特别是“上牌量”等将来可能需要以原始单位输出的列）。后续所有处理将在 df（副本）上进行。
价格区间解析： 如果存在名为“指导价区间”的列，则调用 parse_price(df, "指导价区间")（同时对 df 和 df_raw 都执行），生成平均价格列 price_mean_auto。这一步与前述函数配合，将价格区间转为数值。
缺失值填充： 遍历 DataFrame 的每一列，对于类型为对象（categorical）的列，用该列出现频率最高的值（众数）填充缺失；对于数值列，用该列的中位数填充缺失。这一步保证后续聚类不会因为缺失值而出错，同时尽量减少对数据分布的影响。df_raw 保持与 df 相同的填充，以便两者对应。
特征类型划分： 使用 df.select_dtypes 分离出数值型列列表 numeric_cols 和分类型列列表 categorical_cols。此时 numeric_cols 包括“上牌量”和可能的其他数值列（如均价等），categorical_cols 包括诸如品牌、车型类别、能源类型等字段（数据类型为 object）。
数值归一化： 初始化一个 MinMaxScaler，对所有数值列进行 [0,1] 线性缩放。这一步将“上牌量”等指标缩放到相同范围，避免原始尺度差异过大对聚类距离产生主导影响。
返回结果： 函数返回预处理后的 DataFrame df_prep（已缩放，用于聚类训练）、数值列名列表 numeric_cols、分类列名列表 categorical_cols，以及原始值保留的 DataFrame df_raw（未缩放，用于输出解释）。
输入： df：pandas DataFrame，包含原始数据（读取自 Excel）。
输出： (df_prep, numeric_cols, categorical_cols, df_raw) 元组：
df_prep：经预处理和缩放后的 DataFrame，数值列已归一化，可直接用于聚类。
numeric_cols：列表，预处理后确定的数值型特征列名。
categorical_cols：列表，预处理后确定的分类型特征列名。
df_raw：原始数据的副本，保留了填补缺失后的原始数值（未缩放）的 DataFrame。
关键参数： 该函数主要通过硬编码的逻辑处理数据，不需要外部参数。需要注意的是填充缺失值使用的众数和中位数可能会对数据分布略有影响，但这是常见处理。MinMaxScaler 没有特殊参数，此处使用默认设置将每列按列最小/最大值缩放到 [0,1]。
作用： preprocess 函数承担着为 K-Prototypes 聚类做好数据准备的任务。通过这一系列操作，原始数据中的混杂问题得到解决：价格区间转为均值、缺失值被填充、特征分门别类、数值特征缩放统一。这确保了后续 K-Prototypes 算法能在一个干净、合理尺度的数据集上运行。在项目中，main 调用 preprocess 后得到的输出 df_prep、numeric_cols、categorical_cols 将直接用于构造聚类输入矩阵。同时返回的 df_raw 保留原始尺度数据，便于在输出结果时使用原始单位（如上牌量原始值）进行说明和呈现。总之，preprocess 对数据进行了全面清洗转换，为模型训练和结果诠释打下基础。
Function: Perform comprehensive preprocessing on the raw DataFrame to produce cleaned and transformed data suitable for clustering. The steps include:
Copy Data: Make a copy of the original DataFrame as df_raw to preserve the unscaled original values (especially for fields like “registration count” which we may want to report in original units later). All subsequent transformations will be applied to df (a working copy), while df_raw will mirror certain changes like missing value imputation.
Parse Price Range: If the column "指导价区间" (guidance price range) exists in the DataFrame, call parse_price(df, "指导价区间") to create the price_mean_auto column, and do the same for df_raw to keep it consistent. This converts price ranges into numeric averages, as described earlier.
Missing Value Imputation: Iterate over each column of df. For columns with dtype object (categorical features), fill missing values with the column’s mode (most frequent value). For numeric columns, fill missing values with the median of that column. Both df and df_raw are imputed in the same way so that they remain comparable. This step prevents the clustering from failing due to missing values and minimizes distortion of data distributions.
Feature Type Segregation: Use df.select_dtypes to get the list of numeric columns numeric_cols and the list of categorical columns categorical_cols. At this point, numeric_cols would include “registration count” and any other numeric features (like the newly created price mean, etc.), and categorical_cols may include features like brand, vehicle category, fuel type, etc. (all of type object).
Normalize Numeric Features: Initialize a MinMaxScaler and apply it to all numeric columns, scaling each feature to the [0,1] range. This brings features like “registration count” onto a comparable scale, preventing features with larger original scales from dominating the distance computations in clustering.
Return Results: The function returns a tuple (df_prep, numeric_cols, categorical_cols, df_raw) where:
df_prep is the preprocessed DataFrame with numeric features scaled, ready for clustering.
numeric_cols is the list of numeric feature names included in clustering.
categorical_cols is the list of categorical feature names included in clustering.
df_raw is a copy of the data with missing values filled but not scaled, preserving original units (used later for output interpretation).
Input: df: pandas DataFrame of the raw data (read from Excel).
Output: A tuple (df_prep, numeric_cols, categorical_cols, df_raw) as described above.
Key Details: This function relies on built-in strategies for data cleaning and does not require external parameters. Notably, it fills missing values using the mode for categorical data and median for numeric data, which is a reasonable approach to avoid biasing means or adding out-of-domain values. The MinMaxScaler is used with default settings, scaling each numeric column based on its own min and max.
Role in Project: The preprocess function is responsible for preparing the data for the K-Prototypes clustering algorithm. It addresses various data issues: converting price ranges to numeric values, handling missing data, segregating feature types, and normalizing numeric features. This ensures that the K-Prototypes model receives a clean and appropriately scaled dataset, which is crucial for its performance. In the project’s workflow, main calls preprocess and uses its outputs df_prep, numeric_cols, and categorical_cols to set up the clustering input matrix. The returned df_raw with original values is kept for use during result output, so that certain results (like cluster averages of “registration count”) can be reported in original units for better interpretability. Overall, preprocess lays the groundwork by transforming raw data into a form suitable for model training and subsequent result interpretation.
run_kprototypes(df_prep, numeric_cols, categorical_cols)
功能： 使用预处理后的数据训练 K-Prototypes 聚类模型，并通过尝试多个簇数来确定最佳簇划分。具体步骤：首先将数据组合成包含数值和类别特征的单一 NumPy 矩阵 X，并生成一个表示哪些列为分类特征的索引列表 cat_idx；然后对预设的每个 
K
K 值，构建 K-Prototypes 模型并拟合数据，记录模型的聚类 成本 (cost) 值；比较不同 
K
K 的成本，选取成本最低的模型作为最佳模型，并返回相关结果。
输入：
df_prep: pandas DataFrame，预处理且数值已归一化后的数据。
numeric_cols: 列表，数值特征列名列表。
categorical_cols: 列表，分类特征列名列表。
输出： (best_k, best_labels, best_model, cost_records) 元组：
best_k: 整数，成本最小的最佳簇数 
K
K。
best_labels: NumPy数组，长度为样本数，每个元素是对应样本的簇标签（0 ~ K-1），来自最佳模型。
best_model: 训练后的 KPrototypes 模型对象（kmodes.KPrototypes 类型），对应最佳 
K
K。
cost_records: 字典，各候选 
K
K 值对应的成本，如 {6: 1234.5, 7: 1100.8, ...}。
关键参数： 该函数内部使用了 K_LIST 常量作为候选簇数量列表（例如 [6,7,...,15]）。在训练模型时，每次迭代均使用 init="Huang" 初始化中心、n_init=5 进行5次随机重启动、random_state=42 固定随机种子保证可重复性。cat_idx 列表标记了矩阵 X 中哪些列是分类型，这是 KPrototypes 模型所需的参数，用于应用匹配距离计算。
作用： 该函数是 K-Prototypes 聚类的执行核心。一方面，它将 pandas DataFrame 转换为了 K-Prototypes 可直接处理的 NumPy 矩阵形式，并明确分类特征的位置索引；另一方面，它通过遍历多个可能的簇数进行聚类训练，并以模型的 cost 值作为评价标准选优。在本项目中，我们以 cost（簇内距离和）最小作为最佳簇划分的判据，这类似于肘部法但选择了成本最低点。run_kprototypes 返回的最佳模型及标签将用于后续的结果输出和分析。例如，best_labels 会被添加回 DataFrame 以标记每条记录所属簇，cost_records 则用于绘制 K vs. Cost 曲线观察成本随簇数的变化趋势。总之，run_kprototypes 封装了模型训练和参数选择逻辑，使主流程能够方便地获取最优聚类方案。
Function: Train the K-Prototypes clustering model on the preprocessed data and determine the optimal clustering by trying multiple values of 
K
K (number of clusters). The function first concatenates the numeric and categorical parts of the data into a single NumPy matrix X and creates an index list cat_idx indicating which columns of X are categorical. Then, for each candidate
K
K in a predefined list, it initializes and fits a K-Prototypes model on X, records the clustering cost for that 
K
K, and tracks the model with the lowest cost. Finally, it returns the results related to the best clustering found.
Input:
df_prep: pandas DataFrame of the preprocessed data (with numeric features scaled).
numeric_cols: list of names of numeric feature columns.
categorical_cols: list of names of categorical feature columns.
Output: A tuple (best_k, best_labels, best_model, cost_records) where:
best_k: an integer representing the number of clusters 
K
K that yielded the lowest cost.
best_labels: a NumPy array of cluster labels (0 to K-1) for each data point, from the best model.
best_model: the trained KPrototypes model object corresponding to best_k.
cost_records: a dictionary mapping each tried 
K
K to its cost (e.g., {6: 1234.5, 7: 1100.8, ...}).
Key Details: This function uses an internal constant K_LIST which holds the list of cluster counts to try (for example, [6, 7, ..., 15]). When fitting the model for each 
K
K, it uses specific parameters: init="Huang" for initialization (Huang’s method for stability with categorical data), n_init=5 to run 5 random starts and choose the best, and random_state=42 for reproducibility. The cat_idx list marks the positions of categorical features in matrix X, which is required by the KPrototypes model to apply the correct distance calculation for those columns. Assertions in the code ensure that X_num is float and X_cat is string type, and that the indices in cat_idx are within bounds.
Role in Project: This function is the core of executing the K-Prototypes clustering. It not only transforms the DataFrame into the format needed for the algorithm (NumPy matrix with categorical indices) but also encapsulates the logic of model training and model selection by cluster count. In this project, we use the total cost (within-cluster sum of distances) as the criterion to choose the best number of clusters — essentially selecting the 
K
K with the lowest cost among those tested. The output best_model and its best_labels are then used for subsequent result reporting. For instance, best_labels will be added to the DataFrame to label each record’s cluster, and cost_records will be used to plot the K vs. Cost curve to visualize how the cost changes as K increases. In summary, run_kprototypes handles the heavy lifting of clustering the data and determining the optimal clusters, making it easy for the main workflow to obtain the best clustering configuration.
save_outputs(df, labels, numeric_cols, categorical_cols, cost_records, best_k, model, df_raw)
功能： 将聚类结果和相关分析输出到文件，包括：完整数据的簇标签、数值特征簇均值（两种版本）、分类特征 Top-N 分布、K vs Cost 数据曲线、以及元信息 JSON。通过将结果保存到指定输出目录，方便后续对聚类结果的查看和分享。主要输出内容如下：
聚类分配结果： 在 df（缩放数据）和 df_raw（原始数据）中分别追加簇标签列“cluster”。将 df_raw（含cluster）输出为 CSV 文件（如 kprototypes_cluster_assignments_full.csv），此文件包含每条记录及其所属簇标签（以原始值便于阅读）。
数值特征簇均值： 计算并保存两份簇级数值特征概要：
缩放版均值： 基于 df，按簇计算所有数值特征的平均值，输出 CSV 文件 kprototypes_cluster_numeric_profile_scaled.csv。这些均值在0-1尺度上。
混合版均值： 在缩放均值的基础上，将其中“上牌量”一列替换为 df_raw 中对应簇的原始均值，以保留“上牌量”的真实数量单位。结果输出 kprototypes_cluster_numeric_profile_mixed.csv。这种混合报告使得除“上牌量”外其他数值仍在0-1范围，但上牌量以实际数量体现，方便解读。
分类特征 Top-10： 对每个分类特征列，计算每个簇中该类别值出现的频数以及占比，并取每簇频数前10的类别组成字典。将每个特征的结果输出为 CSV 文件（命名如 kprototypes_cluster_top_categories__列名.csv）。文件内容包含簇号以及对应的 Top10 类别及频数（及百分比）。
K vs Cost 曲线： 将 cost_records 字典转换为 DataFrame（包含 K 值及对应成本），输出为 CSV 文件 kprototypes_cost_by_k.csv。同时绘制成本随 K 的折线图并保存为图片 kprototypes_cost_curve.png，用于观察聚类成本的变化趋势。
元信息 JSON： 将聚类的一些关键信息（如 best_k, 使用的 numeric_cols 和 categorical_cols, cost_records, 备注等）保存为 JSON 文件 kprototypes_meta.json。特别地，备注中说明了聚类使用的是归一化数据，以及 numeric_profile_mixed 中“上牌量”已还原为原始单位。
输入：
df: pandas DataFrame，预处理并缩放后的数据（聚类用），函数假定已经包含最新的簇标签列。
labels: NumPy数组，聚类标签，长度与 df 行数相同。
numeric_cols: 列表，数值特征列名列表。
categorical_cols: 列表，分类特征列名列表。
cost_records: 字典，各尝试的 K 值及对应聚类成本。
best_k: 整数，最优簇数。
model: 训练好的 KPrototypes 模型对象（可从中获取聚类中心等信息，如果需要）。
df_raw: pandas DataFrame，原始数据填充缺失后的版本。应与 df 有相同的行索引，并未缩放，用于输出原始值。
输出： 无显式返回值。函数执行后，会在 OUTPUT_DIR（脚本顶部配置的输出目录，例如 ./out_kproto）下生成多个文件，如上述CSV、PNG、JSON等，涵盖聚类结果和分析。
关键参数： 函数内部使用了 os.makedirs 确保输出目录存在。保存输出时文件名固定（按约定构造）。绘图使用 matplotlib，简单绘制了 K 对应成本的曲线并保存。在 Top-10 类别计算中，对每个簇采用 value_counts().head(10) 获得前十类别，并用字典格式表示频数和比例。
作用： save_outputs 将模型的结果及评估信息完整地记录下来，是项目产出的最终一步。通过这个函数，聚类分析的成果被保存为多份报告：既有逐个样本的聚类标签明细，也有聚类整体的统计概要，还有模型评估曲线和配置元数据。这些输出文件便于项目参与者和其他读者理解聚类的情况。例如，业务人员可以直接查看 cluster_assignments_full.csv 了解每款车型（或记录）的簇归属，数据分析人员可以根据 cluster_numeric_profile_mixed.csv 和各 Top-10 类别文件了解每个簇的特征特性。成本曲线和 JSON 元信息则有助于记录模型选择的依据和过程。总之，save_outputs 汇总并输出了聚类分析的各方面结果，为报告撰写和后续应用提供了便利。
Function: Save the clustering results and related analyses to files, including: cluster labels for the full dataset, cluster mean profiles for numeric features (in two versions), top-N category distributions for categorical features, the K vs Cost curve data, and a metadata JSON. These outputs are written to a designated output directory, making it easy to review and share the clustering results. The main outputs are:
Cluster Assignments: Adds the cluster label column “cluster” to both the scaled DataFrame df and the original-value DataFrame df_raw. It then saves df_raw (with cluster labels) as a CSV file (e.g., kprototypes_cluster_assignments_full.csv), so each record with its original feature values and assigned cluster can be examined. (Using original values in the output makes it more interpretable).
Numeric Feature Cluster Means: Computes and saves two versions of cluster-wise numeric summaries:
Scaled Means: Using df (scaled data), calculate the mean of each numeric feature for each cluster, and save to kprototypes_cluster_numeric_profile_scaled.csv. These means are on the normalized 0-1 scale.
Mixed Means: Make a copy of the scaled means, then replace the “registration count” column’s values with the mean calculated from df_raw for that cluster (thus using the original unit for that one feature). Save this as kprototypes_cluster_numeric_profile_mixed.csv. In this mixed profile, “registration count” is in the original scale (e.g., number of vehicles), whereas other numeric features remain on 0-1 scale. This makes the key metric more understandable while still showing relative values of other features.
Categorical Features Top-10: For each categorical feature column, determine the top 10 most frequent categories within each cluster along with their counts and percentages. Save each result as a CSV file (named like kprototypes_cluster_top_categories__<ColumnName>.csv). Each such file lists, for each cluster, the top categories and the count (and proportion) of each.
K vs Cost Data: Convert the cost_records (a dictionary of K to cost) into a pandas DataFrame and save it as kprototypes_cost_by_k.csv. Also, plot the cost against K as a line chart and save the figure as kprototypes_cost_curve.png. This curve helps visualize how the clustering cost decreases as the number of clusters increases.
Metadata JSON: Save key information about the clustering run in a JSON file kprototypes_meta.json. This includes best_k, the lists of numeric_cols and categorical_cols used, the full cost_records, and a note explaining that clustering was done on scaled data and that in numeric_profile_mixed the “registration count” is shown in original units.
Input:
df: pandas DataFrame of the scaled data used for clustering. This is expected to already have the cluster labels column added (the function adds it if not).
labels: NumPy array of cluster labels for each sample (length equal to number of rows in df).
numeric_cols: list of numeric feature names.
categorical_cols: list of categorical feature names.
cost_records: dictionary of costs for each tried K.
best_k: integer, the optimal number of clusters determined.
model: the trained KPrototypes model object for the best clustering (from which one could extract cluster centroids if needed).
df_raw: pandas DataFrame of the original data (with missing values filled), having the same number of rows as df, used to output values in original units.
Output: This function does not return a value. Instead, it creates multiple output files in the OUTPUT_DIR (which is defined at the top of the script, e.g., ./out_kproto). The files include the CSVs, PNG, and JSON as described above, containing the clustering results and analysis.
Key Details: The function uses os.makedirs to ensure the output directory exists. Filenames are predefined according to a consistent naming scheme. The cost curve is plotted using matplotlib with K on the x-axis and cost on the y-axis, and saved to disk. In computing the top-10 categories, the code uses a combination of groupby and apply to generate a dictionary of top 10 values with their counts and percentages for each cluster. Each cluster’s result is then saved as a row in the respective CSV, which might need interpretation (the dictionary is saved as a string representation).
Role in Project: The save_outputs function consolidates and writes out all relevant results of the clustering analysis, making it the final step of the project pipeline. Through this function, the outcomes of clustering are documented in various forms: a detailed assignment of each record to a cluster, summary profiles of each cluster’s characteristics, evaluation plots, and metadata. These outputs are crucial for stakeholders to understand and utilize the clustering results. For instance, a business analyst can inspect kprototypes_cluster_assignments_full.csv to see which cluster each vehicle (or record) belongs to; data analysts can use kprototypes_cluster_numeric_profile_mixed.csv and the top-categories files to interpret the defining features of each cluster (e.g., what is the average price or most common vehicle type in Cluster 2?); the cost curve and JSON info help record how the clustering solution was chosen and what parameters were used. In summary, save_outputs serves to package the clustering results in a comprehensive and accessible way, facilitating reporting and any downstream use of the clustered data.
main (K-Prototypes 聚类主流程)
功能： 脚本的主程序段，将上述各函数串联执行，实现 K-Prototypes 聚类分析的完整流程。主要步骤包括：读取数据、预处理、训练模型（含选定最佳簇数）、保存结果。
执行流程：
读取 Excel： 利用 pd.read_excel 加载输入的 Excel 数据文件到 DataFrame df。文件路径和输出目录在脚本开头通过常量设定（如 INPUT_EXCEL 和 OUTPUT_DIR）。
数据预处理： 调用 preprocess(df) 对数据进行清洗和特征准备，获取 df_prep, numeric_cols, categorical_cols, df_raw。此时数据已填补缺失，数值特征已归一化，特征列表也分类明确。
训练 K-Prototypes 模型： 调用 run_kprototypes(df_prep, numeric_cols, categorical_cols) 进行聚类训练和簇数选择，获取 best_k, labels, best_model, cost_records。根据设定的 K 列表自动挑选最佳模型。
导出结果： 调用 save_outputs(df_prep, labels, numeric_cols, categorical_cols, cost_records, best_k, model, df_raw) 保存所有聚类结果文件。
（打印信息）： 在 save_outputs 内部或之后，会打印一行总结，注明最优簇数及输出位置（例如控制台输出 ✅ 最优 K=...，输出已保存等）。
作用： 与前一个脚本类似，K-Prototypes 脚本的 main 部分（在 if __name__ == "__main__": 下）负责按照顺序执行各步骤，将数据输入转化为聚类输出。由于所有功能都已封装为函数，main 中逻辑清晰，易于理解和修改。通过阅读 main，可以了解该项目使用 K-Prototypes 进行聚类的具体过程和配置。例如，可以方便地调整 K_LIST 以尝试不同的簇范围，或者修改输入文件路径以应用于新的数据。总之，main 把数据管道连接起来，驱动完成 K-Prototypes 聚类分析并输出成果。用户在使用本脚本时，只需确保配置好路径参数，然后运行脚本即可生成上述所有结果文件。
Main Program Execution: The main block of the script (under if __name__ == "__main__":) sequences the execution of the above functions to carry out the full K-Prototypes clustering process. The steps are:
Read Excel: Load the input Excel file into a DataFrame df using pd.read_excel. (The path of the file and output directory are defined as constants at the top, e.g., INPUT_EXCEL and OUTPUT_DIR.)
Preprocess Data: Call preprocess(df) to clean and prepare the data, obtaining df_prep, numeric_cols, categorical_cols, df_raw. At this point, missing values are handled, numeric features are scaled, and we have lists of feature names by type.
Train K-Prototypes: Call run_kprototypes(df_prep, numeric_cols, categorical_cols) to perform the clustering and find the optimal cluster count, receiving best_k, labels, best_model, cost_records. This automates the model training and selection of the best solution according to the predefined K list.
Save Outputs: Call save_outputs(df_prep, labels, numeric_cols, categorical_cols, cost_records, best_k, model, df_raw) to write all the result files to disk.
(Console log): Inside save_outputs or after it, the script prints a summary line indicating the best K found and where the results have been saved (e.g., a message like “✅ Best K=X, outputs saved to ...”).
Role: Similar to the first script, the main section here orchestrates the entire pipeline for K-Prototypes clustering, converting raw data input into final outputs. Since all functionality is modularized into functions, the logic in main is straightforward and easy to follow or modify. By inspecting main, one can understand how the K-Prototypes method is applied: data is read and preprocessed, the clustering is run with a range of K values, and results are saved. This design makes it simple to tweak parameters (for instance, adjust K_LIST to try a different range of clusters, or change file paths to run on a new dataset). In summary, main connects the data pipeline and drives the completion of the K-Prototypes clustering analysis, producing all the output artifacts. To use this script, one mainly needs to set the correct input file path and desired parameters at the top, then run the script to automatically generate all the result files as described.
