# Exploratory-Research-of-Car-Registration-in-China
This data analysis project explores the potential pattern of the number of car registration in one month in China by clustering, Association Rule Mining and Outlier Detection.

中国汽车2025年5月上牌量聚类分析 – 技术说明 (中英文双语)

项目方法整体概述 | Project Methodology Overview

中国汽车2025年5月上牌量聚类分析项目旨在利用无监督学习方法从大量车辆上牌数据中发现潜在模式和类别。数据集包含了多种类型的特征，包括数值特征（如上牌量、价格等）和分类特征（如车辆类型、燃料类别、城市等）。由于特征类型多样，我们采用了两种聚类方法：基于 Gower 距离的层次聚类 和 K-Prototypes 聚类算法，分别针对混合数据类型的距离计算和直接聚类进行优化。项目流程包括数据预处理、使用两种方法分别进行聚类分析、选择适当的簇数，以及结果输出与简单可视化。通过这两种方法的对比，能够从不同角度刻画2025年5月汽车市场上牌数据的聚类模式，为进一步的市场细分和业务决策提供支持。


聚类方法原理解释 | Clustering Methods and Principles

基于 Gower 距离的层次聚类原理 | Hierarchical Clustering with Gower Distance

Gower 距离是一种针对混合类型数据的相似度度量，适用于包含数值、分类等不同类型特征的数据。对于任意两个样本 i 和 j，Gower 距离首先计算每个特征上的差异，再综合得到总体差异。具体而言，对每个特征 k 定义距离 d^{(k)}(i,j)：若特征 k 为数值型，则采用归一化差值 d^{(k)}(i,j) = \frac{|x_{ik} - x_{jk}|}{R_k}，其中 R_k 是特征 k 在数据中的范围（最大值减去最小值）；若特征 k 为分类型，则 d^{(k)}(i,j) = 0 （当 x_{ik} = x_{jk}）或 1（当 x_{ik} \neq x_{jk}）。如果某特征在某个样本中缺失，则该特征对该对样本的不相似度不予计算。对于所有特征，Gower 不相似度定义为各特征距离的加权平均：

$$
d_{ij}^{\text{(Gower)}} = \frac{\sum_{k=1}^{p} w_k^{(ij)} , d^{(k)}(i,j)}{\sum_{k=1}^{p} w_k^{(ij)}} ,,
$$

其中 p 为特征总数，权重 w_k^{(ij)} = 1 表示样本 i,j 在第 k 个特征上均有值（非缺失），若任一缺失则 w_k^{(ij)} = 0。通过上述公式得到的 Gower 距离介于0和1之间，数值越大表示样本差异越大。在本项目中，我们使用 Gower 距离矩阵作为层次聚类算法的输入，对混合型数据计算严谨的样本间距离。

层次聚类采用凝聚式（自底向上）策略：开始时每个样本各自成为一类，不断合并最近的簇直至达到预定的簇数 K。我们选择“平均链接（average linkage）”作为聚类准则，即定义两个簇 A 和 B 之间的距离为簇间所有成对样本距离的平均值：

$$
d(A, B) = \frac{1}{|A| \cdot |B|} \sum_{i \in A} \sum_{j \in B} d(i,j) ,,
$$

其中 d(i,j) 是样本 i 和 j 之间的距离（在本项目中为 Gower 距离）。平均链接方法能够综合考虑簇间所有样本的距离，从而生成相对均衡的簇。利用预先计算的 Gower 距离矩阵，本项目通过 AgglomerativeClustering 实现层次聚类，将 metric="precomputed" 与平均链接结合，以直接使用自定义距离矩阵。我们针对簇数 K 进行尝试（例如 K=2 到 9），使用轮廓系数(silhouette score)评估每种簇划分的质量，从中选取轮廓系数最高的 K 作为最佳聚类数量。层次聚类的结果包括每个样本的簇标签，以及基于簇的特征概况分析，如数值特征均值和主要分类特征分布等。此外，我们对距离矩阵采样应用经典多维尺度法 (MDS) 将高维距离映射到二维平面，以可视化聚类结果的整体结构。


K-Prototypes 聚类算法原理 | K-Prototypes Clustering Algorithm

K-Prototypes 算法是针对混合数据（同时含有数值和类别特征）的一种划分式聚类方法，由 Huang 在1998年提出。它综合了 K-Means 和 K-Modes 算法的思想：针对数值型特征，使用与 K-Means 相同的距离度量（通常为欧氏距离的平方）；针对分类特征，使用简单匹配不相似度（类别不同则距离为1，相同则为0），类似 K-Modes 中对类别变量的处理。为了将两种不同类型特征的差异合成为一个统一的距离度量，K-Prototypes 引入了一个可调权重参数 \gamma，用于平衡数值距离和类别距离的影响。对于给定数据点 x 和簇中心（原型） c，定义 K-Prototypes 距离为：

$$
d_{\text{kproto}}(x, c) = \sum_{j=1}^{m} (x_j - c_j)^2 ;+; \gamma \sum_{l=1}^{r} \mathbf{1}(x’_l \neq c’_l),,
$$

其中前一项求和覆盖所有数值特征 j（共有 m 个，使用数值差的平方），后一项覆盖所有分类特征 l（共有 r 个，\mathbf{1}(\cdot) 为指示函数，当分类值不相同时取1，相同时取0），c_j 和 c’_l 分别是簇中心在对应数值和分类特征上的值。参数 \gamma 权衡了类别不匹配的惩罚幅度，防止类别特征对距离的贡献被数值特征掩盖（用户可根据数据特征尺度调整 \gamma，也可使用默认估计值）。聚类的目标是最小化数据点到其所属簇原型的上述距离之和，即最小化总的簇内距离代价。

K-Prototypes 算法的迭代过程与 K-Means 类似：
1.	初始化 – 随机选择 K 个初始原型（包括数值均值部分和分类模式部分），或使用 Huang 提出的改进初始化方法。
2.	分配簇 – 将每个数据点分配到与其距离 d_{\text{kproto}} 最近的原型所属的簇。
3.	更新原型 – 对每个簇，重新计算新的原型：数值特征取簇内所有点的均值，分类特征取簇内出现频率最高的类别（模式）。
4.	重复执行“分配簇”和“更新原型”步骤，直到簇分配不再变化或达到预定的迭代次数。算法收敛后得到稳定的簇划分结果。

在本项目中，我们使用 kmodes 库提供的实现 (KPrototypes 类) 来对数据执行 K-Prototypes 聚类。在预处理阶段，我们将所有数值特征归一化到 [0,1] 区间，这样可以减少选择 \gamma 时的难度，因为所有特征的数值尺度相近。模型训练过程中，我们针对一组候选簇数 K 值（如 6 至 15）反复运行聚类，将 成本 (cost)（即簇内距离的总和）作为评价指标。随着 K 增大，聚类成本通常会下降，我们选择其中成本最小的簇数作为最佳 K。聚类完成后，每个样本获得一个簇标签，我们可以进一步提取每个簇的特征画像，包括数值特征的均值（本项目中特别保留“上牌量”以原始单位呈现簇均值）和每个类别特征中占比最高的几种类别。K-Prototypes 聚类方法能够高效地处理大规模数据，其直接迭代优化的机制使其在混合数据聚类中表现良好。

cluster_gower_pipeline.py 脚本函数详解 | Explanation of Functions in 
cluster_gower_pipeline.py

该脚本实现了基于 Gower 距离和层次聚类的完整分析流程，从数据预处理到结果输出。下面对脚本中各主要函数的作用、输入输出、关键参数及在项目中的用途进行逐一说明。

This script implements the full analysis pipeline using Gower distance and hierarchical clustering, covering data preprocessing through result output. The following is a detailed explanation of each major function in the script, including its purpose, inputs/outputs, key parameters, and role in the project.

extract_price_mean(s)
•	功能： 提取价格区间字符串中的数值并计算均值。如果输入字符串包含形如 “A-B” 的价格区间，则返回区间中两个数值的平均值；如果只包含单一数值，则直接返回该数值；如果无法提取数字则返回 NaN。该函数的实现通过正则表达式搜索字符串中的数值，并进行简单的字符串替换来处理可能存在的不同破折号字符。

•	输入： s：表示价格的字符串（可能为区间或单值），或缺失值 NaN。

•	输出： float 类型的平均价格值。如果输入为区间字符串，输出为区间两端数值的平均；如果输入为单个数字字符串，则输出该数字转换的浮点值；无法提取数字时输出 NaN。

•	关键参数： 无其他参数；函数内部定义了对中文格式破折号的替换和对字符串中数字的提取逻辑。

•	作用： 本项目数据包含车辆指导价区间等信息，该函数用于将价格区间转换为可用于分析的数值型均价。例如，对于 “9.98-12.98万” 这样的价格区间字符串，函数会返回 11.48（单位为万），从而为后续聚类提供一个代表性价格数值。在脚本中，extract_price_mean 被应用于检测数据列中是否存在价格区间格式，并自动生成一个新的数值列“price_mean_auto”，以丰富数值特征集合。

looks_like_city_or_level(name)
•	功能： 判断给定的列名是否可能代表“城市、级别、车型、类别”等分类属性。实现方式是将列名转换为小写字符串，并检查其中是否包含特定关键词（“城市”，“级别”，“车型”，“类别”）。如果包含任一关键词则返回 True，否则返回 False。

•	输入： name：字符串，表示数据列的名称。

•	输出： 布尔值。True 表示列名看起来属于城市/级别/车型/类别这类特征，False 表示否。

•	关键参数： 函数内部定义了关键词列表 keys = ["城市", "级别", "车型", "类别"] 作为识别标志。比较时不区分大小写。

•	作用： 在项目的数据预处理阶段，需要筛选哪些列作为分类特征参与聚类。一般而言，高基数（unique 值很多）的分类变量在聚类中作用有限且计算代价高，脚本中采用了阈值（如唯一值超过200则不考虑）的策略。然而，一些重要的分类信息如城市、车辆级别等可能具有较多唯一值但仍有分析价值。looks_like_city_or_level 函数用于识别这类列名，以便在选择分类特征时进行例外处理——即使唯一值数量较多，只要列名包含上述关键词之一，仍将其视作分类特征纳入聚类分析。这确保了关键业务维度（地域、车型类别等）不会因取值种类多而被忽略。


gower_distance(X_num, X_cat, num_ranges)
•	功能： 基于前述的 Gower 公式计算一组混合类型数据的距离矩阵。该函数接受数值特征矩阵和分类特征矩阵，逐对计算所有样本之间的 Gower 距离。实现细节包括：对每个数值特征列，按列范围归一化差值累加距离，并记录有效比较的权重计数；对每个分类特征列，比较类别是否相同，不同则距离计为1，相同为0，并相应累加距离和权重。最后将距离累加矩阵按权重平均，得到样本两两之间的平均距离。函数返回一个大小为 n \times n 的对称距离矩阵（numpy 数组）。

•	输入：

o	X_num: NumPy数组，形状为 (n, p)，包含 n 个样本的 p 个数值特征。如果没有数值特征则传入 None。

o	X_cat: NumPy数组，形状为 (n, q)，包含 n 个样本的 q 个分类特征（类型为 object 或字符串）。如果没有分类特征则传入 None。

o	num_ranges: NumPy数组，长度为 p，对应每个数值特征的取值范围（max-min）。用于归一化数值距离。如果某特征范围为0（所有值相等），函数内部会将其视为1以避免除零。

•	输出： NumPy二维数组 D，形状为 (n, n)，表示 n 个样本之间的两两距离。矩阵为对称矩阵，且对角线为0。

•	关键参数： 函数没有显式其他参数，但内部使用了分块计算 (block 的大小512) 来处理大型矩阵以兼顾效率和内存。另外，函数在计算过程中分别累积了数值部分的距离和权重矩阵 W_num，以及分类部分的距离和权重矩阵 W_cat，最后将两者相加得到总距离矩阵和总权重矩阵。在合并结果时，对那些在所有特征上均无有效比较的样本对（权重和为0）设定权重为1，以避免除以0 的情况。

•	作用： 本函数是实现 Gower 距离聚类的核心。它将混合型特征的数据转化为可用于聚类算法的距离度量。在项目中，我们对抽样后的数据调用 gower_distance 得到距离矩阵 D。随后层次聚类算法以此矩阵为依据进行聚类，因此 gower_distance 的准确性和效率直接影响聚类结果。由于 Gower 距离计算代价与 O(n^2) 级别的样本对数相关，为兼顾速度，函数实现中采用了向量化和分块处理策略。结果矩阵 D 被保存为 .npy 文件以备查。总而言之，该函数将预处理后的数据转换为距离矩阵，是后续基于距离的聚类和可视化分析（如 MDS）的基础。

classical_mds_from_distance(D, max_points, seed=42)

•	功能： 对给定的距离矩阵执行经典多维尺度分析（Classical MDS），将高维距离关系映射到二维坐标。该函数先在距离矩阵 D 中随机抽取最多 max_points 个样本（若样本总数不超过该阈值则全部使用），然后对这些子样本的距离矩阵应用经典MDS算法：通过特征值分解将距离矩阵转换为欧氏空间坐标。具体步骤包括：构造中心矩阵 J = I - \frac{1}{m}\mathbf{1}\mathbf{1}^T（m 为选取的样本数），利用 B = -\frac{1}{2} J (D^{(2)}) J 将距离矩阵平方形式转换为内积矩阵，然后对 B 进行特征分解，取最大的两个特征值对应的特征向量并按 \sqrt{\lambda} 比例缩放，得到二维嵌入坐标。函数返回选取的样本索引和对应的二维坐标数组。

•	输入：

o	D: NumPy二维数组，形状为 (n, n)，输入的距离矩阵。

o	max_points: 整数，指定参与MDS降维的最大样本数。如果 n > max_points，则函数将随机选择 max_points 个样本进行MDS；如果 n 较小则使用全部样本。

o	seed: 整数，随机种子，用于在 n 很大时重复选择子样本，使每次运行结果可重复（默认为42）。

•	输出： 二元组 (idx, X2d)：其中 idx 是所选子样本在原始距离矩阵中的索引列表（长度为 m）；X2d 是形状 (m, 2) 的 NumPy 数组，对应每个选中样本的二维坐标。

•	关键参数： 该函数实现了经典MDS的标准算法，没有额外可调参数。max_points 和 seed 用于控制抽样过程。当数据规模较大时，通过限制 max_points 可以降低 MDS 的计算量和内存消耗。此外，函数中对特征值进行了截断，只取前两个正特征值及其向量用于构造二维坐标，这是因为我们只关心二维表示。

•	作用： 虽然聚类是在高维空间（基于多个特征的 Gower 距离）中完成的，但为了直观展示簇间关系，我们需要将样本映射到二维平面。classical_mds_from_distance 函数为此提供了解决方案：通过MDS，我们可以得到每个样本在二维空间的位置，同时尽可能保持原始距离矩阵所表达的样本间距离关系。项目中，该函数被调用以对随机选取的至多 MDS_SUBSET（例如2000）个样本进行降维。生成的二维坐标被用于绘制散点图，每个点用颜色或标记区分簇标签，从而可以直观评估聚类效果，例如各簇是否彼此分离清晰或存在重叠。总之，此函数将距离矩阵转换为可视化友好的形式，在项目报告和结果分析中扮演辅助角色。

main()
 (层次聚类流程主函数 | Hierarchical Clustering Pipeline Main)
 
•	功能： 脚本的主入口，统筹执行整个 Gower 距离层次聚类分析流程。从读取数据、预处理、计算距离、聚类选取最佳簇数、输出结果到绘制可视化，一系列步骤均由 main() 调用完成。

•	流程概述 (Inputs/Outputs)：

1.	数据读取： 从预定义路径读取 Excel 数据文件，存入 DataFrame（df）。
2.	价格特征处理： 检测列名中是否有包含“价/price/售价”等字样的列，并判断其中是否存在价格区间格式的数据。如有，则利用前述 extract_price_mean 函数生成平均价格列 price_mean_auto 添加到 DataFrame。
3.	日期特征处理： 查找数据中的日期时间列，将其转换为Ordinal数值（将日期映射为整数序号）以便纳入数值特征参与聚类。新增列名为原日期列名加 _ordinal 后缀。
4.	特征选择： 去除诸如 “id/编号/index” 等无关列。将剩余特征按数据类型拆分为数值型特征列表 numeric_cols 和分类型特征列表 categorical_cols。在此过程中应用 looks_like_city_or_level 函数：即使某分类列唯一值较多，只要列名指示其为城市/级别等，则仍保留在 categorical_cols 中。另外，如果先前生成了 price_mean_auto 列，则确保其包含在数值特征列表中。若最终选取的数值+分类特征总数为0（理论上不会发生），则抛出异常终止。
5.	数值归一化： 计算所有数值特征的最小值和最大值，以便获取每个特征的范围 num_ranges。将 DataFrame 中的数值特征减去各自最小值，使之归一化到 [0, max-min] 区间（注意并非标准 [0,1]，但保留了相对尺度）。若某特征范围为0则后续在计算距离时特殊处理。
6.	下采样 (Sampling)： 确定用于聚类计算的样本子集。如果数据总行数 N 大于预设最大样本数 SAMPLE_MAX（例如10000），则随机无放回抽取 SAMPLE_MAX 条记录作为样本子集，否则使用全体数据。记录下采样索引，以便后续对聚类结果解释对应原数据。下采样的主要目的是控制 Gower 距离矩阵的规模在可处理范围内。
7.	距离矩阵计算： 调用 gower_distance(X_num, X_cat, num_ranges) 计算上述样本子集的距离矩阵 D。计算完成后，将矩阵保存为文件（如 gower_distance_sample.npy）以备将来使用或验证。
8.	簇数选择 (K selection)： 初始化一个空列表收集不同簇数下的轮廓系数。遍历预定义的簇数范围 K_RANGE（如2到9），对于每个 k：
	使用 AgglomerativeClustering 模型进行聚类，指定簇数为 k、距离度量为预计算矩阵、链接方式为平均。注意，由于不同版本的 sklearn 参数命名差异，代码中尝试了 metric="precomputed" 或老版本的 affinity="precomputed"。
	获取聚类标签 labels，基于距离矩阵计算该聚类结果的轮廓系数 score。将 k 及对应轮廓系数附加到列表。跟踪当前最高的轮廓分和对应的簇数及标签。
9.	确定最佳聚类： 根据上述循环找到轮廓系数最高的 k（记为 best_k）及其对应的簇标签分配 best_labels。将所有尝试的 k 及轮廓系数保存为 CSV 文件（silhouette_scores.csv）供参考。
10.	结果保存： 将样本子集的 DataFrame 增加一列“cluster”表示所属簇，并输出为 CSV 文件（cluster_assignments_sample.csv）。随后，基于簇计算聚类结果的概要：对每个簇计算数值特征均值，保存为 cluster_numeric_profile.csv；对每个分类特征，计算各簇中每个类别取值所占比例，并提取每簇最常见的前三大类别，分别保存为文件（文件名格式如 cluster_top_categories__列名.csv）。这些概要信息方便了解每个簇的特征组成。
11.	MDS 可视化： 调用 classical_mds_from_distance(D, max_points=MDS_SUBSET, seed=RANDOM_SEED) 获取二维坐标。然后绘制散点图，将每个点根据其簇标签着色，实现对聚类结果的二维可视化，并保存图片（例如 gower_clusters_mds_classical_subset.png）和对应坐标数据 CSV。
12.	结束打印： 在控制台打印聚类完成的信息，包括原始总样本数与抽样数、最终采用的数值和分类特征列表、以及最佳簇数和对应的轮廓系数，以供快速查看。

•	作用： main() 函数串联起上述所有步骤，保证聚类分析流程按正确顺序执行。其作用相当于脚本的“大脑”，协调各子函数完成各自任务并传递数据。通过阅读 main() 的流程，我们可以清晰了解项目方法：预处理数据 -> 计算 Gower 距离矩阵 -> 尝试不同簇数的层次聚类并评估 -> 输出最佳结果及分析概要 -> 降维可视化。这一流程将原始数据成功转换为有意义的聚类划分和信息摘要，为后续业务分析提供了基础。需要注意的是，此脚本主要针对示例或抽样数据进行聚类分析（控制在 SAMPLE_MAX 条以内），如果要处理全量数据需要留意内存和计算时间开销。通过灵活调整参数如 K_RANGE、LINKAGE（链接方法）等，用户也可以进一步尝试不同聚类设定以比较效果。总之，main() 实现了整个聚类分析的自动化执行和结果导出，是连接数据处理、模型训练和结果产出的核心。

kprototypes_pycharm_run.py 脚本函数详解 | Explanation of Functions in 
kprototypes_pycharm_run.py

此脚本用于执行 K-Prototypes 聚类分析，适合在 PyCharm 等环境中直接运行。它涵盖了从数据预处理到模型训练、结果保存的完整流程。下面对脚本中定义的主要函数进行详细说明，包括其功能、输入输出、参数和在项目流程中的作用。

parse_price(df, col)

•	功能： 将 DataFrame 中指定列 col 的价格区间字符串解析为数值均价。该函数会在原 DataFrame 中新增一列 price_mean_auto，其值为对于每一行 df[col] 的解析结果。解析逻辑与前述 extract_price_mean 类似：如果字符串中包含“–”或“-”表示一个区间，则取划分后的两个数字的平均；如果不包含区间符号则尝试直接转换为浮点数；解析失败则设为 NaN。

•	输入：

o	df: pandas DataFrame，包含原始数据。

o	col: 字符串，要解析的价格区间列名（例如 "指导价区间"）。

•	输出： 返回带有新列的 DataFrame（原 DataFrame 的副本亦被就地修改）。在输出的 DataFrame 中，多出一列 price_mean_auto，其值为所给列每行价格的均值。

•	关键参数： 无其他独立参数。函数内部定义了一个嵌套的辅助函数 get_mean(price_str)，用于对单个字符串进行解析。注意该函数会去除字符串中的“万”字等单位影响，并只处理数值部分。

•	作用： 数据集中如果存在表示价格范围的列，如“指导价区间”，需要将其转换为数值特征以供聚类算法使用。parse_price 封装了这一转换过程，方便重复使用。在本项目中，如果输入数据包含 "指导价区间" 列，preprocess 函数会调用 parse_price 将其转化为均价列。这样，车辆的价格信息就能以连续数值形式加入聚类分析，提高聚类对车型定位的把握。

preprocess(df)

•	功能： 对原始数据 DataFrame 进行全面预处理，生成适合聚类的清洗和变换后数据。具体操作步骤：

1.	数据拷贝： 创建原始 DataFrame 的副本 df_raw 以保留未经缩放的原始数据（特别是“上牌量”等将来可能需要以原始单位输出的列）。后续所有处理将在 df（副本）上进行。
2.	价格区间解析： 如果存在名为“指导价区间”的列，则调用 parse_price(df, "指导价区间")（同时对 df 和 df_raw 都执行），生成平均价格列 price_mean_auto。这一步与前述函数配合，将价格区间转为数值。
3.	缺失值填充： 遍历 DataFrame 的每一列，对于类型为对象（categorical）的列，用该列出现频率最高的值（众数）填充缺失；对于数值列，用该列的中位数填充缺失。这一步保证后续聚类不会因为缺失值而出错，同时尽量减少对数据分布的影响。df_raw 保持与 df 相同的填充，以便两者对应。
4.	特征类型划分： 使用 df.select_dtypes 分离出数值型列列表 numeric_cols 和分类型列列表 categorical_cols。此时 numeric_cols 包括“上牌量”和可能的其他数值列（如均价等），categorical_cols 包括诸如品牌、车型类别、能源类型等字段（数据类型为 object）。
5.	数值归一化： 初始化一个 MinMaxScaler，对所有数值列进行 [0,1] 线性缩放。这一步将“上牌量”等指标缩放到相同范围，避免原始尺度差异过大对聚类距离产生主导影响。
6.	返回结果： 函数返回预处理后的 DataFrame df_prep（已缩放，用于聚类训练）、数值列名列表 numeric_cols、分类列名列表 categorical_cols，以及原始值保留的 DataFrame df_raw（未缩放，用于输出解释）。

•	输入： df：pandas DataFrame，包含原始数据（读取自 Excel）。

•	输出： (df_prep, numeric_cols, categorical_cols, df_raw) 元组：

o	df_prep：经预处理和缩放后的 DataFrame，数值列已归一化，可直接用于聚类。

o	numeric_cols：列表，预处理后确定的数值型特征列名。

o	categorical_cols：列表，预处理后确定的分类型特征列名。

o	df_raw：原始数据的副本，保留了填补缺失后的原始数值（未缩放）的 DataFrame。

•	关键参数： 该函数主要通过硬编码的逻辑处理数据，不需要外部参数。需要注意的是填充缺失值使用的众数和中位数可能会对数据分布略有影响，但这是常见处理。MinMaxScaler 没有特殊参数，此处使用默认设置将每列按列最小/最大值缩放到 [0,1]。

•	作用： preprocess 函数承担着为 K-Prototypes 聚类做好数据准备的任务。通过这一系列操作，原始数据中的混杂问题得到解决：价格区间转为均值、缺失值被填充、特征分门别类、数值特征缩放统一。这确保了后续 K-Prototypes 算法能在一个干净、合理尺度的数据集上运行。在项目中，main 调用 preprocess 后得到的输出 df_prep、numeric_cols、categorical_cols 将直接用于构造聚类输入矩阵。同时返回的 df_raw 保留原始尺度数据，便于在输出结果时使用原始单位（如上牌量原始值）进行说明和呈现。总之，preprocess 对数据进行了全面清洗转换，为模型训练和结果诠释打下基础。


run_kprototypes(df_prep, numeric_cols, categorical_cols)

•	功能： 使用预处理后的数据训练 K-Prototypes 聚类模型，并通过尝试多个簇数来确定最佳簇划分。具体步骤：首先将数据组合成包含数值和类别特征的单一 NumPy 矩阵 X，并生成一个表示哪些列为分类特征的索引列表 cat_idx；然后对预设的每个 K 值，构建 K-Prototypes 模型并拟合数据，记录模型的聚类 成本 (cost) 值；比较不同 K 的成本，选取成本最低的模型作为最佳模型，并返回相关结果。

•	输入：

o	df_prep: pandas DataFrame，预处理且数值已归一化后的数据。

o	numeric_cols: 列表，数值特征列名列表。

o	categorical_cols: 列表，分类特征列名列表。

•	输出： (best_k, best_labels, best_model, cost_records) 元组：

o	best_k: 整数，成本最小的最佳簇数 K。

o	best_labels: NumPy数组，长度为样本数，每个元素是对应样本的簇标签（0 ~ K-1），来自最佳模型。

o	best_model: 训练后的 KPrototypes 模型对象（kmodes.KPrototypes 类型），对应最佳 K。

o	cost_records: 字典，各候选 K 值对应的成本，如 {6: 1234.5, 7: 1100.8, ...}。

•	关键参数： 该函数内部使用了 K_LIST 常量作为候选簇数量列表（例如 [6,7,...,15]）。在训练模型时，每次迭代均使用 init="Huang" 初始化中心、n_init=5 进行5次随机重启动、random_state=42 固定随机种子保证可重复性。cat_idx 列表标记了矩阵 X 中哪些列是分类型，这是 KPrototypes 模型所需的参数，用于应用匹配距离计算。

•	作用： 该函数是 K-Prototypes 聚类的执行核心。一方面，它将 pandas DataFrame 转换为了 K-Prototypes 可直接处理的 NumPy 矩阵形式，并明确分类特征的位置索引；另一方面，它通过遍历多个可能的簇数进行聚类训练，并以模型的 cost 值作为评价标准选优。在本项目中，我们以 cost（簇内距离和）最小作为最佳簇划分的判据，这类似于肘部法但选择了成本最低点。run_kprototypes 返回的最佳模型及标签将用于后续的结果输出和分析。例如，best_labels 会被添加回 DataFrame 以标记每条记录所属簇，cost_records 则用于绘制 K vs. Cost 曲线观察成本随簇数的变化趋势。总之，run_kprototypes 封装了模型训练和参数选择逻辑，使主流程能够方便地获取最优聚类方案。

save_outputs(df, labels, numeric_cols, categorical_cols, cost_records, best_k, model, df_raw)

•	功能： 将聚类结果和相关分析输出到文件，包括：完整数据的簇标签、数值特征簇均值（两种版本）、分类特征 Top-N 分布、K vs Cost 数据曲线、以及元信息 JSON。通过将结果保存到指定输出目录，方便后续对聚类结果的查看和分享。主要输出内容如下：

1.	聚类分配结果： 在 df（缩放数据）和 df_raw（原始数据）中分别追加簇标签列“cluster”。将 df_raw（含cluster）输出为 CSV 文件（如 kprototypes_cluster_assignments_full.csv），此文件包含每条记录及其所属簇标签（以原始值便于阅读）。
2.	数值特征簇均值： 计算并保存两份簇级数值特征概要：
	缩放版均值： 基于 df，按簇计算所有数值特征的平均值，输出 CSV 文件 kprototypes_cluster_numeric_profile_scaled.csv。这些均值在0-1尺度上。
	混合版均值： 在缩放均值的基础上，将其中“上牌量”一列替换为 df_raw 中对应簇的原始均值，以保留“上牌量”的真实数量单位。结果输出 kprototypes_cluster_numeric_profile_mixed.csv。这种混合报告使得除“上牌量”外其他数值仍在0-1范围，但上牌量以实际数量体现，方便解读。
3.	分类特征 Top-10： 对每个分类特征列，计算每个簇中该类别值出现的频数以及占比，并取每簇频数前10的类别组成字典。将每个特征的结果输出为 CSV 文件（命名如 kprototypes_cluster_top_categories__列名.csv）。文件内容包含簇号以及对应的 Top10 类别及频数（及百分比）。
4.	K vs Cost 曲线： 将 cost_records 字典转换为 DataFrame（包含 K 值及对应成本），输出为 CSV 文件 kprototypes_cost_by_k.csv。同时绘制成本随 K 的折线图并保存为图片 kprototypes_cost_curve.png，用于观察聚类成本的变化趋势。
5.	元信息 JSON： 将聚类的一些关键信息（如 best_k, 使用的 numeric_cols 和 categorical_cols, cost_records, 备注等）保存为 JSON 文件 kprototypes_meta.json。特别地，备注中说明了聚类使用的是归一化数据，以及 numeric_profile_mixed 中“上牌量”已还原为原始单位。

•	输入：
o	df: pandas DataFrame，预处理并缩放后的数据（聚类用），函数假定已经包含最新的簇标签列。

o	labels: NumPy数组，聚类标签，长度与 df 行数相同。

o	numeric_cols: 列表，数值特征列名列表。

o	categorical_cols: 列表，分类特征列名列表。

o	cost_records: 字典，各尝试的 K 值及对应聚类成本。

o	best_k: 整数，最优簇数。

o	model: 训练好的 KPrototypes 模型对象（可从中获取聚类中心等信息，如果需要）。

o	df_raw: pandas DataFrame，原始数据填充缺失后的版本。应与 df 有相同的行索引，并未缩放，用于输出原始值。

•	输出： 无显式返回值。函数执行后，会在 OUTPUT_DIR（脚本顶部配置的输出目录，例如 ./out_kproto）下生成多个文件，如上述CSV、PNG、JSON等，涵盖聚类结果和分析。

•	关键参数： 函数内部使用了 os.makedirs 确保输出目录存在。保存输出时文件名固定（按约定构造）。绘图使用 matplotlib，简单绘制了 K 对应成本的曲线并保存。在 Top-10 类别计算中，对每个簇采用
value_counts().head(10) 获得前十类别，并用字典格式表示频数和比例。

•	作用： save_outputs 将模型的结果及评估信息完整地记录下来，是项目产出的最终一步。通过这个函数，聚类分析的成果被保存为多份报告：既有逐个样本的聚类标签明细，也有聚类整体的统计概要，还有模型评估曲线和配置元数据。这些输出文件便于项目参与者和其他读者理解聚类的情况。例如，业务人员可以直接查看 cluster_assignments_full.csv 了解每款车型（或记录）的簇归属，数据分析人员可以根据 cluster_numeric_profile_mixed.csv 和各 Top-10 类别文件了解每个簇的特征特性。成本曲线和 JSON 元信息则有助于记录模型选择的依据和过程。总之，save_outputs 汇总并输出了聚类分析的各方面结果，为报告撰写和后续应用提供了便利。

main
 (K-Prototypes 聚类主流程)
 
•	功能： 脚本的主程序段，将上述各函数串联执行，实现 K-Prototypes 聚类分析的完整流程。主要步骤包括：读取数据、预处理、训练模型（含选定最佳簇数）、保存结果。

•	执行流程：
1.	读取 Excel： 利用 pd.read_excel 加载输入的 Excel 数据文件到 DataFrame df。文件路径和输出目录在脚本开头通过常量设定（如 INPUT_EXCEL 和 OUTPUT_DIR）。
2.	数据预处理： 调用 preprocess(df) 对数据进行清洗和特征准备，获取 df_prep, numeric_cols, categorical_cols, df_raw。此时数据已填补缺失，数值特征已归一化，特征列表也分类明确。
3.	训练 K-Prototypes 模型： 调用 run_kprototypes(df_prep, numeric_cols, categorical_cols) 进行聚类训练和簇数选择，获取 best_k, labels, best_model, cost_records。根据设定的 K 列表自动挑选最佳模型。
4.	导出结果： 调用 save_outputs(df_prep, labels, numeric_cols, categorical_cols, cost_records, best_k, model, df_raw) 保存所有聚类结果文件。
5.	（打印信息）： 在 save_outputs 内部或之后，会打印一行总结，注明最优簇数及输出位置（例如控制台输出 ✅ 最优 K=…，输出已保存等）。

•	作用： 与前一个脚本类似，K-Prototypes 脚本的 main 部分（在 if __name__ == "__main__": 下）负责按照顺序执行各步骤，将数据输入转化为聚类输出。由于所有功能都已封装为函数，main 中逻辑清晰，易于理解和修改。通过阅读 main，可以了解该项目使用 K-Prototypes 进行聚类的具体过程和配置。例如，可以方便地调整 K_LIST 以尝试不同的簇范围，或者修改输入文件路径以应用于新的数据。总之，main 把数据管道连接起来，驱动完成 K-Prototypes 聚类分析并输出成果。用户在使用本脚本时，只需确保配置好路径参数，然后运行脚本即可生成上述所有结果文件。


