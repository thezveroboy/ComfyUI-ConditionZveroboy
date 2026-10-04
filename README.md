# ComfyUI-ConditionZveroboy

## English

Node pack for CONDITIONING operations in ComfyUI: saving to file and loading, element-wise addition and subtraction, and boolean operations over the token sets of multiple conditionings.

Core scenario. There are two images of the same character taken under different conditions: camera angle, lighting and background differ. Each image yields a conditioning. The task is to extract the features common to both conditionings and discard the features specific to each frame. The AND node solves exactly this problem: it selects the tokens of the first conditioning that have close matches in all the others, and forms the resulting conditioning from them. Non-matching tokens are not included in the result. The remaining boolean nodes are built on the same principle of pairwise token comparison, but with a different selection criterion.

The pack is not tied to a specific architecture. All operations run at the level of tensors shaped (batch, seq, dim) and do not rely on peculiarities of individual encoders, so the nodes apply to any DiT models: Krea2, Qwen-Image, Minimax and others. The mandatory requirement: all inputs of one node must come from the same model and the same encoder, i.e. have matching batch and dim. If this requirement is violated, the node fails with an incompatibility error. Sequence lengths may differ; no alignment is required.

All multi-input nodes have an expandable input set. Connecting the last free slot automatically creates the next one (cond_3, cond_4 and so on). There are no per-input weights. The result is determined by token cosine similarity and a single threshold parameter.

The threshold parameter sets the minimum cosine similarity at which two tokens are considered matching. The range is 0.0 to 1.0, the default is 0.7. If the resulting conditioning comes out too short, the threshold should be lowered. If unrelated features leak into the result, it should be raised. The optimal value depends on the encoder and the specific data and is found experimentally.

Internal representation. Each conditioning is treated as a set of token embeddings. For a token from one conditioning, the most similar token in the other conditioning is found, and the decision is made based on the similarity value. Tokens with high mutual similarity are interpreted as shared features (character, style). Tokens without close matches are interpreted as frame-specific features (background, lighting, camera angle). If no token passes the selection, the node returns a single best pair instead of an empty result, since an empty conditioning breaks the sampler.

The AND node returns the intersection. It selects the tokens of the first input that have matches above the threshold in all other inputs. Each selected token is averaged with its nearest neighbors from the other conditionings, which suppresses the noise of individual frames. Example: two photos of a grey cat with violet eyes, one with a sofa and warm light, the other with a street and cold light. The shared features turn out to be the cat, the private ones the background and lighting. The result describes the cat without attachment to the shooting location.

The OR node returns the union. The vectors of all inputs are concatenated along the sequence axis with no filtering or averaging. Example: the first conditioning describes a character, the second clothing, the third a location. The result contains the features of all three sources and is fed to the sampler as a single description. The node is appropriate when the sources complement rather than contradict each other.

The XOR node returns the symmetric difference. It selects tokens that have no matches above the threshold in any of the other inputs; the contributions of all inputs are combined. Example: two photos of the same cat differing in collar color and background. The shared cat features are excluded, the collar and background features remain. The result describes the differences between the frames.

The NOT node is a unary inversion. The sign of all vectors of the single input is flipped. Example: there is a conditioning of an unwanted background. Its inversion, mixed in with a small weight via Add, weakens the influence of that background in the final generation. The inversion has no standalone semantic meaning; the node is used as a utility in composite chains.

The NAND node returns the inversion of the intersection. The AND result is computed first, then the sign is flipped. Example: AND over two cat photos isolates the cat, while NAND over the same inputs yields its complement. Used in composite schemes where negation of the shared part is required.

The NOR node returns the inversion of the union. The OR result is computed first, then the sign is flipped. Example: the union of character, clothing and location is inverted for use as a complex negative condition. Used analogously to NAND.

The XNOR node returns the inversion of the difference. The XOR result is computed first, then the sign is flipped. Example: the differences isolated by XOR are inverted for subsequent subtraction from other branches. This is a utility node, rarely used for direct generative tasks.

The Remove node implements a set difference of the form A NOT B. Only the tokens of the first input that have no matches above the threshold in any of the other inputs are kept. Example: the first input contains a character against a street background, the second only a similar street background. The matching background vectors are excluded, the character vectors are kept. The standard application is removing background, lighting or location by example.

The Add and Subtract nodes perform element-wise arithmetic with no similarity analysis. Add sums the padding-aligned tensors of all inputs. Subtract subtracts all subsequent inputs from the first. Example: adding a character conditioning and a style conditioning yields their direct superposition. Subtracting a background conditioning from a scene conditioning weakens the background. The result contains both shared and private features without separation, so the boolean nodes are preferable for isolating a character.

The Save and Load nodes provide persistence. Save writes the conditioning into the models/conditions directory under the given name in a format compatible with the Condition-Utils pack, and passes the input through unchanged without breaking the chain. Load lists the files in the directory and returns the selected one. Example: the output of a heavy encoder is saved once, and all subsequent experiments load the ready conditioning without re-running encoding.

Practical notes. AND produces a meaningful result when the inputs genuinely share a component: two frames of one character isolate the character, while frames of unrelated objects yield a nearly empty result, which correctly reflects the absence of anything shared. If AND returns too few tokens, lower the threshold or verify that the character is present in all inputs. For Remove, conversely, the first input must contain a mixture and the rest samples of what is being removed. The output length of AND, XOR and Remove is variable and determined by the number of tokens passing the selection at the given threshold.

## Русский

Пакет нод для операций над CONDITIONING в ComfyUI: сохранение в файл и загрузка, поэлементные сложение и вычитание, а также булевы операции над множествами токенов нескольких кондишенов.

Базовый сценарий. Имеются два изображения одного персонажа, снятые в разных условиях: отличаются ракурс, освещение и фон. Из каждого изображения получается кондишен. Задача: выделить признаки, общие для обоих кондишенов, и отбросить признаки, специфичные для каждого кадра. Нода AND решает именно эту задачу: отбирает токены первого кондишена, имеющие близкие соответствия во всех остальных, и формирует из них результирующий кондишен. Несовпадающие токены в результат не включаются. Остальные булевы ноды построены на том же принципе попарного сравнения токенов, но с другим критерием отбора.

Пакет не привязан к конкретной архитектуре. Все операции выполняются на уровне тензоров формы (batch, seq, dim) и не используют особенностей отдельных энкодеров, поэтому ноды применимы к любым DiT-моделям: Krea2, Qwen-Image, Minimax и другим. Обязательное условие: все входы одной ноды должны происходить от одной модели и одного энкодера, то есть иметь совпадающие batch и dim. При нарушении этого условия нода завершается с ошибкой несовместимости. Длины последовательностей при этом могут различаться, выравнивание не требуется.

Все многовходовые ноды имеют расширяемый набор входов. При подключении последнего свободного слота автоматически создаётся следующий (cond_3, cond_4 и далее). Индивидуальных весов на входах нет. Результат определяется косинусной схожестью токенов и единственным параметром threshold.

Параметр threshold задаёт минимальную косинусную схожесть, при которой два токена считаются соответствующими друг другу. Диапазон от 0.0 до 1.0, значение по умолчанию 0.7. Если результирующий кондишен получается слишком коротким, порог следует понизить. Если в результат проходят посторонние признаки, порог следует повысить. Оптимальное значение зависит от энкодера и конкретных данных и подбирается экспериментально.

Внутреннее представление. Каждый кондишен рассматривается как множество токенов-эмбеддингов. Для токена из одного кондишена находится наиболее схожий токен в другом кондишене, и по величине схожести принимается решение. Токены с высокой взаимной схожестью интерпретируются как общие признаки (персонаж, стиль). Токены без близких соответствий интерпретируются как частные признаки кадра (фон, освещение, ракурс). Если ни один токен не проходит отбор, нода возвращает одну наилучшую пару вместо пустого результата, поскольку пустой кондишен нарушает работу сэмплера.

Нода AND возвращает пересечение. Отбираются токены первого входа, имеющие соответствия выше порога во всех остальных входах. Каждый отобранный токен усредняется со своими ближайшими соседями из остальных кондишенов, что подавляет шум отдельных кадров. Пример: два фото серой кошки с фиолетовыми глазами, на одном диван и тёплый свет, на другом улица и холодный свет. Общими оказываются признаки кошки, частными — фон и освещение. Результат описывает кошку без привязки к месту съёмки.

Нода OR возвращает объединение. Векторы всех входов конкатенируются вдоль оси последовательности без фильтрации и усреднения. Пример: первый кондишен описывает персонажа, второй — одежду, третий — локацию. Результат содержит признаки всех трёх источников и подаётся в сэмплер как единое описание. Нода уместна, когда источники дополняют, а не противоречат друг другу.

Нода XOR возвращает симметрическую разность. Отбираются токены, не имеющие соответствий выше порога ни в одном из остальных входов, вклады всех входов объединяются. Пример: два фото одной кошки, отличающиеся цветом ошейника и фоном. Общие признаки кошки исключаются, остаются признаки ошейников и фона. Результат описывает различия между кадрами.

Нода NOT — унарная инверсия. Знак всех векторов единственного входа меняется на противоположный. Пример: имеется кондишен нежелательного фона. Его инверсия, подмешанная с малым весом через Add, ослабляет влияние этого фона в итоговой генерации. Самостоятельного семантического смысла инверсия не имеет, нода используется как служебная в составных цепочках.

Нода NAND возвращает инверсию пересечения. Вычисляется результат AND, затем меняется знак. Пример: AND двух фото кошки выделяет кошку, NAND тех же входов даёт её дополнение. Применяется в составных схемах, где требуется отрицание общего.

Нода NOR возвращает инверсию объединения. Вычисляется результат OR, затем меняется знак. Пример: объединение персонажа, одежды и локации инвертируется для использования в качестве сложного негативного условия. Применяется аналогично NAND.

Нода XNOR возвращает инверсию разности. Вычисляется результат XOR, затем меняется знак. Пример: различия, выделенные XOR, инвертируются для последующего вычитания из других веток. Нода служебная, для прямых генеративных задач используется редко.

Нода Remove реализует разность множеств вида A NOT B. Из первого входа сохраняются только токены, не имеющие соответствий выше порога ни в одном из остальных входов. Пример: первый вход содержит персонажа на фоне улицы, второй — только схожий уличный фон. Совпавшие векторы фона исключаются, векторы персонажа сохраняются. Стандартное применение — удаление фона, освещения или локации по образцу.

Ноды Add и Subtract выполняют поэлементную арифметику без анализа схожести. Add суммирует выровненные паддингом тензоры всех входов. Subtract вычитает все последующие входы из первого. Пример: сложение кондишена персонажа и кондишена стиля даёт их прямую суперпозицию. Вычитание кондишена фона из кондишена сцены ослабляет фон. Результат содержит как общие, так и частные признаки без разделения, поэтому для выделения персонажа предпочтительны булевы ноды.

Ноды Save и Load обеспечивают персистентность. Save записывает кондишен в каталог models/conditions под заданным именем в формате, совместимом с пакетом Condition-Utils, и пропускает вход дальше без изменений, не разрывая цепочку. Load перечисляет файлы каталога и возвращает выбранный. Пример: результат тяжёлого энкодера сохраняется один раз, все последующие эксперименты загружают готовый кондишен и не требуют повторного кодирования.

Практические замечания. AND даёт осмысленный результат, когда входы действительно имеют общую составляющую: два кадра одного персонажа выделяют персонажа, а кадры несвязанных объектов дают почти пустой результат, что корректно отражает отсутствие общего. Если AND возвращает слишком мало токенов, следует понизить порог либо убедиться, что персонаж присутствует на всех входах. Для Remove, наоборот, требуется, чтобы первый вход содержал смесь, а остальные — образцы удаляемого. Длина результата AND, XOR и Remove переменна и определяется числом токенов, прошедших отбор при заданном пороге.

## 中文

用于 ComfyUI 中 CONDITIONING 操作的节点包：保存到文件与加载、逐元素加法与减法，以及对多个 conditioning 的 token 集合进行布尔运算。

典型场景。同一角色有两张在不同条件下拍摄的图像：机位、光照和背景各不相同。每张图像生成一个 conditioning。任务是提取两个 conditioning 共有的特征，并丢弃每一帧特有的特征。AND 节点正是解决这个问题的：它选出第一个 conditioning 中在其余所有 conditioning 中都有相近对应 token，并以此构成结果 conditioning。不匹配的 token 不会进入结果。其余布尔节点基于同样的 token 两两比较原理，只是筛选标准不同。

本包不绑定具体架构。所有运算都在形状为 (batch, seq, dim) 的张量层面执行，不依赖任何编码器的特性，因此这些节点适用于任何 DiT 模型：Krea2、Qwen-Image、Minimax 等。强制要求：同一个节点的所有输入必须来自同一模型和同一编码器，即 batch 和 dim 必须一致。违反该条件时节点将以不兼容错误终止。序列长度可以不同，无需对齐。

所有多输入节点都具有可扩展的输入集。连接最后一个空闲槽位后会自动创建下一个（cond_3、cond_4，以此类推）。各输入没有独立权重，结果由 token 余弦相似度与唯一的 threshold 参数决定。

threshold 参数定义两个 token 被视为相互对应的最小余弦相似度。范围为 0.0 到 1.0，默认值为 0.7。如果结果 conditioning 过短，应降低阈值；如果无关特征混入结果，应提高阈值。最优值取决于编码器和具体数据，需通过实验确定。

内部表示。每个 conditioning 被视为一组 token embedding。对于来自一个 conditioning 的 token，在另一个 conditioning 中找到最相似的 token，并根据相似度大小作出判断。互相似度高的 token 被解释为共有特征（角色、风格），没有相近对应的 token 被解释为该帧私有特征（背景、光照、机位）。如果没有任何 token 通过筛选，节点将返回一个最佳匹配对而非空结果，因为空 conditioning 会破坏采样器的工作。

AND 节点返回交集。选出第一个输入中在其余所有输入中都有高于阈值对应的 token。每个选出的 token 与其在其他 conditioning 中的最近邻取平均，从而抑制单帧噪声。例如：两张灰猫紫眼睛的照片，一张是沙发与暖光，另一张是街道与冷光。共有的是猫的特征，私有的是背景与光照，结果描述的是猫本身，而不依附于拍摄地点。

OR 节点返回并集。所有输入的向量沿序列轴拼接，不做过滤也不取平均。例如：第一个 conditioning 描述角色，第二个描述服装，第三个描述场景，结果同时包含三个来源的特征，作为统一描述送入采样器。当各来源相互补充而非相互矛盾时适合使用该节点。

XOR 节点返回对称差。选出在其他任何输入中都没有高于阈值对应的 token，并合并所有输入的贡献。例如：同一只猫的两张照片，项圈颜色和背景不同。共有的猫特征被排除，留下项圈与背景特征，结果描述的是两帧之间的差异。

NOT 节点是一元取反。将其唯一输入的所有向量变号。例如：存在一个不需要的背景的 conditioning，将其取反后通过 Add 以小权重混入，可以削弱该背景在最终生成中的影响。取反本身没有独立的语义含义，该节点用作复合链路中的辅助工具。

NAND 节点返回交集的取反。先计算 AND 的结果，再变号。例如：两张猫照片的 AND 提取出猫，相同输入的 NAND 给出其补集。用于需要对共有部分取反的复合方案。

NOR 节点返回并集的取反。先计算 OR 的结果，再变号。例如：将角色、服装与场景的并集取反，用作复杂的负面条件。用法与 NAND 类似。

XNOR 节点返回差集的取反。先计算 XOR 的结果，再变号。例如：XOR 分离出的差异被取反，用于后续从其他分支中减去。这是辅助节点，直接生成任务中很少使用。

Remove 节点实现 A NOT B 形式的集合差。仅保留第一个输入中在其他任何输入中都没有高于阈值对应的 token。例如：第一个输入是街道背景中的角色，第二个是相似的纯街道背景，匹配的背景向量被排除，角色向量被保留。标准用途是按样本去除背景、光照或场景。

Add 和 Subtract 节点执行不分析相似度的逐元素算术。Add 将所有输入经 padding 对齐后的张量相加，Subtract 从第一个输入中减去其余所有输入。例如：角色 conditioning 与风格 conditioning 相加得到二者的直接叠加，从场景 conditioning 中减去背景 conditioning 可以削弱背景。结果同时包含共有与私有特征而不做区分，因此要分离角色应优先使用布尔节点。

Save 和 Load 节点提供持久化。Save 以与 Condition-Utils 包兼容的格式将 conditioning 写入 models/conditions 目录中指定名称的文件，并将输入原样透传，不中断链路。Load 列出目录中的文件并返回所选文件。例如：重型编码器的输出只需保存一次，后续所有实验直接加载现成的 conditioning，无需重复编码。

实践说明。当输入确实具有共有成分时 AND 才给出有意义的结果：同一角色的两帧分离出角色，而无关物体的帧给出近乎空的结果，这正确反映了没有共有内容。如果 AND 返回的 token 过少，应降低阈值，或确认角色是否出现在所有输入中。Remove 则相反，要求第一个输入是混合体，其余输入是待去除内容的样本。AND、XOR 和 Remove 的结果长度是可变的，取决于在给定阈值下通过筛选的 token 数量。
