# PCOS-sequential
# 临床决策支持预测工具

本工具基于提供的临床特征数据，输出相关的风险概率预测。

## 环境配置

请首先安装必要的Python依赖库。

```bash
pip install -r requirements.txt
```

## 使用方法

请使用 `predict.py` 脚本进行预测。您需要准备一个Pandas DataFrame作为输入，并确保其包含所有必需的特征列。

可以参考 `predict.py` 文件底部的示例来了解如何调用 `make_predictions` 函数。

```python
# 示例代码
from predict import make_predictions
import pandas as pd

# ... 创建一个包含所需特征的DataFrame ...
# new_patient_data = pd.DataFrame(...) 

# 获取预测结果
# predictions = make_predictions(new_patient_data)
# print(predictions)
```

**免责声明**: 本工具提供的预测结果仅供研究参考，不应作为临床决策的唯一依据。 
