import pandas as pd
import numpy as np
from io import BytesIO
import logging

class ExcelDataLoader:
    """
    用于读取二进制Excel文件的数据加载器
    """
    
    def __init__(self, file_path=None, binary_data=None):
        """
        初始化DataLoader
        
        参数:
            file_path: Excel文件路径
            binary_data: 二进制数据（如果从内存读取）
        """
        self.file_path = file_path
        self.binary_data = binary_data
        self.df = None
        self.x_train = None
        self.y_train = None
        self.feature_means = {}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """
        加载Excel数据
        """
        try:
            if self.binary_data is not None:
                # 从二进制数据读取
                self.df = pd.read_excel(BytesIO(self.binary_data), header=0)
                self.logger.info("从二进制数据成功加载Excel文件")
            elif self.file_path is not None:
                # 从文件路径读取
                self.df = pd.read_excel(self.file_path, header=0)
                self.logger.info(f"从文件 {self.file_path} 成功加载Excel文件")
            else:
                raise ValueError("必须提供file_path或binary_data参数")
                
            self.logger.info(f"数据形状: {self.df.shape}")
            self.logger.info(f"列名: {list(self.df.columns)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据时出错: {e}")
            return False
    
    def preprocess_data(self):
        """
        预处理数据:
        - 选择D列到X列作为特征
        - 选择Z列作为标签
        - 删除Z列为NA的行
        - 对特征列的NA值用该列均值填充
        """
        if self.df is None:
            self.logger.error("请先加载数据")
            return False
        
        try:
            # 获取所有列名
            all_columns = list(self.df.columns)
            self.logger.info(f"所有可用列: {all_columns}")
            
            # 找到D列到X列的列名
            # 假设列名是Excel风格的字母（A, B, C, ...）
            # 如果不是，可能需要调整这部分逻辑
            d_to_x_columns = []
            z_column = None
            
            for col in all_columns:
                # 检查列名是否为单个大写字母
                if isinstance(col, str) and col.isalpha() and col.isupper() and len(col) == 1:
                    if 'D' <= col <= 'X':
                        d_to_x_columns.append(col)
                    elif col == 'Z':
                        z_column = col
            
            # 如果没找到字母列名，尝试使用位置索引
            if not d_to_x_columns:
                self.logger.info("未找到字母列名，使用位置索引")
                # D列是第4列（索引3），X列是第24列（索引23）
                d_to_x_columns = all_columns[3:24]  # 注意：Python切片是左闭右开
                z_column = all_columns[25] if len(all_columns) > 25 else None
            
            self.logger.info(f"特征列 (D-X): {d_to_x_columns}")
            self.logger.info(f"标签列 (Z): {z_column}")
            
            if not d_to_x_columns:
                raise ValueError("未找到D到X列的数据")
            
            if z_column is None:
                raise ValueError("未找到Z列")
            
            # 创建数据副本进行处理
            processed_df = self.df.copy()
            
            # 删除Z列为NA的行
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna(subset=[z_column])
            removed_rows = initial_rows - len(processed_df)
            self.logger.info(f"删除了 {removed_rows} 行Z列为NA的数据")
            
            # 提取特征和标签
            x_data = processed_df[d_to_x_columns].copy()
            y_data = processed_df[z_column].copy()
            
            # 计算每列的均值（用于填充NA值）
            self.feature_means = {}
            for col in d_to_x_columns:
                if x_data[col].isna().any():
                    col_mean = x_data[col].mean()
                    self.feature_means[col] = col_mean
                    x_data[col] = x_data[col].fillna(col_mean)
                    self.logger.info(f"列 {col} 的NA值已用均值 {col_mean:.4f} 填充")
            
            # 转换为numpy数组
            self.x_train = x_data.values
            self.y_train = y_data.values
            
            self.logger.info(f"预处理完成 - 特征形状: {self.x_train.shape}, 标签形状: {self.y_train.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据预处理时出错: {e}")
            return False
    
    def get_train_data(self):
        """
        获取训练数据
        
        返回:
            x_train: 特征数据 (numpy array)
            y_train: 标签数据 (numpy array)
        """
        if self.x_train is None or self.y_train is None:
            self.logger.warning("数据未加载或预处理，请先调用load_data()和preprocess_data()")
            return None, None
        
        return self.x_train, self.y_train
    
    def get_dataframe(self):
        """
        获取原始DataFrame
        """
        return self.df
    
    def get_feature_means(self):
        """
        获取用于填充NA值的各列均值
        """
        return self.feature_means


# 使用示例
def main():
    # 方法1: 从文件路径读取
    loader = ExcelDataLoader(file_path="data.xlsx")
    
    # 方法2: 从二进制数据读取（如果已经将文件读入内存）
    # with open("data.xlsx", "rb") as f:
    #     binary_data = f.read()
    # loader = ExcelDataLoader(binary_data=binary_data)
    
    # 加载和预处理数据
    if loader.load_data() and loader.preprocess_data():
        x_train, y_train = loader.get_train_data()
        
        print(f"特征数据形状: {x_train.shape}")
        print(f"标签数据形状: {y_train.shape}")
        print(f"前5行特征数据:\n{x_train[:5]}")
        print(f"前5个标签:\n{y_train[:5]}")
        
        # 获取用于填充的均值信息
        feature_means = loader.get_feature_means()
        if feature_means:
            print("各列填充的均值:")
            for col, mean_val in feature_means.items():
                print(f"  {col}: {mean_val:.4f}")
    else:
        print("数据加载或预处理失败")


if __name__ == "__main__":
    main()