import pickle
import torch.nn as nn
import os


class orm(nn.Module):
    def __init__(self, data_file="sql/sql.pkl"):
        super().__init__()
        self.data_file = data_file

    def save(self, sql):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'ab+') as f:
                pickle.dump(sql, f)
                f.write(b'\n')
            # 如果文件不存在，则创建新文件并写入数据
        else:
            with open(self.data_file, 'wb+') as f:
                pickle.dump(sql, f)
                f.write(b'\n')
