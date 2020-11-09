import numpy as np
import torch.utils.data
import pandas as pd

class DHH2020_get_dataset():
      def getDB(file_name):
            file = pd.read_csv(file_name)
            x = file.iloc[0:667, 1:19].values
            y = file.iloc[0:667, 19:20].values

            # remove variable: 0, 5, 16 (GW)
            # high correlation variables: 5, 7, 11, 12, 14, 15, 16 (SY)
            #x = np.delete(x, 16, 1)
            #x = np.delete(x, 5, 1)
            #x = np.delete(x, 0, 1)
            
            # x = np.delete(x, 13, 1)
            # x = np.delete(x, 10, 1)
            # x = np.delete(x, 9, 1)
            # x = np.delete(x, 8, 1)
            # x = np.delete(x, 6, 1)
            # x = np.delete(x, 4, 1)
            # x = np.delete(x, 3, 1)
            # x = np.delete(x, 2, 1)
            # x = np.delete(x, 1, 1)
            # x = np.delete(x, 0, 1)

            #np.random.shuffle(x)
            #np.random.shuffle(y)

            x_train_0 = x[(x[:, 17:18] == 0).squeeze(), :17] #[:566, :]
            x_train_1 = x[(x[:, 17:18] == 1).squeeze(), :17]
            y_train_0 = y[(x[:, 17:18] == 0).squeeze(), :17] #[:566, :]
            y_train_1 = y[(x[:, 17:18] == 1).squeeze(), :17]

            x_valid_0 = x_train_0[299:, :17] # 100
            x_valid_1 = x_train_1[167:, :17] # 100
            y_valid_0 = y_train_0[299:, :17] # 100
            y_valid_1 = y_train_1[167:, :17] # 100

            return x_train_0, y_train_0, x_valid_0, y_valid_0, x_train_1, y_train_1, x_valid_1, y_valid_1

class DHH2020_train(torch.utils.data.Dataset):
      def __init__(self, x, y):
            self.x_train = torch.tensor(x, dtype=torch.float32)#.unsqueeze(1)
            self.y_train = torch.tensor(y, dtype=torch.long)
            
      def __len__(self):
            return len(self.y_train)

      def __getitem__(self, idx):
            return self.x_train[idx], self.y_train[idx]

class DHH2020_valid(torch.utils.data.Dataset):
      def __init__(self, x, y):
            self.x_valid = torch.tensor(x, dtype=torch.float32)#.unsqueeze(1)
            self.y_valid = torch.tensor(y, dtype=torch.long)
      
      def __len__(self):
            return len(self.y_valid)

      def __getitem__(self, idx):
            return self.x_valid[idx], self.y_valid[idx]

class DHH2020_test(torch.utils.data.Dataset):
      def __init__(self, file_name):
            file = pd.read_csv(file_name)
            x = file.iloc[0:287, 1:18].values

            #x0 = np.delete(x, 18, 1)
            #x1 = np.delete(x, 17, 1)

            # high correlation variables: 5, 7, 11, 12, 14, 15, 16
            #x0 = np.delete(x0, 16, 1)
            #x0 = np.delete(x0, 5, 1)
            #x0 = np.delete(x0, 0, 1)
            # x0 = np.delete(x0, 13, 1)
            # x0 = np.delete(x0, 10, 1)
            # x0 = np.delete(x0, 9, 1)
            # x0 = np.delete(x0, 8, 1)
            # x0 = np.delete(x0, 6, 1)
            # x0 = np.delete(x0, 4, 1)
            # x0 = np.delete(x0, 3, 1)
            # x0 = np.delete(x0, 2, 1)
            # x0 = np.delete(x0, 1, 1)
            # x0 = np.delete(x0, 0, 1)

            # high correlation variables: 5, 7, 11, 12, 14, 15, 16
            #x1 = np.delete(x1, 16, 1)
            #x1 = np.delete(x1, 5, 1)
            #x1 = np.delete(x1, 0, 1)
            # x1 = np.delete(x1, 13, 1)
            # x1 = np.delete(x1, 10, 1)
            # x1 = np.delete(x1, 9, 1)
            # x1 = np.delete(x1, 8, 1)
            # x1 = np.delete(x1, 6, 1)
            # x1 = np.delete(x1, 4, 1)
            # x1 = np.delete(x1, 3, 1)
            # x1 = np.delete(x1, 2, 1)
            # x1 = np.delete(x1, 1, 1)
            # x1 = np.delete(x1, 0, 1)

            #self.x0_test = torch.tensor(x0, dtype=torch.float32)#.unsqueeze(1)
            #self.x1_test = torch.tensor(x1, dtype=torch.float32)#.unsqueeze(1)
            self.x = torch.tensor(x, dtype=torch.float32)#.unsqueeze(1)
      
      def __len__(self):
            return len(self.x)

      def __getitem__(self, idx):
            return self.x[idx]