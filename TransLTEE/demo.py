class encoder(nn.Module):
    def __init__
    
    
    
class decoder


def MLP(nn.Module):
    def __init__(input_dim):
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.linear2 = nn.Linear(input_dim//2, input_dim//4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out1 = self.linear1(x)
        out1 = self.relu(out1)
        out = self.sigmoid(self.linear2(out1))
        
class model(nn.Module):
    def __init__():
        self.E1 = MLP()
        self.E2 = MLP()
        self.D1 = decoder()
        self.D2 = decoder()
        self.loss = F.BCE()
        self.embedding_dim = 25

    def fit(self, x0, y0, x1, y1, batch_size = 100, max_epoch = 10, lr = 0.01, lamb = 1e-4):
        input_dim = x0.shape[1]
        number_sample0 = x0.shape[0]
        number_sample1 = x1.shape[0]
        
        total_batch = number_sample0 // 100

        idx0 = np.arange(number_sample0)
        idx1 = np.arange(number_sample1)
        
        opt1 = torch.opt.Adam(self.E1.parameters(), lr = lr, weight_decay = lamb)
        opt0 = torch.opt.Adam(self.E0.parameters(), lr = lr, weight_decay = lamb)
        
        for epoch in range(max_epoch):
            np.random.shuffle(idx0)
            np.random.shuffle(idx1)
            
            encode_0 = self.E1.forward(x0)
            encode_1 = self.E2.forward(x1)
            
            #IPM
            
            for i in range(total_batch):
                select_idx0 = idx0[i*batch_size, (i+1)*batch_size]
                select_idx1 = idx1[i*batch_size * x1.shape[0]/x0.shape[0], (i+1)*batch_size * x1.shape[0]/x0.shape[0]]
                
                sub_x0 = x0[select_idx0]
                sub_x1 = x1[select_idx1]
                
                sub_y0 = y0[select_idx0]
                sub_y1 = y1[select_idx1]
                
                encode_0 = self.E1.forward(sub_x0)
                encode_1 = self.E2.forward(sub_x1)
                
                y0_pred_sequence = self.D1.forward(encode_0)
                y1_pred_sequence = self.D2.forward(encode_1)
                
                loss0 = torch.sum((sub_y0 - y0_pred_sequence) ** 2)
                loss1 = torch.sum((sub_y1 - y1_pred_sequence) ** 2)
                
                total_loss = loss0 + loss1
                
                opt1.zero_grad()
                
        