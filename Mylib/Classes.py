class MSCTDDataset(Dataset):
    def __init__(self, img_dir, mode , transform=None, target_transform=None,resize = None):
        # downloading text files from github
        print('\nLoading Text Files')
        os.system('git clone https://github.com/XL2248/MSCTD.git')
        #unziping images
        if mode == 'train':
          print('Loading Train Images')
          os.system('unzip -n '+ os.path.join(img_dir,'train_ende.zip'))
          os.system('mv train_ende train')
          print('Train Images Count:', len(os.listdir('train')))
          os.system('cp -r MSCTD/MSCTD_data/ende/english_train.txt -t train')
          os.system('cp -r MSCTD/MSCTD_data/ende/image_index_train.txt -t train')
          os.system('cp -r MSCTD/MSCTD_data/ende/sentiment_train.txt -t train')
        

        if mode == 'dev':
          print('Loading Validation Images')
          os.system('unzip -n '+ os.path.join(img_dir,'dev.zip'))
          print('Dev Images Count:', len(os.listdir('dev')))
          os.system('cp -r MSCTD/MSCTD_data/ende/english_dev.txt -t dev')
          os.system('cp -r MSCTD/MSCTD_data/ende/image_index_dev.txt -t dev')
          os.system('cp -r MSCTD/MSCTD_data/ende/sentiment_dev.txt -t dev')
        

        if mode == 'test':
          print('Loading Test Images')
          os.system('unzip -n '+ os.path.join(img_dir,'test.zip'))
          print('Test Images Count:', len(os.listdir('test')))
          os.system('cp -r MSCTD/MSCTD_data/ende/english_test.txt -t test')
          os.system('cp -r MSCTD/MSCTD_data/ende/image_index_test.txt -t test')
          os.system('cp -r MSCTD/MSCTD_data/ende/sentiment_test.txt -t test')

        os.system('rm -r MSCTD')
        # processing text files and saving them as attribute of dataset
        if mode == 'val':
            mode = 'dev'
        file1 = open(mode + '/sentiment_' + mode + '.txt', 'r')
        Lines = file1.readlines()
        file1.close()
        label = []
        for line in Lines:
            line = line.strip()
            label.append(int(line))         
        self.sentiment = np.array(label)

        file1 = open(mode + '/english_' + mode + '.txt', 'r')
        Lines = file1.readlines()
        file1.close()
        text = []
        for line in Lines:
            line = line.strip()
            text.append(line)  
        self.text = text

        image_index = []
        file1 = open(mode + '/image_index_' + mode + '.txt', 'r')
        Lines = file1.readlines()
        file1.close()
        text = []
        for line in Lines:
            line = line.strip()
            image_index.append(line) 
        self.image_index = image_index

        self.mode = mode
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        
    def __len__(self):
        return len(self.sentiment)

    def __getitem__(self, idx):
        img_path = os.path.join(self.mode, f'{idx}.jpg')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        sentiment = self.sentiment[idx]
        text = self.text[idx]
        if self.resize:
              image = cv2.resize(image, self.resize) 
        else :
              image = cv2.resize(image, (1280,633)) 

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            sentiment = self.target_transform(sentiment)
        return {'text':text ,'image':image, 'sentiment':(sentiment)}
##########################################################################################################################################
# Defining the Neural Network Layers, Neurons and Activation Function
class CNN1(nn.Module):
    def __init__(self, p = 0):
        self.p = p
        
        super(CNN1, self).__init__()
        self.flatten = nn.Flatten()
        
        self.conv2d_relu_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=self.p),

            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=None, padding=0),
            nn.Dropout(p=self.p),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=None, padding=0),
            nn.Dropout(p=self.p),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='valid'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=self.p),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='valid'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=self.p),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='valid'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=None, padding=0),
            nn.Dropout(p=self.p),
            )
        
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(8, 3),
            )

    def forward(self, x):
        x = self.conv2d_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
##########################################################################################################################################
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
##########################################################################################################################################
# Defining the Neural Network Layers, Neurons and Activation Function
class NNtfid(nn.Module):
    def __init__(self, p, dim):
        self.p = p
        self.dim = dim
        super(NNtfid, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.dim, 64),
                      nn.ReLU(),
                      nn.Dropout(p),
                      nn.Linear(64, 3),
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
##########################################################################################################################################
# Defining the Neural Network Layers, Neurons and Activation Function
class NNglove(nn.Module):
    def __init__(self, dim):
        
        self.dim = dim
        super(NNglove, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
                        nn.Linear(self.dim, 256),
                       nn.BatchNorm1d(256),                      
                      
                      nn.ReLU(),
                      nn.Dropout(0.7),
                      nn.Linear(256, 128),
                      nn.BatchNorm1d(128),                      
                      nn.ReLU(),
                      nn.Dropout(0.6),
                      nn.Linear(128, 64),
                      nn.BatchNorm1d(64),                      
                      nn.ReLU(),
                      nn.Dropout(0.7),
                      nn.Linear(64, 3),
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits   
##########################################################################################################################################
class BertClassifier(nn.Module):
    def __init__(self,bert_model , p = 0.5):
        self.p = p
        self.bert_model = bert_model
        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(self.bert_model)
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(768, 3),
            nn.ReLU(),
            )
        
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        final_layer = self.linear_relu_stack(pooled_output)
        return final_layer
##########################################################################################################################################