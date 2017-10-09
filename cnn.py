"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
1. Added larger bechmark testing
2. Added configuration script for operating through editing a text get_file
3. Added multiple dataset loaders
"""

import numpy as np
import datahelper
#from w2v import train_word2vec
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence


class CNN():
    """
        The cnn text classifier model according to https://arxiv.org/abs/1408.5882 (CNN model for sentence classification
    """

    def __init__(self,embedding_dim=50,sequence_length=400,maxwords=5000,filtersize=(3,8),num_filters=10,dropout_prob=(0.5,0.8),hidden_dim=50,batch_size=10,epochs=1,verbose=1,mtype="CNN-static"):
        self.DATAPATH="./data"
        self.embedding_dim=embedding_dim
        self.filtersize=filtersize
        self.num_filters=num_filters
        self.dropout_prob=dropout_prob
        self.hidden_dim=hidden_dim
        self.batch_size=batch_size
        self.epoch=epochs
        self.silent=verbose
        self.Type=None
        self.sequence_length=sequence_length
        self.max_words=maxwords
        self.model_type=mtype
        self.x_train, self.y_train, self.x_test, self.y_test, self.vocabulary_inv= None,None,None,None
        
    def GenEmbedding(self):
        if self.model_type in ["CNN-non-static", "CNN-static"]:
                embedding_weights = train_word2vec(np.vstack((self.x_train, self.x_test)), self.vocabulary_inv, num_features=self.embedding_dim,
                                                min_word_count=min_word_count, context=context)
                if model_type == "CNN-static":
                    self.x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in self.x_train])
                    self.x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in self.x_test])
                    print("x_train static shape:", self.x_train.shape)
                    print("x_test static shape:", self.x_test.shape)

        elif self.model_type == "CNN-rand":
            embedding_weights = None
        else:
            raise ValueError("Unknown model type")
        
        
    def ConvBlock(self,num_filters,ks,pad="valid",activation="relu",stride=1):
        """
            Adds a specified number of ZeroPadding and Covolution layers
            to the model, and a MaxPooling layer at the very end.

            Args:
                layers (int):   The number of zero padded convolution layers
                                to be added to the model.
                filters (int):  The number of convolution filters to be 
                                created for each layer.
        """
        model = self.model
        ishape=Input(shape=(self.sequence_length,self.embedding_dim))
        model.add(Convolution1D(input_dim=ishape,filters=num_filters,kernel_size=ks,padding=pad,activation=activation,strides=stride))
        
    def FlatBlock(self):
        model=self.model
        model.add(Flatten())
        
    def PoolBlock(self,size):
        model=self.model
        model.add(MaxPooling1D(ppol_size=size))

    def FCBlock(self,size=10,lactivation="relu"):
        """
            Adds a fully connected layer  to the model with a specificied activiation
            Args:   size and activation of dense layers
            Returns:   None
        """
        model = self.model
        model.add(Dense(size, activation=lactivation))
       
    def DropBlock(self,prob=0.5):
        model = self.model
        model.add(Dropout(prob))

    def GraphBlocK(self):
        graph_in = Input(shape=(sequence_length, embedding_dim))
        convs = []
        for fsz in self.filter_sizes:
            conv = Convolution1D(nb_filter=self.num_filters,
                                filter_length=fsz,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1)(graph_in)
            pool = MaxPooling1D(pool_length=2)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)

        if len(filter_sizes)>1:
            out = Merge(mode='concat')(convs)
        else:
            out = convs[0]
        
        gh=Model(input=graph_in,output=out)
        self.model.add(gh)

        

    def create(self):
        """
            Creates the network achitecture
            Args:   None
            Returns:   None
       
        """
                
        model=self.model=Sequential()
        self.model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights))
        self.DropBlock(self.dropout_prob[0])
        self.GraphBlocK()
        self.FCBlock(self.hidden_dim,lactivation='relu')
        self.DropBlock(self.dropout_prob[1])
        self.FCBlock(1,lactivation="sigmoid")
            
              
    def fit(self, x_train,y_train,x_test,y_test,lr=0.001):
        """
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(optimizer="adam",
                loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
          validation_data=(x_test, y_test), verbose=self.silent)

    def predict(self,x_test):
        pass
   
    def report(self,ytrue,ypred):
        pass
    
    def getdata(self,index="imdb"):
        if index=="imdb":
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_words, start_char=None,
                                                              oov_char=None, index_from=None)

            self.x_train = sequence.pad_sequences(x_train, maxlen=self.sequence_length, padding="post", truncating="post")
            self.x_test = sequence.pad_sequences(x_test, maxlen=self.sequence_length, padding="post", truncating="post")

            self.vocabulary = imdb.get_word_index()
            self.vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
            self.vocabulary_inv[0] = "<PAD/>"
        else:
            print("Non databases available for "+ index)
        
        return self.x_train, self.y_train, self.x_test, self.y_test, self.vocabulary_inv
    
def main():
    
    net=CNN()
    x_train,y_train,x_test,y_test,vocabulary_inv=net.getdata()
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
    net.create()
    net.fit(x_train,y_train,x_test,y_test)

main()