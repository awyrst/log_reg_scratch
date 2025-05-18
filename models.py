import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math



# preprocess class takes in a dataset after having been read by pandas.
# it then has some built-in methods to modify the data and stuff, before
# outputting it as a numpy multidimensional array to be inputted into 
# an instance of model.


#note that the "randomize" flag uses the same seed every time. This is mainly for cross-validation,
#so that when we're comparing models, they're training and testing on the same folds
class preprocess_visualize():
    #constructor
    def __init__(self, data, kind, randomize):
        #convert to numpy array
        dataset = (data.to_numpy()).copy()
        #remove the first column
        dataset = (dataset)[:, 1:]
    
        self.kind_of_data = kind

        if randomize == "randomize":
            np.random.seed(42)
            np.random.shuffle(dataset)
        
        self.data = dataset
        
        

    #randomize the data
    def shuffle(self):
        np.random.shuffle(self.data)

        
#plot the feature distribution - input the column number and plot the histogram 
    def plot_feature_distribution(self, col_index):
        
        column_data = (self.data)[:, col_index]

        #changes the backend or something
        matplotlib.use("TkAgg")

        #if the column index is referencing the labels, the plot will have the appropriate title
        if (col_index == self.data.shape[1]-1):
           
            #two classes, so two bins. Apparently the hist function has three outputs
            n, bins, patches = plt.hist(column_data, bins=2, rwidth=0.8, alpha=0.9, edgecolor="black")

            #add the frequency next to each bin
            for i in range(len(patches)):
                plt.text(bins[i] + 0.1, n[i], f'{int(n[i])}', color='black', fontsize=8)

            # Assign colors to each bin
            colours = ["purple", "orange"]

            for patch, color in zip(patches, colours):  
                patch.set_facecolor(color)

            plt.title("Histogram of labels")
        
        else:

            count, bins, n = plt.hist(column_data, bins=20, color="skyblue", alpha=0.9, edgecolor="black")
            index = str(col_index+1)
            
            #compute mean and std_dev
            mean = np.mean(column_data)
            std_dev = np.std(column_data)

            #add some addidional labels
            plt.axvline(mean, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2f}" )
            plt.axvline(mean - std_dev, color="black", linestyle="dashed", linewidth=2, label=f"\u00B1 Std Dev = {std_dev: .2f}")
            plt.axvline(mean + std_dev, color="black", linestyle="dashed", linewidth=2)

            x = np.linspace(min(self.data[:,col_index]), max(self.data[:,col_index]), 1000)
            bin_width = bins[1] - bins[0]
            gaussian = (count.sum()*bin_width)*(1/(std_dev*np.sqrt(2* np.pi)))*np.exp(-0.5*((x-mean)/std_dev)**2)
            plt.plot(x, gaussian, 'r-', linewidth=2)
            #add legend
            plt.legend()
            plt.title("Histogram of feature "+index)

        plt.xlabel("Feature Value")
        plt.ylabel("Number of Occurrences")
        #plt.grid(True)
        plt.show()
        

#define the sigmoid
def sigmoid_scalar(x):
    return (1/(1+np.exp(-x)))
    

class model():

    # constructor, takes in the learning rate, a parameter that specifies how the
    # learning rate changes over time, called dynamic learn, and an instance of preprocess_visualize
    def __init__(self, learn_rate, num_steps, dynamic_learn, preprocessed):

        #local variable for the dataset
        dataset = preprocessed.data.copy()
      
        #give the model a learning rate
        self.rate = learn_rate
      
        # give it the labels
        labels = dataset[:, dataset.shape[1]-1]
        # must initialize new array to store the values
        labels_asValues = np.empty(dataset.shape[0], dtype=float)
        #must convert the labels into ones or zeroes
        for i in range(len(labels)):
            if labels[i] == 'Normal':
                labels_asValues[i] = 0.
            else:
                labels_asValues[i] = 1.
        self.y = labels_asValues

        # give it the 'X' matrix
        # first add the column of dummy features
        features = dataset
        
        features[:, features.shape[1]-1] = np.ones(features.shape[0])
        self.X = features.astype(np.float64)
                
        # one parameter for every feature, plus an extra for the dummy feature
        self.W = np.zeros(self.X.shape[1])

        self.dynamic_learn = dynamic_learn
        self.num_steps = num_steps

# normalize the data
    def normalize(self):
        
        for i in range(self.X.shape[1]):
            mean = np.mean(self.X[:,i])
            std_dev = np.std(self.X[:,i])
            for j in range(self.X.shape[0]):
                if std_dev != 0:
                    self.X[j,i] = (self.X[j,i]-mean)/std_dev

# add another feature
    def quadratic_feature(self, col_num):
        X = np.zeros((self.X.shape[0], self.X.shape[1]+1))
        X[:, :-2] = self.X[:, :-1]
        X[:, -2] = np.power((self.X[:, col_num]),2)
        X[:, -1] = np.ones(self.X.shape[0])
        self.X = X
        self.W = np.zeros(self.X.shape[1])

    def cubic_feature(self, col_num):
        X = np.zeros((self.X.shape[0], self.X.shape[1]+1))
        X[:, :-2] = self.X[:, :-1]
        X[:, -2] = np.power((self.X[:, col_num]),3)
        X[:, -1] = np.ones(self.X.shape[0])
        self.X = X
        self.W = np.zeros(self.X.shape[1])
    
    def remove(self, cols):
        self.X = np.delete(self.X,cols,1)
        self.W = np.zeros(self.X.shape[1])
    
    # the "fit" method: takes in the data, labels and learning rate
    # doesn't return anything, but updates the parameters 'W' according
    # to the gradient descent. Also returns an array of the accuracies
    # on training data at every step of gradient descent.

def fit(m0del):

    arr = np.zeros(500)    
        #re-initialize weights to 0
    m0del.W = np.zeros(m0del.X.shape[1])
        
        

    #initialize a counter
    k = 0

    #keep track of initial learning rate
    lrn_rate = m0del.rate
    for k in range(m0del.num_steps):
            #initialize the gradient
        grad_W = np.zeros(len(m0del.W))
        for j in range(len(m0del.W)):

                #compute the partial derivative of the loss w.r.t. Wj
            for i in range(m0del.X.shape[0]):
                h = np.dot(m0del.W, m0del.X[i,:])
                grad_W[j] += (m0del.y[i]-sigmoid_scalar(h))*m0del.X[i,j]

            # the 1/k+1 learning rate scheme

        if m0del.dynamic_learn == "k+1":
            m0del.rate = lrn_rate/(k+1)

            # the power learning rate scheme. Didn't use this in our report though.
        elif m0del.dynamic_learn == "power":
            if k > 1 and k <= 10:
                m0del.rate = lrn_rate
            elif k > 10 and k <= 20:
                m0del.rate = math.pow(lrn_rate, 2)
            elif k > 20 and k <= 30:
                m0del.rate = math.pow(lrn_rate, 3)
            elif k > 30 and k <= 40:
                m0del.rate = math.pow(lrn_rate, 4)
            elif k > 40 and k <= 50:
                m0del.rate = math.pow(lrn_rate, 5)
            elif k > 50 and k <= 60:
                m0del.rate = math.pow(lrn_rate, 6)
            elif k > 60 and k<= 70:
                m0del.rate = math.pow(lrn_rate, 7)
                    
            elif k > 70 and k <= 80:
                m0del.rate = math.pow(lrn_rate, 8)
            elif k > 80 and k <= 90:
                m0del.rate = math.pow(lrn_rate, 9)
            elif k > 90 and k <= 100:
                m0del.rate = math.pow(lrn_rate, 10)
            else:
                m0del.rate = math.pow(lrn_rate, 11)

        print("Step size: "+str(np.linalg.norm(m0del.rate*grad_W)))
        print("norm of weights: "+str(np.linalg.norm(m0del.W)))
        acc_on_training_data = Accu_eval(m0del.y, predict(m0del))
        arr[k] = acc_on_training_data
        print("accuracy on training data: "+str(acc_on_training_data)+"%")
        print("Percent difference: "+str(100*np.linalg.norm(m0del.rate*grad_W)/np.linalg.norm(m0del.W))+"%")
        print("learning rate: "+str(m0del.rate))

            #if the gradient becomes too small, break
        if np.linalg.norm(100*np.linalg.norm(m0del.rate*grad_W)/np.linalg.norm(m0del.W)) < 0.1 :
            return arr

            #update the jth parameter
        m0del.W += m0del.rate*grad_W

        k += 1
        print('\n'+str(k))

    return arr

#this version of fit is to be run from inside the crossvalidation function only
def fit_crossval(m0del):

    m0del.W = np.zeros(m0del.X.shape[1])
        

    #initialize a counter
    k = 0

    #keep track of initial learning rate
    lrn_rate = m0del.rate
    for k in range(m0del.num_steps):
            #initialize the gradient
        grad_W = np.zeros(len(m0del.W))
        for j in range(len(m0del.W)):

                #compute the partial derivative of the loss w.r.t. Wj
            for i in range(m0del.X.shape[0]):
                h = np.dot(m0del.W, m0del.X[i,:])
                grad_W[j] += (m0del.y[i]-sigmoid_scalar(h))*m0del.X[i,j]

            # the 1/k+1 learning rate scheme

        if m0del.dynamic_learn == "k+1":
            m0del.rate = lrn_rate/(k+1)

            # the power learning rate scheme
        elif m0del.dynamic_learn == "power":
            if k > 1 and k <= 10:
                m0del.rate = lrn_rate
            elif k > 10 and k <= 20:
                m0del.rate = math.pow(lrn_rate, 2)
            elif k > 20 and k <= 30:
                m0del.rate = math.pow(lrn_rate, 3)
            elif k > 30 and k <= 40:
                m0del.rate = math.pow(lrn_rate, 4)
            elif k > 40 and k <= 50:
                m0del.rate = math.pow(lrn_rate, 5)
            elif k > 50 and k <= 60:
                m0del.rate = math.pow(lrn_rate, 6)
            elif k > 60 and k<= 70:
                m0del.rate = math.pow(lrn_rate, 7)
                    
            elif k > 70 and k <= 80:
                m0del.rate = math.pow(lrn_rate, 8)
            elif k > 80 and k <= 90:
                m0del.rate = math.pow(lrn_rate, 9)
            elif k > 90 and k <= 100:
                m0del.rate = math.pow(lrn_rate, 10)
            else:
                m0del.rate = math.pow(lrn_rate, 11)

        print("Step size: "+str(np.linalg.norm(m0del.rate*grad_W)))
        print("norm of weights: "+str(np.linalg.norm(m0del.W)))
        acc_on_training_data = Accu_eval(m0del.y, predict(m0del))
        print("accuracy on training data: "+str(acc_on_training_data)+"%")
        print("Percent difference: "+str(100*np.linalg.norm(m0del.rate*grad_W)/np.linalg.norm(m0del.W))+"%")
        print("learning rate: "+str(m0del.rate))

            #if the gradient becomes too small, break
        if np.linalg.norm(100*np.linalg.norm(m0del.rate*grad_W)/np.linalg.norm(m0del.W)) < 0.1 :
            return m0del

            #update the jth parameter
        m0del.W += m0del.rate*grad_W

        k += 1
        print('\n'+str(k))

    return m0del
            

    # takes as input a set of test data, and outputs the predicted labels
def predict(m0del):
        #create vector of arguments to the sigmoid
    y_pred1 = (np.matmul(m0del.X, m0del.W))
       
        #apply the sigmoid to all elements of previous vector
    y_pred = np.zeros(len(y_pred1))

    for i in range(len(y_pred1)):
        y_pred[i] = sigmoid_scalar(y_pred1[i]) 

        #round up or down from 0.5 - indicates that the decision boundary is at 0.5
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5:
            y_pred[i] = 0.
        else:
            y_pred[i] = 1.
    return y_pred

# takes the input and output labels and outputs an accuray score
def Accu_eval(y, pred_labels):
    
    n = len(y)
    m = 0
    for i in range(len(y)):
        if pred_labels[i] == y[i]:
            m += 1

    return 100*m/n

 

#crossvalidation function
def crossvalidate_10(models):
    #array to hold accuracies across models
    arr1 = np.zeros(len(models))
        #extract the number of rows
    rows = models[0].X.shape[0]
    
    
        #extract the size of each fold
    fold_size = rows//10
    
    #iterate on every model in the array of models
    for i in range (len(models)):
        learning_rate = models[i].rate
        #array to hold accuracy level of each fold
        arr = np.zeros(10)
            #for every model, run the train and test ten times
        for j in range(10):
                #every iteration, we save a copy of the features and labels
            features = (models[i].X).copy()
            labels = (models[i].y).copy()
                #we also save a copy of the X_test and Y_test
            X_test = (models[i].X)[j*fold_size:(j+1)*fold_size,:].copy()
            y_test = (models[i].y)[j*fold_size:(j+1)*fold_size].copy()
                #remove X_test and y_test from the original data

            X_train = np.delete(features, np.s_[j * fold_size:(j + 1) * fold_size], axis=0)
            y_train = np.delete(labels, np.s_[j * fold_size:(j + 1) * fold_size])

            models[i].X = X_train
            models[i].y = y_train

                #train the model on the 9/10 folds
            models[i] = fit_crossval(models[i])
                #reassign the model's X and y values to the test data
            models[i].X = X_test
            models[i].y = y_test
                #run accu-eval on that fold
            arr[j] = Accu_eval(models[i].y, predict(models[i]))
            print("model "+str(i+1)+" accuracy on fold "+str(j+1)+" is: "+str(arr[j])+"%")
                #change data back to original copies
            models[i].X = features
            models[i].y = labels
            models[i].rate = learning_rate
        arr1[i] = np.mean(arr)
        print("cross-validation accuracy on model "+str(i+1)+" is "+str(arr1[i])+"%")
    print(arr1)
    return arr1
        


