import pandas as pd
import numpy as np
import models as mds

# Use pandas to store data in a dataframe
df_data1 = pd.read_csv('./CKD.csv')
df_data2 = pd.read_csv('./Battery_Dataset.csv')


#make the instances of preprocess_visualize with appropriate datasets
banana1 = mds.preprocess_visualize(df_data1, "CKD", "randomize")
banana2 = mds.preprocess_visualize(df_data2, "battery", "randomize")

#this is the array to store the data
data_csvfile = np.empty((11,2))

#step one is to generate the models with the removed features. Do so as in the rem_30_80_percent_mean.py file.
# refer to that file for a few more comments on what's happening in this process
model1_1 = mds.model(1,500,"k+1",banana1)
model1_2 = mds.model(1,500,"k+1",banana1)
model1_3 = mds.model(1, 500, "k+1", banana1)

model1_1.normalize()
model1_2.normalize()
model1_3.normalize()

model2_1 = mds.model(1,500,"k+1",banana2)
model2_2 = mds.model(1,500,"k+1",banana2)
model2_3 = mds.model(1, 500, "k+1", banana2)

model2_1.normalize()
model2_2.normalize()
model2_3.normalize()

                    #remove features at 30% threshold
mds.fit(model1_2)
mds.fit(model2_2)

np.delete(model1_2.W, 28)
np.delete(model2_2.W, 32)

model1_2.W = np.abs(model1_2.W)
mean1 = np.mean(model1_2.W)
model1_2.W = model1_2.W/mean1 

model2_2.W = np.abs(model2_2.W)
mean2 = np.mean(np.abs(model2_2.W))
model2_2.W = model2_2.W/mean2


# make the new models

model1_2_2 = mds.model(1,500,"k+1",banana1)

model1_2_2.normalize()

model2_2_2 = mds.model(1,500,"k+1",banana2)

model2_2_2.normalize()

#remove the features at the 30% threshold
arr1 = np.array([], dtype=np.int32)
for i in range(28):
    if model1_2.W[i] < 0.3:
        arr1 = np.append(arr1, i)        
print(arr1)        
model1_2_2.remove(arr1)

arr1 = np.array([], dtype=np.int32)
for i in range(32):
    if model2_2.W[i] < 0.3:
        arr1 = np.append(arr1, i)  
print(arr1)
model2_2_2.remove(arr1)


                        #Remove the features at the 120% threshold
mds.fit(model1_3)
mds.fit(model2_3)

np.delete(model1_3.W, 28)
np.delete(model2_3.W, 32)


model1_3.W = np.abs(model1_3.W)
mean1 = np.mean(model1_3.W)
model1_3.W = model1_3.W/mean1 



model2_3.W = np.abs(model2_3.W)
mean2 = np.mean(model2_3.W)
model2_3.W = model2_3.W/mean2


# make the new models

model1_3_2 = mds.model(1,500,"k+1",banana1)

model1_3_2.normalize()

model2_3_2 = mds.model(1,500,"k+1",banana2)

model2_3_2.normalize()

#remove the features from new models at a threshold of 120%
arr1 = np.array([], dtype=np.int32)
for i in range(28):
    if model1_3.W[i] < 1.2:
        arr1 = np.append(arr1, i)

model1_3_2.remove(arr1)

arr1 = np.array([], dtype=np.int32)
for i in range(32):
    if model2_3.W[i] < 1.2:
        arr1 = np.append(arr1, i)

model2_3_2.remove(arr1)

#now that we have all we need for the models with removed features, generate the other models

#all models for  CKD

m0del1_1 = mds.model(1e-1,500,"constant",banana1)
m0del1_2 = mds.model(1e-3,500,"constant",banana1)
m0del1_3 = mds.model(1e-5, 500, "constant", banana1)

m0del1_4 = mds.model(10,500,"k+1",banana1)
m0del1_5 = mds.model(1e-1,500,"k+1",banana1)
m0del1_6 = mds.model(1e-3, 500, "k+1", banana1)

m0del1_7 = model1_1

#models with features added for CKD

m0del1_8 = mds.model(1,500,"k+1",banana1)
m0del1_9 = mds.model(1, 500, "k+1", banana1)

#add quadratic and cubic features
for i in range(m0del1_8.X.shape[1]-1):
   m0del1_8.quadratic_feature(i)

for i in range(m0del1_9.X.shape[1]-1):
    m0del1_9.quadratic_feature(i)

for i in range(32):
    m0del1_9.cubic_feature(i)

#normalize all of their data except 7, 10, 11

m0del1_1.normalize()
m0del1_2.normalize()
m0del1_3.normalize()
m0del1_4.normalize()
m0del1_5.normalize()
m0del1_6.normalize()
m0del1_8.normalize()
m0del1_9.normalize()


#models with features removed, data already normalized
    #30% threshold
m0del1_10 = model1_2_2 
    #120% threshold
m0del1_11 = model1_3_2

#array of all models for CKD dataset
CKD_models = [m0del1_1, m0del1_2, m0del1_3, m0del1_4, m0del1_5, m0del1_6, m0del1_7, m0del1_8,
              m0del1_9, m0del1_10, m0del1_11]

#all models for  Battery

m0del2_1 = mds.model(1e-1,500,"constant",banana2)
m0del2_2 = mds.model(1e-3,500,"constant",banana2)
m0del2_3 = mds.model(1e-5, 500, "constant", banana2)

m0del2_4 = mds.model(10,500,"k+1",banana2)
m0del2_5 = mds.model(1e-1,500,"k+1",banana2)
m0del2_6 = mds.model(1e-3, 500, "k+1", banana2)

m0del2_7 = model2_1

#instantiate models with features added

m0del2_8 = mds.model(1,500,"k+1",banana2)
m0del2_9 = mds.model(1, 500, "k+1", banana2)

#add features to models
for i in range(m0del2_8.X.shape[1]-1):
   m0del2_8.quadratic_feature(i)

for i in range(m0del2_9.X.shape[1]-1):
    m0del2_9.quadratic_feature(i)

for i in range(32):
    m0del2_9.cubic_feature(i)

#normalize all of their data except 7, 10, 11

m0del2_1.normalize()
m0del2_2.normalize()
m0del2_3.normalize()
m0del2_4.normalize()
m0del2_5.normalize()
m0del2_6.normalize()
m0del2_8.normalize()
m0del2_9.normalize()


#models with features removed, data already normalized (for Battery this time)
    #30% threshold
m0del2_10 = model2_2_2 
    #120% threshold
m0del2_11 = model2_3_2

#array of each model for the battery dataset for crossvalidation
battery_models = [m0del2_1, m0del2_2, m0del2_3, m0del2_4, m0del2_5, m0del2_6, m0del2_7, m0del2_8,
              m0del2_9, m0del2_10, m0del2_11]



#crossavlidate returns an array of the accuracies of each model given
data_csvfile[:,0] = mds.crossvalidate_10(CKD_models)
data_csvfile[:,1] = mds.crossvalidate_10(battery_models)


sub_arr = data_csvfile

rows = 12
cols = 3

arr = np.full((rows, cols), np.nan)

# Fill the first row and first column
arr[0, 1:] = np.arange(1, cols)
arr[1:, 0] = np.arange(1, rows)

# Define insertion point (row, col) where sub_arr[0,0] should go
start_row, start_col = 1, 1  

# Insert the sub-array into the main array
arr[start_row:start_row + sub_arr.shape[0], start_col:start_col + sub_arr.shape[1]] = sub_arr
print(arr)
df = pd.DataFrame(arr)
print(arr1)
# Save to CSV
df.to_csv("crossvalidation_accuracies.csv", index=False, header=False)

