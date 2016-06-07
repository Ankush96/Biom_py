
# coding: utf-8

# In[13]:

import numpy as np
import pandas as pd
from skimage import io, color
from skimage.util import img_as_float, img_as_ubyte
#%matplotlib inline
import os
import glob
import time


# In[14]:

DATA_DIR = os.path.abspath("../Data/test-ssrbc2015/segmentation")

def get_paths(directory):
    """Gets the filenames of all sclera images in the given directory along with their 
        ground truth images
        Args:
            directory: The path to the root folder
        Output:
            images: List of paths to files containing images
            ground: List of paths to files containing corresponding ground truth images
    """
    imgs = glob.glob(DATA_DIR+"/[0-9]*/E*.jpg")
    imgs.sort()
    gt = glob.glob(DATA_DIR+"/[0-9]*/M*.jpg")
    gt.sort()
    
    # The following lines of code remove those examples whose image and ground truth 
    # sizes don't match
    images = []
    ground = []
    for i in range(len(imgs)):
        img = io.imread(imgs[i])
        g = io.imread(gt[i])
        if g.shape[0] == img.shape[0] and g.shape[1] == img.shape[1]:
            images.append(imgs[i])
            ground.append(gt[i])

    return images, ground

data, ground_truth = get_paths(DATA_DIR)


# In[15]:

# Hyper Params
TRAIN_PATCHES = 1200
VALIDATION_PATCHES = TRAIN_PATCHES/3
TEST_PATCHES = TRAIN_PATCHES/3
PATCHES_PER_IMAGE = 100
POSITIVE_PROPORTION = 0.5               # Proportion of patches having positive class
NUM_IMAGES = (TRAIN_PATCHES+VALIDATION_PATCHES+TEST_PATCHES)/PATCHES_PER_IMAGE
PATCH_DIM = 31                          # Dimension of window used as a sample

current_img_index = -1                  # Index of the current image in 'train'
current_img = io.imread(data[0])
current_gt = img_as_float(io.imread(ground_truth[0]))


# In[16]:

def load_next_img(data, gt_data):
    """When we have extracted 'PATCHES_PER_IMAGE' number of patches from our 
       current image we call this function to change the current image
       Args:
           data: The list of paths to the sclera images
           gt_data: List of paths to the corresponding ground truth images
       
    """
    global current_img_index, current_img, current_gt
    if current_img_index < NUM_IMAGES-1 and current_img_index < len(data) - 1:
        current_img_index +=1
        print "Working on image %d"%(current_img_index + 1)
        current_img = io.imread(data[current_img_index])                    
        current_gt = img_as_float(io.imread(gt_data[current_img_index])) 
        # Some ground truth images are loaded as 3 channel images, which we convert
        if (current_gt.shape) > 2:
            current_gt = color.rgb2gray(current_gt)
        return True
    else:
        print 'End of data extraction'
        return False


# In[18]:

begin = time.time()
print "Creating DataFrame"
df = pd.DataFrame(index = np.arange(NUM_IMAGES*PATCHES_PER_IMAGE), columns = np.arange(PATCH_DIM**2*3+1))
print "Dataframe ready"


# In[6]:

def save_img_data(data, gt_data):
    """Extracts PATCHES_PER_IMAGE number of patches from each image
        
       It maintains a count of positive and negative patches and maintains
       the ratio POSITIVE_PROPORTION = pos/(pos+neg)
       Args:
           data: The list of paths to the sclera images
           gt_data: List of paths to the corresponding ground truth images 
    
    """
    pos_count = 0
    neg_count = 0
    global df
    while pos_count +neg_count < PATCHES_PER_IMAGE: 
        # Choose a random point
        i = np.random.randint(PATCH_DIM/2,current_img.shape[0]-PATCH_DIM/2-1)
        j = np.random.randint(PATCH_DIM/2,current_img.shape[1]-PATCH_DIM/2-1)
        
        h = (PATCH_DIM - 1)/2
        ind = current_img_index*PATCHES_PER_IMAGE+pos_count+neg_count
        # If a positive sample is found and positive count hasn't reached its limit
        if int(current_gt[i,j])==1 and pos_count < POSITIVE_PROPORTION*PATCHES_PER_IMAGE:
            df.loc[ind][0:-1] = np.reshape(current_img[i-h:i+h+1,j-h:j+h+1], -1)
            df.loc[ind][PATCH_DIM**2*3] = int(current_gt[i,j])
            pos_count += 1
        # If a negative sample is found and negative count hasn't reached its limit
        elif int(current_gt[i,j])==0 and neg_count < (1-POSITIVE_PROPORTION)*PATCHES_PER_IMAGE:
            df.loc[ind][0:-1] = np.reshape(current_img[i-h:i+h+1,j-h:j+h+1], -1)
            df.loc[ind][PATCH_DIM**2*3] = int(current_gt[i,j])
            neg_count += 1
   


# In[19]:

while load_next_img(data, ground_truth):
    start = time.time()
    save_img_data(data, ground_truth)
    print "Time taken for this image = %f secs" %( (time.time()-start))


# ### Split data into Train, Validation and Test sets

# In[20]:

train_df = df[:TRAIN_PATCHES]
valid_df = df[TRAIN_PATCHES:TRAIN_PATCHES+VALIDATION_PATCHES]
test_df = df[TRAIN_PATCHES+VALIDATION_PATCHES:TRAIN_PATCHES+VALIDATION_PATCHES+TEST_PATCHES]
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ### Mean Normalising the data

# In[21]:

print "Mean Normalising"
last = len(df.columns) -1
mean_img = np.mean(train_df)[:-1]

train_labels = train_df[last]
valid_labels = valid_df[last]
test_labels = test_df[last]

mean_normalised_train_df = train_df - np.mean(train_df)
mean_normalised_train_df[last] = train_labels

mean_normalised_valid_df = valid_df - np.mean(train_df)
mean_normalised_valid_df[last] = valid_labels

mean_normalised_test_df = test_df - np.mean(train_df)
mean_normalised_test_df[last] = test_labels


# In[22]:

print "Randomly shuffling the datasets"
mean_normalised_train_df = mean_normalised_train_df.iloc[np.random.permutation(len(train_df))]
mean_normalised_train_df = mean_normalised_train_df.reset_index(drop=True)
mean_normalised_test_df = mean_normalised_test_df.iloc[np.random.permutation(len(test_df))]
mean_normalised_test_df = mean_normalised_test_df.reset_index(drop=True)
mean_normalised_valid_df = mean_normalised_valid_df.iloc[np.random.permutation(len(valid_df))]
mean_normalised_valid_df = mean_normalised_valid_df.reset_index(drop=True)


# In[24]:

print "Writing to pickle"
mean_img.to_pickle('../Data/test-ssrbc2015/segmentation/mean_img.pkl')
mean_normalised_train_df.to_pickle('../Data/test-ssrbc2015/segmentation/mn_train.pkl')
mean_normalised_valid_df.to_pickle('../Data/test-ssrbc2015/segmentation/mn_validation.pkl')
mean_normalised_test_df.to_pickle('../Data/test-ssrbc2015/segmentation/mn_test.pkl')


# In[25]:

print "Total time taken = %f mins" %( (time.time()-begin)/60.0)

