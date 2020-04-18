# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
## Data Capture
I record 2 laps driving in one direction of the track and also 2 more laps driving in the opposite direction to make sure the turns are reversed.
Left| Center| Right
----|-------|-------
![left](https://i.imgur.com/5fSyJ7p.jpg)|![Center](https://i.imgur.com/dg34jLD.jpg)|![Right](https://i.imgur.com/80XBux3.jpg)


## Preprocess
### 1.Divide data
I have learned is to have 20–30% of training data as the validation set to compare the validation loss and training loss so that it can avoid overfitting.
```
train_len = int(0.8*len(samples))
valid_len = len(samples) - train_len
train_samples, validation_samples = data.random_split(samples, lengths=[train_len, valid_len])
```
### 2.Loading Images in Dataloader
In Pytorch, I use the Dataset class and the Dataloader function to generator images.First,I defined a function to take an image and randomly crop it and flip it horizontally along with taking the negative of the steering data.The cropping, basically, helps the model to focus only on the road by taking away the sky and other distracting stuff in the image and flipping is done to make sure the images are generalized to left and right turns, essentially keeping the car at the center of the road. 

```
def augment(imgName, angle):
  name = 'data/IMG/' + imgName.split('/')[-1]
  current_image = cv2.imread(name)
  current_image = current_image[65:-25, :, :]
  if np.random.rand() < 0.5:
    current_image = cv2.flip(current_image, 1)
    angle = angle * -1.0  
  return current_image, angle
```
In order to implement this, we would have to overload a few functions of the class, namely, the __getitem__, __len__ and __init__ functions.
I define the Dataloader class and pass on this augment function to the input batch samples, concatenate the steering data and images and return it.

```
class Dataset(data.Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index)
        steering_angle = float(batch_samples[3])
        center_img, steering_angle_center = augment(batch_samples[0], steering_angle)
        left_img, steering_angle_left = augment(batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right = augment(batch_samples[2], steering_angle - 0.4)
        center_img = self.transform(center_img)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)
        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)
      
    def __len__(self):
        return len(self.samples)
```
### 3.Normalize and set data
I defined transformations to normalize the image array values to the range 0–1 using a lambda function. And use Dataloader function to add everything in a generator that will be called batch-wise during training. We define a batch size of 32 and shuffle them while passing it to the DataLoader.
```
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 0}

transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

training_set = Dataset(train_samples, transformations)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(validation_samples, transformations)
validation_generator = data.DataLoader(validation_set, **params)
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
First,I used the Nvidia's paper ```End to End Learning for Self-Driving Cars``` to train my model.This model used strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. 

![](https://i.imgur.com/gdN30IU.png)

However,it was not good performance. So I find another [repo](https://github.com/hminle/car-behavioral-cloning-with-pytorch/) to modeified the model. This is the architicture:

```
NetworkLight(
  (conv_layers): Sequential(
    (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2))
    (1): ELU(alpha=1.0)
    (2): Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2))
    (3): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
    (4): Dropout(p=0.25)
  )
  (linear_layers): Sequential(
    (0): Linear(in_features=3648, out_features=50, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=50, out_features=10, bias=True)
    (3): Linear(in_features=10, out_features=1, bias=True)
  )
)
```
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128.Compared with original model,I reduce the three convolution layers and add MaxPool2d layer. It will be more stable than original paper's architecture.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

```
optimizer = optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.MSELoss()
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I use 22 ephochs to train my model, and it get good performance.


## Ourput Video

The output video i put on the youtube and video.mp4 in file.

[![](https://i.imgur.com/pwB2PAh.jpg)](https://www.youtube.com/watch?v=eZNhmmIW7ao&t=13s)




