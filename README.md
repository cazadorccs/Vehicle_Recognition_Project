
# Use of Deeplearning Models for the Recognition of Emergency Vehicles

Classification and recognition of emergency vehicles in traffic surveillance cameras offer an early warning to ensure the prompt reaction to the traffic about emergency stating. Much existing traffic surveillance camera detection monitoring based on computer vision technology has achieved high detection rates. Some of them are very innovative and have a higher novelty index.

One of the innovative ideas is detecting an emergency vehicle in the moving traffic routes from the input of a video surveillance camera.
Experiments conducted on various benchmarking databases show that the proposed algorithm successfully distinguishes the emergency vehicles from the traffic in a surveillance camera. The results show that the proposed algorithm has an innovative classification method, a higher accuracy rate, as well as comparable recall and specificity, compared with other methods.

This paper proposes an algorithm that addresses a novel visual analysis technique for the detection of a moving emergency vehicle in traffic surveillance camera using the fundamentals of convolution neural network.

Experiments conducted on several benchmark databases show that the proposed algorithm successfully distinguishes emergency vehicles from a test image. The results show that the proposed algorithm has an innovative classification method, higher accuracy rate, as well as comparable recall and specificity, compared to other methods.


## Deployment

To deploy this project, we created a very simple artificial neural network in Python with Keras and Tensorflow as backend, we learned to use a Python development environment with Anaconda and Jupyter Notebook, we also used numpy to handle Keras arrays, we imported the Sequential model type and the "normal" Dense layer type.

Finally we evaluated and predicted obtaining satisfactory results.

You must run it according to the following requirements:

1.  Windows operating system

2. Install Anaconda: 
you can download it from (https://www.anaconda.com/products/individual "Anaconda Individual Edition
Download For Windows")

![Anaconda](https://github.com/cazadorccs/Vehicle_Recognition_Project/blob/main/Anaconda.png)
    :target: https://www.anaconda.com/products/individual

 After installing Anaconda you can find it here

 ![Menu_windows]((https://github.com/cazadorccs/Vehicle_Recognition_Project/blob/main/Menu_windows.png)

3. We have to open the anaconda prompt so from here we just install pip install


Keras and TensorFlow are open source Python libraries for working with neural networks, creating machine learning models and performing deep learning. Because Keras is a high level API for TensorFlow, they are installed together.

In general, there are two ways to install Keras and TensorFlow:

Install a Python distribution that includes hundreds of popular packages (including Keras and TensorFlow) such as ActivePython.
Use pip to install TensorFlow, which will also install Keras at the same time.

we use pip to install

**TensorFlow Requirements**

TensorFlow and Keras require Python 3.6+ (Python 3.8 requires TensorFlow 2.2+) , and the latest version of pip. You can determine the version of Python installed on your computer by running the following command:


```bash
    python3 --version
```
Output should be similar to:
```bash
    Python 3.8.2
```
Run the following command to ensure that the latest version of pip is installed:
```bash
pip install --upgrade pip
```
To install TensorFlow for CPU and GPU processors, run the following command:
```bash
pip install tensorflow
```
The installation installs a slew of TensorFlow and Keras dependencies:
```bash
tensorflow                                 
├── absl-py~=0.10                          
│   └── six                                
├── astunparse~=1.6.3                      
│   ├── six<2.0,>=1.6.1                    
│   └── wheel<1.0,>=0.23.0                 
├── flatbuffers~=1.12.0                    
├── gast==0.3.3                            
├── google-pasta~=0.2                      
│   └── six                                
├── grpcio~=1.32.0                         
│   └── six>=1.5.2                         
├── h5py~=2.10.0                           
│   ├── numpy>=1.7                         
│   └── six                                
├── keras-preprocessing~=1.1.2             
│   ├── numpy>=1.9.1                       
│   └── six>=1.9.0                         
├── numpy~=1.19.2                          
├── opt-einsum~=3.3.0                      
│   └── numpy>=1.7                         
├── protobuf>=3.9.2                        
│   └── six>=1.9                           
├── six~=1.15.0                            
├── tensorboard~=2.4                       
│   ├── absl-py>=0.4                       
│   │   └── six                            
│   ├── google-auth-oauthlib<0.5,>=0.4.1   
│   │   ├── google-auth>=1.0.0             
│   │   │   ├── cachetools<5.0,>=2.0.0     
│   │   │   ├── pyasn1-modules>=0.2.1      
│   │   │   │   └── pyasn1<0.5.0,>=0.4.6   
│   │   │   ├── rsa<5,>=3.1.4              
│   │   │   │   └── pyasn1>=0.1.3          
│   │   │   ├── setuptools>=40.3.0         
│   │   │   └── six>=1.9.0                 
│   │   └── requests-oauthlib>=0.7.0       
│   │       ├── oauthlib>=3.0.0            
│   │       └── requests>=2.0.0            
│   │           ├── certifi>=2017.4.17     
│   │           ├── chardet<5,>=3.0.2      
│   │           ├── idna<3,>=2.5           
│   │           └── urllib3<1.27,>=1.21.1  
│   ├── google-auth<2,>=1.6.3              
│   │   ├── cachetools<5.0,>=2.0.0         
│   │   ├── pyasn1-modules>=0.2.1          
│   │   │   └── pyasn1<0.5.0,>=0.4.6       
│   │   ├── rsa<5,>=3.1.4                  
│   │   │   └── pyasn1>=0.1.3              
│   │   ├── setuptools>=40.3.0             
│   │   └── six>=1.9.0                     
│   ├── grpcio>=1.24.3                     
│   │   └── six>=1.5.2                     
│   ├── markdown>=2.6.8                    
│   ├── numpy>=1.12.0                      
│   ├── protobuf>=3.6.0                    
│   │   └── six>=1.9                       
│   ├── requests<3,>=2.21.0                
│   │   ├── certifi>=2017.4.17             
│   │   ├── chardet<5,>=3.0.2              
│   │   ├── idna<3,>=2.5                   
│   │   └── urllib3<1.27,>=1.21.1          
│   ├── setuptools>=41.0.0                 
│   ├── six>=1.10.0                        
│   ├── tensorboard-plugin-wit>=1.6.0      
│   ├── werkzeug>=0.11.15                  
│   └── wheel>=0.26                        
├── tensorflow-estimator<2.5.0,>=2.4.0    
├── termcolor~=1.1.0                       
├── typing-extensions~=3.7.4               
├── wheel~=0.35                            
└── wrapt~=1.12.1
```
In the diagram above we can see that Numpy and Keras are installed among the dependencies.

But we must install other dependencies such as seaborn

Seaborn is a Python data visualization library based on matplotlib. 
It provides a high-level interface for drawing attractive and informative statistical .

```bash
pip install seaborn
```
4. Now we only have to download our data set and the project
you can download it from (https://disk.yandex.com/d/1hi6kB_WQUp42g "is a compressed .rar file, that is the complete structure of our project.")

