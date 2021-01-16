# AI_Questions_and_Answers
This repository is for hosting unique Artificial Intelligence Questions and Answers.

#### Q1: What is the rationale behind connections in Neural Networks?
<code> Frank Rosenblatt (American Psychologist known for Rosenblatt Perceptron) had met a biologist. Biologist used to feed a dog and run a bell simultaneously. After some days when he ran the bell without feeding the dog, saliva came out from dog’s mouth. They found out that Neurons which fire when eating food and neurons that fire while hearing bell were connected when fires multiple times together. This ‘Neurons which fire together wire together’ concept is the reason behind connection in a neural network.</code>
#### Q2: Differentiate between a 3D vector and 3D tensor in TensorFlow.
<code> Difference is quite ambiguous. Both ‘number of entries along the axis’ and ‘number of axis in a Tensor’ are the ‘dimensions’ of a tensor in TensorFlow. However, former is the associated with vector (nD-Vector) and later is with Tensor (nD-Tensor) </code> 
</br><code>Ex:  import numpy as np </code>
<code>     x= np.array([7,1,5]) 
           Here, x is a 3D-vector and 1D-Tensor. </code>
#### Q3: Explain the notion behind the name ‘TensorFlow’.
<code>Tensors are nothing but multidimensional numpy arrays. These Tensors flow through the nodes/units of a Neural Network. While flowing many tensor operations are performed on them in each node. This ‘flow’ of Tensor between the nodes is fundamental of a Neural Network. So, Google named their open-source software library for deep learning as Tensorflow.</code>
#### Q4: Why is Deep Neural Network (DNN) not an all-time solution over traditional ML?
<code>DNN provides High performance over regular ML and learns complex representations. However, there are three major drawbacks.</code>
- <code> DNNs are considered as ‘black box’ whereas Machine learning models are ‘white box’. DNNs leave a lot to be desired in terms of interpretability. In other words, one can’t interpret the patterns learnt by DNN unlike Traditional ML.</code>
- <code> DNNs requires large data to learn complex patterns. </code>
- <code> DNNs needs substantial expertise for model tuning and architectural design.</code>
#### Q5: Why is Neural Network known as Universal Functional Approximator (UFA)?
<code>Machine learning models are the representations of the processes in the real world. A process represents a relationship between the inputs and outputs. For example, model to predict sales is a process wherein relation between the past and the present sale is considered (since it is sequential). Using Neural network, we can represent any processes in the real world. Hence, they are called as universal process approximator.
</code>
#### Q6: Why is normalization preferred in Deep Learning over standardization?
<code>Normalization scales the variables value down between 0 and 1 (usually). On the other hand, standardization scales the variables based on the normal distribution. When we scale the data between standard deviations from mean, range of the values will be higher for the variables. Consequently, High range of values results in large gradients which deter network in converging into a global minimum. Normalization formats the data into a lower range which are compatible with weights of the network. </code>
#### Q7: Identify involvement of machine learning, statistics and robotics in each tasks.
- <code> Recommendation engine: ML and statistics only  </code>
- <code> Chatbot: ML only  </code>
- <code>Self-driving car: ML, statistics and Robotics  </code>
- <code>Rockets: robotics only  </code> 
#### Q8: Identify systems as AI or not.
- <code> Predicting millage of a vehicle: AI </code>
- <code> Excel sheet that calculates predefined function on a data: not AI </code>
- <code> Big data processing using Hadoop: not AI </code>
- <code> Chatbot: AI </code>
#### Q9: What is the meaning of ‘learning’ in Deep Learning?
<code>It is basically learning about what values to assign to each weight. For each epoch model computes the gradient of the loss function with respect to each weight that has been set earlier. This gradient will be multiplied with a learning rate (0.01 to 0.001).  Weights will get updated  with calculated values by removing the previous one. Assignment of weights are done based on how these incremental changes are affecting the loss function in each epoch.</code>
#### Q10: How do you calculate threshold reconstruction error in autoencoder for anomaly detection?
<code>Fixing a value of autoencoder’s threshold in anomaly is challenging and tricky. It is highly dependent on the business problem and the required KPIs. There are broadly two approaches for finding the thresholds. </code>
- <code> When we have labelled data, based on the percentile of anomalies we fix the threshold. </code>
- <code> When we don’t have labelled data, we find it by optimizing against the loss function. </code>
#### Q11: Why is feature engineering less important in Deep learning unlike traditional machine learning?
<code>Data scientists perform feature engineering to increase the quality of data representation. Features are transformed into a new representation. Traditional ML algorithms’ hypothesis space is not rich enough to learn hypothesis space on their own. This necessitates a data scientist to perform feature engineering on his own before feeding data to an algorithm. However, deep learning models learn complex patterns without any assistance. This property solves a practitioner’s problem by great extent.</code>
#### Q12: How do deep learning models outperform traditional models in time series analysis?
<code>Deep learning models such as DeepAR outperform traditional time series models hands down by incorporating following factors.</code>
- <code>Considers related variables which impacts prediction accuracy. For ex: price </code>
- <code>Takes metadata into account to handle seasonality </code>
- <code>Handles new items with no historical data. For ex: If Adidas shoes are new, it will consider Nike shoes data (similar data available) </code>
