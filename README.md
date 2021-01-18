# AI_Questions_and_Answers
This repository is for content writing on basic and unusual Q&As related to artificial intelligence.<br/>
## Editor: Manohar Karanth


#### Q1: What is the idea behind connections in neural networks?
<code> Frank Rosenblatt (American Psychologist known for Rosenblatt Perceptron) had met a biologist who used to feed a dog and run a bell simultaneously. After some days when he had run the bell without feeding the dog, saliva came out from the dog’s mouth. They found out that Neurons that fire when eating food and neurons that fire while hearing bell were connected when fired multiple times together. This ‘Neurons which fire together wire together’ concept is the idea behind connections and weights in a neural network.</code>
#### Q2: Differentiate between a 3D vector and 3D tensor in TensorFlow.
<code> Difference is quite ambiguous. Both ‘number of entries along the axis’ and ‘number of axis in a tensor’ are the ‘dimensions’ of a tensor in TensorFlow. However, former is the associated with vector (nD-Vector) and latter is with tensor (nD-Tensor) </code> 
</br><code>Ex:  import numpy as np </code>
<code>     x= np.array([7,1,5]) 
           Here, x is a 3D-vector and 1D-tensor. </code>
#### Q3: Explain the notion behind the name ‘TensorFlow’.
<code>Tensors are nothing but multidimensional numpy arrays. These Tensors flow through the nodes/units of a Neural Network. While flowing many tensor operations are performed on them in each node. This ‘flow’ of Tensors between the nodes is the foundation of a Neural Network. So, Google named their open-source software library for deep learning as Tensorflow.</code>
#### Q4: Why is deep neural network (DNN) not an all-time solution over traditional ML?
<code>DNN provides High performance over regular ML and learns complex representations. However, there are four major drawbacks.</code>
- <code> DNNs are considered as ‘black box’ whereas Machine learning models are ‘white box’. DNNs lacks the much needed interpretability factor. In other words, one can’t interpret the patterns learnt by DNN unlike Traditional ML.</code>
- <code> DNNs require large data to learn complex patterns. </code>
- <code> DNNs usually need hardware accelerators. </code>
- <code> DNNs need substantial expertise for model tuning and architectural design.</code>
#### Q5: Why is neural network known as universal process approximator?
<code>Machine learning models are the representations of the processes in the real world. A process represents a relationship between the inputs and outputs. For example, model to predict sales is a process wherein relation between the past and the present sale is considered (since it is sequential). Using Neural network, we can represent any processes in the real world. Hence, they are called as universal process approximator.
</code>
#### Q6: Why is normalization preferred in deep learning over standardization?
<code>Normalization scales the variables value down between 0 and 1 (usually). On the other hand, standardization scales the variables based on the normal distribution. When we scale the data between standard deviations from mean, range of the values will be higher for the variables. Consequently, High range of values results in large gradients which deter network in converging into a global minimum. Normalization formats the data into a lower range which are compatible with weights of the network. </code>
#### Q7: Identify involvement of machine learning, statistics and robotics in each task.
- <code> Recommendation engine: ML and statistics only  </code>
- <code> Chatbot: ML only  </code>
- <code>Self-driving car: ML, statistics and Robotics  </code>
- <code>Rockets: robotics only  </code> 
#### Q8: Identify systems as AI or not.
- <code> Predicting mileage of a vehicle: AI </code>
- <code> Excel sheet that calculates predefined function on a data: not AI </code>
- <code> Big data processing using Hadoop: may or may not AI </code>
- <code> Chatbot: AI </code>
#### Q9: What is the meaning of ‘learning’ in deep learning?
<code>It is basically learning about what values to assign to each weight. For each epoch model computes the gradient of the loss function with respect to each weight that has been set earlier. This gradient will be multiplied with a learning rate (usually 0.01 to 0.001).  Weights will get updated  with calculated values by removing the previous ones. Assignment of weights are done based on how these incremental changes are affecting the loss function in each epoch.</code>
#### Q10: How do you calculate reconstruction error's threshold in autoencoder for anomaly detection?
<code>Fixing a value for autoencoder’s threshold in anomaly is challenging and tricky. It is highly dependent on the business problem and the required KPIs. There are broadly two approaches for finding the thresholds. </code>
- <code> When we have labelled data, based on the percentile of anomalies we fix the threshold. </code>
- <code> When we don’t have labelled data, we find it by optimizing against the loss function. </code>
#### Q11: Why is feature engineering less important in deep learning unlike traditional machine learning?
<code>Data scientists perform feature engineering to increase the quality of data representations. Features are transformed into new representations. Traditional ML algorithms’ hypothesis space is not rich enough to learn required features on their own. This necessitates a data scientist to perform feature engineering manually before feeding data to an algorithm. But deep learning models are capable to learn complex patterns without any assistance.</code>
#### Q12: How do deep learning models outperform traditional models in time series analysis?
<code>Deep learning models such as DeepAR outperform traditional time series models by incorporating following factors.</code>
- <code>Considers related variables which impact prediction accuracy. For ex: price is considered while forecasting sales. </code>
- <code>Takes metadata into account to handle seasonality. For ex: item metadata in sales.</code>
- <code>Handles new items with no historical data. For ex: If Adidas shoes are new, it will consider Nike shoes data (similar and available data). </code>
#### Q13: How to handle poor quality data?
<code> Data science is driven by data, without which one is just another person with opinion. Poor quality data is a common scenario and we can tackle it by following means</code>
- <code>Data cleaning: missing value imputations, outlier handing, balancing data through resampling. </code>
- <code>Data augmentation: generating more data from existing and available data. </code>
- <code>Data risk analysis: establishing data reliability and risk involved. </code> 
#### Q14: Why is Bag-of-Words not applied with deep learning?
<code>Bag-Of-Words (BoW) is a tokenization method which does not preserve the order. It generates tokens which are understood as texts and not as sequences. Structure of the sequences is also won’t be preserved. So, BoW is typically used in traditional machine learning and not used in deep learning.</code>
