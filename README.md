# discription
Credit card fraud detection is a critical task for financial institutions and payment systems. With the rise in digital transactions, fraud detection systems must become more robust, scalable, and accurate. In this project, we use a Convolutional Neural Network (CNN) to detect fraudulent credit card transactions. We are leveraging deep learning techniques to identify patterns in the transaction data that differentiate fraudulent transactions from legitimate ones.

This project employs a CNN model that is specifically designed to handle imbalanced datasets, which is common in fraud detection, where fraudulent transactions are much fewer than legitimate ones. SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset before feeding it into the model. Additionally, techniques like batch normalization, dropout, and data scaling are employed to improve the accuracy of the model.

Objective:
Detect fraudulent credit card transactions using deep learning techniques.
Use Convolutional Neural Networks (CNN) for feature extraction and classification of transactions.
Improve model performance by addressing data imbalance and scaling features.


Project Description: Credit Card Fraud Detection Using Convolutional Neural Networks (CNN)
Overview:
Credit card fraud detection is a critical task for financial institutions and payment systems. With the rise in digital transactions, fraud detection systems must become more robust, scalable, and accurate. In this project, we use a Convolutional Neural Network (CNN) to detect fraudulent credit card transactions. We are leveraging deep learning techniques to identify patterns in the transaction data that differentiate fraudulent transactions from legitimate ones.

This project employs a CNN model that is specifically designed to handle imbalanced datasets, which is common in fraud detection, where fraudulent transactions are much fewer than legitimate ones. SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset before feeding it into the model. Additionally, techniques like batch normalization, dropout, and data scaling are employed to improve the accuracy of the model.

##Objective:
Detect fraudulent credit card transactions using deep learning techniques.
Use Convolutional Neural Networks (CNN) for feature extraction and classification of transactions.
Improve model performance by addressing data imbalance and scaling features.
Key Concepts and Techniques Used:
Data Preprocessing:

Missing Data Handling: The dataset is checked for missing values and columns with no significance (e.g., "Time" is dropped).
Scaling: The "Amount" feature is normalized using StandardScaler to ensure uniformity across features.
Class Renaming: The target column "Class" is renamed to "class" for consistency.
Handling Missing Classes: Any rows with missing target values are dropped.
Handling Imbalanced Data:

SMOTE (Synthetic Minority Over-sampling Technique): Since fraud cases are rare, we use SMOTE to generate synthetic samples for the minority class (fraud cases) to balance the dataset.
Model Architecture:

1D Convolutional Layers (Conv1D): Convolutional layers are used to capture the local patterns and relationships between features in a 1D structure.
Batch Normalization: Used to normalize the activations and gradients, which helps in faster training and better performance.
Dropout: To prevent overfitting, dropout layers are added with different rates to remove certain connections during training.
Max Pooling: This reduces the dimensionality of the features, summarizing the most important aspects of the features extracted by the convolutional layers.
Dense Layers: Fully connected layers for final classification. A sigmoid activation function is used to classify transactions as fraud (1) or non-fraud (0).
Optimization: Adam optimizer is used with a learning rate of 0.0001 to optimize the model.
Model Training and Evaluation:

The model is trained for 10 epochs, and CPU time is measured for training.
Performance Metrics: The model is evaluated based on accuracy, precision, recall, and F1 score to measure its effectiveness in classifying fraud cases.
Confusion Matrix: The confusion matrix is plotted to visualize the number of true positives, false positives, true negatives, and false negatives.
Mixed Precision Training:

BFloat16 Precision: The model is trained using bfloat16 (a lower precision type) to reduce memory usage and improve computational efficiency without sacrificing model accuracy.
Dataset:
The dataset used in this project is the Credit Card Fraud Detection dataset, which contains transactions made by credit cards. It includes the following columns:

Features: Various anonymized features about each transaction (e.g., transaction amount, anonymized variables representing card features).
Target Label (class): The target variable indicating whether the transaction was fraud (1) or non-fraud (0).
Model Flow:
Data Import and Preprocessing:

Load the dataset using pandas.
Clean the dataset by handling missing values and scaling numerical features.
Apply SMOTE to balance the class distribution.
Split the data into training and testing sets.
Normalize the features using StandardScaler.
Model Building:

Create a Sequential CNN model using Conv1D, BatchNormalization, Dropout, and Dense layers.
Compile the model using the Adam optimizer and binary cross-entropy loss.
Train the model on the processed training data.
Evaluate the model performance using metrics such as accuracy, precision, recall, F1 score, and the confusion matrix.
Performance Evaluation:

The model's performance is evaluated on the testing set.
CPU time is measured to analyze training efficiency.
The results are visualized using matplotlib and seaborn to plot the confusion matrix and compare CPU times with and without mixed precision training.
Learning Curves:

Learning curves for both accuracy and loss are plotted to visualize model performance over training epochs.
Results:
The CNN model trained on the credit card fraud dataset using bfloat16 precision showed faster training times and reasonable accuracy on detecting fraud.
The model achieved good classification results based on precision, recall, and F1 score, which are critical metrics for fraud detection, as it’s more important to correctly identify fraud cases than non-fraud cases.
Confusion Matrix helped in visualizing the performance in terms of false positives and false negatives.
Conclusion:
This project demonstrates how Convolutional Neural Networks (CNNs) can be effectively applied to credit card fraud detection by handling imbalanced data and using advanced techniques like SMOTE for data balancing and mixed precision training for computational efficiency. By using deep learning, the model can learn complex patterns in the transaction data and predict fraudulent activities with high accuracy, making it an essential tool for financial institutions and payment systems to prevent fraud.

##Key Libraries Used:
TensorFlow/Keras for building the CNN model.
Pandas for data manipulation and preprocessing.
Scikit-learn for data scaling and splitting, as well as evaluation metrics.
Imbalanced-learn (SMOTE) for balancing the dataset.
Matplotlib & Seaborn for data visualization.



##Model Architecture:

1D Convolutional Layers (Conv1D): Convolutional layers are used to capture the local patterns and relationships between features in a 1D structure.
Batch Normalization: Used to normalize the activations and gradients, which helps in faster training and better performance.
Dropout: To prevent overfitting, dropout layers are added with different rates to remove certain connections during training.
Max Pooling: This reduces the dimensionality of the features, summarizing the most important aspects of the features extracted by the convolutional layers.
Dense Layers: Fully connected layers for final classification. A sigmoid activation function is used to classify transactions as fraud (1) or non-fraud (0).
Optimization: Adam optimizer is used with a learning rate of 0.0001 to optimize the model.

##Conclusion:
This project demonstrates how Convolutional Neural Networks (CNNs) can be effectively applied to credit card fraud detection by handling imbalanced data and using advanced techniques like SMOTE for data balancing and mixed precision training for computational efficiency. By using deep learning, the model can learn complex patterns in the transaction data and predict fraudulent activities with high accuracy, making it an essential tool for financial institutions and payment systems to prevent fraud.

