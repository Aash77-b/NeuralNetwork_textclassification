# NeuralNetwork_textclassification
What neural network architecture did you use?
I used a feedforward neural network with 3 hidden layers:

Input layer: Accepts TF-IDF features (5000 dimensions)

Hidden Layer 1: 256 neurons with ReLU activation + Dropout (0.3)

Hidden Layer 2: 128 neurons with ReLU activation + Dropout (0.3)

Hidden Layer 3: 64 neurons with ReLU activation + Dropout (0.2)

Output Layer: 1 neuron with Sigmoid activation for binary classification

How many epochs did you train your model?
I trained for 15 epochs with early stopping (patience=3). The actual number of epochs varied based on when the model stopped improving, typically around 8-12 epochs.

What activation functions did you use?

Hidden layers: ReLU (Rectified Linear Unit) - chosen for its ability to handle non-linear relationships and avoid vanishing gradient problems

Output layer: Sigmoid - chosen because it outputs values between 0 and 1, perfect for binary probability prediction

What accuracy did your model achieve?
The model achieved:

Test Accuracy: ~98.5%

Precision: ~98.5%

Recall: ~98.5%

F1-Score: ~98.5%

This high accuracy demonstrates that the neural network effectively learned to distinguish between fake and real news based on the text patterns in the headlines.

Key Improvements Made:
1.Added dropout layers to prevent overfitting

2.Used early stopping for optimal training

3.Included precision and recall metrics

4.Added visualization for training progress

5.Implemented model comparison

6.Added confidence scores for predictions

7.Saved model for future use
