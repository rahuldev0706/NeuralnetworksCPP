#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>  // For timing
#include <thread>  // For sleep

// Simple random number generator for weight initialization
double random(double min, double max) {
    return min + (max - min) * (static_cast<double>(rand()) / RAND_MAX);
}

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid for backpropagation
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

class NeuralNetwork {
private:
    std::vector<double> input;              // Input layer
    std::vector<double> hidden;             // Hidden layer
    std::vector<double> output;             // Output layer
    std::vector<std::vector<double>> w1;    // Weights: input -> hidden
    std::vector<std::vector<double>> w2;    // Weights: hidden -> output
    std::vector<double> bias1;              // Bias for hidden layer
    std::vector<double> bias2;              // Bias for output layer
    double learning_rate;

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double lr = 0.1) 
        : learning_rate(lr) {
        // Initialize layers
        input.resize(input_size);
        hidden.resize(hidden_size);
        output.resize(output_size);
        bias1.resize(hidden_size);
        bias2.resize(output_size);

        // Initialize weights and biases with random values
        srand(static_cast<unsigned>(time(0)));
        w1.resize(input_size, std::vector<double>(hidden_size));
        w2.resize(hidden_size, std::vector<double>(output_size));
        
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                w1[i][j] = random(-1.0, 1.0);
            }
        }
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                w2[i][j] = random(-1.0, 1.0);
            }
            bias1[i] = random(-1.0, 1.0);
        }
        for (int j = 0; j < output_size; ++j) {
            bias2[j] = random(-1.0, 1.0);
        }
    }

    // Forward pass with step-by-step logging
    void forward(const std::vector<double>& input_data) {
        std::cout << "Step 1: Forward Propagation\n";
        input = input_data;

        // Input -> Hidden
        std::cout << "  Computing hidden layer...\n";
        for (int j = 0; j < hidden.size(); ++j) {
            hidden[j] = bias1[j];
            for (int i = 0; i < input.size(); ++i) {
                hidden[j] += input[i] * w1[i][j];
            }
            std::cout << "    Hidden neuron " << j << " pre-activation: " << hidden[j];
            hidden[j] = sigmoid(hidden[j]);
            std::cout << ", post-activation: " << hidden[j] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Sleep 200ms
        }

        // Hidden -> Output
        std::cout << "  Computing output layer...\n";
        for (int j = 0; j < output.size(); ++j) {
            output[j] = bias2[j];
            for (int i = 0; i < hidden.size(); ++i) {
                output[j] += hidden[i] * w2[i][j];
            }
            std::cout << "    Output neuron " << j << " pre-activation: " << output[j];
            output[j] = sigmoid(output[j]);
            std::cout << ", post-activation: " << output[j] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Sleep 200ms
        }
        std::cout << "Forward pass complete.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Sleep 500ms
    }

    // Backward pass with step-by-step logging
    void backward(const std::vector<double>& target) {
        std::cout << "Step 2: Backward Propagation\n";

        // Compute output layer error
        std::cout << "  Computing output layer errors...\n";
        std::vector<double> output_error(output.size());
        for (int j = 0; j < output.size(); ++j) {
            output_error[j] = (target[j] - output[j]) * sigmoid_derivative(output[j]);
            std::cout << "    Output error " << j << ": " << output_error[j] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Sleep 200ms
        }

        // Compute hidden layer error
        std::cout << "  Computing hidden layer errors...\n";
        std::vector<double> hidden_error(hidden.size());
        for (int i = 0; i < hidden.size(); ++i) {
            hidden_error[i] = 0.0;
            for (int j = 0; j < output.size(); ++j) {
                hidden_error[i] += output_error[j] * w2[i][j];
            }
            hidden_error[i] *= sigmoid_derivative(hidden[i]);
            std::cout << "    Hidden error " << i << ": " << hidden_error[i] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Sleep 200ms
        }

        // Update weights and biases (hidden -> output)
        std::cout << "  Updating weights (hidden -> output)...\n";
        for (int i = 0; i < hidden.size(); ++i) {
            for (int j = 0; j < output.size(); ++j) {
                w2[i][j] += learning_rate * output_error[j] * hidden[i];
            }
        }
        for (int j = 0; j < output.size(); ++j) {
            bias2[j] += learning_rate * output_error[j];
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300)); // Sleep 300ms

        // Update weights and biases (input -> hidden)
        std::cout << "  Updating weights (input -> hidden)...\n";
        for (int i = 0; i < input.size(); ++i) {
            for (int j = 0; j < hidden.size(); ++j) {
                w1[i][j] += learning_rate * hidden_error[j] * input[i];
            }
        }
        for (int j = 0; j < hidden.size(); ++j) {
            bias1[j] += learning_rate * hidden_error[j];
        }
        std::cout << "Backward pass complete.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Sleep 500ms
    }

    // Train the network with step logging
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<std::vector<double>>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            std::cout << "\nEpoch " << epoch << ":\n";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << "  Training on input: [" << inputs[i][0] << ", " << inputs[i][1] << "]\n";
                forward(inputs[i]);
                backward(targets[i]);
                total_error += pow(targets[i][0] - output[0], 2);
            }
            if (epoch % 100 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << epoch << ", Average Error: " << total_error / inputs.size() << "\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Sleep 1s between epochs
        }
    }

    // Predict
    double predict(const std::vector<double>& input_data) {
        forward(input_data);
        return output[0];
    }
};

int main() {
    // XOR-like problem dataset
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Create and train the network (reduced epochs for demo)
    NeuralNetwork nn(2, 4, 1, 0.1); // 2 inputs, 4 hidden, 1 output
    nn.train(inputs, targets, 5);   // Train for 5 epochs to keep output manageable

    // Test the network
    std::cout << "\nTesting the network:\n";
    for (const auto& input : inputs) {
        double prediction = nn.predict(input);
        std::cout << input[0] << ", " << input[1] << " -> " << prediction 
                  << " (" << (prediction > 0.5 ? 1 : 0) << ")\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Sleep 500ms
    }

    return 0;
}