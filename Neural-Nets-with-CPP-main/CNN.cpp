#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

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

class CNN {
private:
    int input_size;                         // Input image size (e.g., 8x8)
    int filter_size;                        // Convolutional filter size (e.g., 3x3)
    int pool_size;                          // Pooling window size (e.g., 2x2)
    int hidden_size;                        // Fully connected layer size
    int output_size;                        // Output size (1 for binary classification)
    double learning_rate;

    std::vector<std::vector<double>> filter; // 2D convolution filter
    std::vector<std::vector<double>> feature_map; // Output of convolution
    std::vector<std::vector<double>> pooled; // Output of pooling
    std::vector<double> hidden;             // Fully connected layer
    std::vector<double> output;             // Output layer

    std::vector<double> Wfc;                // Fully connected weights (pooled -> hidden)
    std::vector<double> Wout;               // Hidden -> output weights
    double bfc;                             // Fully connected bias
    double bout;                            // Output bias

public:
    CNN(int in_size = 8, int filt_size = 3, int p_size = 2, int hid_size = 4, int out_size = 1, double lr = 0.1)
        : input_size(in_size), filter_size(filt_size), pool_size(p_size), hidden_size(hid_size), 
          output_size(out_size), learning_rate(lr) {
        // Initialize layers and weights
        filter.resize(filter_size, std::vector<double>(filter_size));
        feature_map.resize(input_size - filter_size + 1, std::vector<double>(input_size - filter_size + 1));
        pooled.resize((input_size - filter_size + 1) / pool_size, 
                      std::vector<double>((input_size - filter_size + 1) / pool_size));
        hidden.resize(hidden_size);
        output.resize(output_size);
        Wfc.resize(pooled.size() * pooled[0].size() * hidden_size);
        Wout.resize(hidden_size);

        // Random initialization
        srand(static_cast<unsigned>(time(0)));
        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                filter[i][j] = random(-1.0, 1.0);
            }
        }
        for (int i = 0; i < Wfc.size(); ++i) Wfc[i] = random(-1.0, 1.0);
        for (int i = 0; i < hidden_size; ++i) Wout[i] = random(-1.0, 1.0);
        bfc = random(-1.0, 1.0);
        bout = random(-1.0, 1.0);
    }

    // Convolution operation
    void convolve(const std::vector<std::vector<double>>& input) {
        std::cout << "Step 1: Convolution\n";
        for (int i = 0; i < feature_map.size(); ++i) {
            for (int j = 0; j < feature_map[0].size(); ++j) {
                feature_map[i][j] = 0.0;
                for (int fi = 0; fi < filter_size; ++fi) {
                    for (int fj = 0; fj < filter_size; ++fj) {
                        feature_map[i][j] += input[i + fi][j + fj] * filter[fi][fj];
                    }
                }
                std::cout << "  Feature map[" << i << "][" << j << "] = " << feature_map[i][j] << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Reduced for larger input
            }
        }
    }

    // Max pooling operation
    void max_pool() {
        std::cout << "Step 2: Max Pooling\n";
        for (int i = 0; i < pooled.size(); ++i) {
            for (int j = 0; j < pooled[0].size(); ++j) {
                double max_val = feature_map[i * pool_size][j * pool_size];
                for (int pi = 0; pi < pool_size; ++pi) {
                    for (int pj = 0; pj < pool_size; ++pj) {
                        max_val = std::max(max_val, feature_map[i * pool_size + pi][j * pool_size + pj]);
                    }
                }
                pooled[i][j] = max_val;
                std::cout << "  Pooled[" << i << "][" << j << "] = " << pooled[i][j] << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    // Fully connected layer
    void fully_connected() {
        std::cout << "Step 3: Fully Connected Layer\n";
        int pooled_flat_size = pooled.size() * pooled[0].size();
        std::vector<double> pooled_flat(pooled_flat_size);
        for (int i = 0; i < pooled.size(); ++i) {
            for (int j = 0; j < pooled[0].size(); ++j) {
                pooled_flat[i * pooled[0].size() + j] = pooled[i][j];
            }
        }

        for (int h = 0; h < hidden_size; ++h) {
            hidden[h] = bfc;
            for (int i = 0; i < pooled_flat_size; ++i) {
                hidden[h] += pooled_flat[i] * Wfc[h * pooled_flat_size + i];
            }
            hidden[h] = sigmoid(hidden[h]);
            std::cout << "  Hidden[" << h << "] = " << hidden[h] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        output[0] = bout;
        for (int i = 0; i < hidden_size; ++i) {
            output[0] += hidden[i] * Wout[i];
        }
        output[0] = sigmoid(output[0]);
        std::cout << "  Output = " << output[0] << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Forward pass
    double forward(const std::vector<std::vector<double>>& input) {
        std::cout << "\nForward Pass:\n";
        convolve(input);
        max_pool();
        fully_connected();
        std::cout << "Forward pass complete.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return output[0];
    }

    // Backward pass
    void backward(const std::vector<std::vector<double>>& input, double target) {
        std::cout << "\nBackward Pass:\n";

        // Output layer error
        double output_error = (target - output[0]) * sigmoid_derivative(output[0]);
        std::cout << "Step 1: Output Error = " << output_error << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Hidden layer error
        std::vector<double> hidden_error(hidden_size);
        for (int h = 0; h < hidden_size; ++h) {
            hidden_error[h] = output_error * Wout[h] * sigmoid_derivative(hidden[h]);
            std::cout << "  Hidden error[" << h << "] = " << hidden_error[h] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Fully connected layer gradients
        int pooled_flat_size = pooled.size() * pooled[0].size();
        std::vector<double> pooled_flat(pooled_flat_size);
        for (int i = 0; i < pooled.size(); ++i) {
            for (int j = 0; j < pooled[0].size(); ++j) {
                pooled_flat[i * pooled[0].size() + j] = pooled[i][j];
            }
        }
        std::cout << "Step 2: Updating Fully Connected Weights\n";
        for (int h = 0; h < hidden_size; ++h) {
            for (int i = 0; i < pooled_flat_size; ++i) {
                Wfc[h * pooled_flat_size + i] += learning_rate * hidden_error[h] * pooled_flat[i];
            }
            Wout[h] += learning_rate * output_error * hidden[h];
        }
        bfc += learning_rate * hidden_error[0]; // Simplified
        bout += learning_rate * output_error;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        // Pooling layer error propagation
        std::vector<std::vector<double>> pooled_error(pooled.size(), std::vector<double>(pooled[0].size(), 0.0));
        for (int h = 0; h < hidden_size; ++h) {
            for (int i = 0; i < pooled_flat_size; ++i) {
                pooled_error[i / pooled[0].size()][i % pooled[0].size()] += hidden_error[h] * Wfc[h * pooled_flat_size + i];
            }
        }

        // Feature map error (undo pooling)
        std::vector<std::vector<double>> feature_error(feature_map.size(), std::vector<double>(feature_map[0].size(), 0.0));
        std::cout << "Step 3: Propagating Error to Feature Map\n";
        for (int i = 0; i < pooled.size(); ++i) {
            for (int j = 0; j < pooled[0].size(); ++j) {
                double max_val = feature_map[i * pool_size][j * pool_size];
                int max_i = 0, max_j = 0;
                for (int pi = 0; pi < pool_size; ++pi) {
                    for (int pj = 0; pj < pool_size; ++pj) {
                        if (feature_map[i * pool_size + pi][j * pool_size + pj] > max_val) {
                            max_val = feature_map[i * pool_size + pi][j * pool_size + pj];
                            max_i = pi;
                            max_j = pj;
                        }
                    }
                }
                feature_error[i * pool_size + max_i][j * pool_size + max_j] = pooled_error[i][j];
                std::cout << "  Feature error[" << i * pool_size + max_i << "][" << j * pool_size + max_j << "] = " 
                          << feature_error[i * pool_size + max_i][j * pool_size + max_j] << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // Filter gradients
        std::cout << "Step 4: Updating Filter Weights\n";
        for (int fi = 0; fi < filter_size; ++fi) {
            for (int fj = 0; fj < filter_size; ++fj) {
                double filter_grad = 0.0;
                for (int i = 0; i < feature_map.size(); ++i) {
                    for (int j = 0; j < feature_map[0].size(); ++j) {
                        filter_grad += input[i + fi][j + fj] * feature_error[i][j];
                    }
                }
                filter[fi][fj] += learning_rate * filter_grad;
                std::cout << "  Filter[" << fi << "][" << fj << "] += " << learning_rate * filter_grad << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        std::cout << "Backward pass complete.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Train the network
    void train(const std::vector<std::vector<std::vector<double>>>& inputs, 
               const std::vector<double>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "\nEpoch " << epoch << ":\n";
            double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << "  Training on input " << i << "\n";
                double prediction = forward(inputs[i]);
                backward(inputs[i], targets[i]);
                total_error += pow(targets[i] - prediction, 2);
            }
            std::cout << "Epoch " << epoch << ", Average Error: " << total_error / inputs.size() << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    // Predict
    double predict(const std::vector<std::vector<double>>& input) {
        return forward(input);
    }
};

// Function to display an image (for visualization)
void display_image(const std::vector<std::vector<double>>& img) {
    for (const auto& row : img) {
        for (double val : row) {
            std::cout << (val > 0.5 ? "X " : ". ");
        }
        std::cout << "\n";
    }
}

int main() {
    // Define training dataset: 8x8 images (X and O patterns)
    std::vector<std::vector<std::vector<double>>> inputs = {
        { // Pattern 1: "X" shape (target: 1)
            {1, 0, 0, 0, 0, 0, 0, 1},
            {0, 1, 0, 0, 0, 0, 1, 0},
            {0, 0, 1, 0, 0, 1, 0, 0},
            {0, 0, 0, 1, 1, 0, 0, 0},
            {0, 0, 0, 1, 1, 0, 0, 0},
            {0, 0, 1, 0, 0, 1, 0, 0},
            {0, 1, 0, 0, 0, 0, 1, 0},
            {1, 0, 0, 0, 0, 0, 0, 1}
        },
        { // Pattern 2: "O" shape (target: 0)
            {0, 0, 1, 1, 1, 1, 0, 0},
            {0, 1, 0, 0, 0, 0, 1, 0},
            {1, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 1},
            {0, 1, 0, 0, 0, 0, 1, 0},
            {0, 0, 1, 1, 1, 1, 0, 0}
        }
    };
    std::vector<double> targets = {1.0, 0.0};

    // Display training images
    std::cout << "Training Image 0 (X, Target: 1):\n";
    display_image(inputs[0]);
    std::cout << "\nTraining Image 1 (O, Target: 0):\n";
    display_image(inputs[1]);
    std::cout << "\n";

    // Create and train CNN
    CNN cnn(8, 3, 2, 4, 1, 0.1); // 8x8 input, 3x3 filter, 2x2 pool, 4 hidden, 1 output
    cnn.train(inputs, targets, 10); // Train for 10 epochs

    // Test the network
    std::cout << "\nTesting the network:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        double prediction = cnn.predict(inputs[i]);
        std::cout << "Input " << i << " -> Predicted: " << prediction 
                  << " (" << (prediction > 0.5 ? 1 : 0) << "), Target: " << targets[i] << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Test on a new "image" (user-defined or hardcoded)
    std::vector<std::vector<double>> test_image = {
        {1, 0, 0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0, 1, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 1, 0},
        {1, 0, 0, 0, 0, 0, 0, 1}
    };
    std::cout << "\nTest Image (X shape):\n";
    display_image(test_image);
    double test_prediction = cnn.predict(test_image);
    std::cout << "Test Prediction: " << test_prediction << " (" << (test_prediction > 0.5 ? 1 : 0) << ")\n";

    return 0;
}