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

class RNN {
private:
    int input_size;                         // Size of input (e.g., 1 for scalar input)
    int hidden_size;                        // Number of hidden neurons
    int output_size;                        // Size of output (e.g., 1 for scalar output)
    double learning_rate;

    std::vector<double> hidden;             // Current hidden state
    std::vector<std::vector<double>> hiddens; // Hidden states over time (for BPTT)
    std::vector<double> output;             // Output layer

    // Weight matrices
    std::vector<std::vector<double>> Wxh;   // Input -> Hidden weights
    std::vector<std::vector<double>> Whh;   // Hidden -> Hidden (recurrent) weights
    std::vector<std::vector<double>> Why;   // Hidden -> Output weights

    // Biases
    std::vector<double> bh;                 // Hidden bias
    std::vector<double> by;                 // Output bias

public:
    RNN(int in_size, int hid_size, int out_size, double lr = 0.1)
        : input_size(in_size), hidden_size(hid_size), output_size(out_size), learning_rate(lr) {
        hidden.resize(hidden_size, 0.0);
        output.resize(output_size);
        bh.resize(hidden_size);
        by.resize(output_size);

        // Initialize weights
        srand(static_cast<unsigned>(time(0)));
        Wxh.resize(input_size, std::vector<double>(hidden_size));
        Whh.resize(hidden_size, std::vector<double>(hidden_size));
        Why.resize(hidden_size, std::vector<double>(output_size));

        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                Wxh[i][j] = random(-1.0, 1.0);
            }
        }
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                Whh[i][j] = random(-1.0, 1.0);
            }
            bh[i] = random(-1.0, 1.0);
        }
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                Why[i][j] = random(-1.0, 1.0);
            }
            by[i] = random(-1.0, 1.0);
        }
    }

    // Forward pass for one timestep
    void forward_step(double input, int t) {
        std::cout << "  Timestep " << t << ": Forward Pass\n";
        std::cout << "    Input: " << input << "\n";

        // Compute new hidden state
        std::vector<double> new_hidden(hidden_size, 0.0);
        for (int j = 0; j < hidden_size; ++j) {
            new_hidden[j] = bh[j];
            new_hidden[j] += Wxh[0][j] * input; // Single input
            for (int i = 0; i < hidden_size; ++i) {
                new_hidden[j] += Whh[i][j] * hidden[i]; // Recurrent connection
            }
            new_hidden[j] = sigmoid(new_hidden[j]);
            std::cout << "    Hidden[" << j << "] = " << new_hidden[j] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        hidden = new_hidden;
        if (t >= hiddens.size()) hiddens.push_back(hidden);
        else hiddens[t] = hidden;

        // Compute output
        for (int j = 0; j < output_size; ++j) {
            output[j] = by[j];
            for (int i = 0; i < hidden_size; ++i) {
                output[j] += Why[i][j] * hidden[i];
            }
            output[j] = sigmoid(output[j]);
            std::cout << "    Output[" << j << "] = " << output[j] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    // Forward pass over a sequence
    std::vector<double> forward(const std::vector<double>& sequence) {
        std::cout << "Step 1: Forward Propagation over Sequence\n";
        hidden.assign(hidden_size, 0.0); // Reset hidden state
        hiddens.clear();
        for (int t = 0; t < sequence.size(); ++t) {
            forward_step(sequence[t], t);
        }
        std::cout << "Forward pass complete.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return output;
    }

    // Backward pass (BPTT)
    void backward(const std::vector<double>& sequence, const std::vector<double>& targets) {
        std::cout << "Step 2: Backward Propagation Through Time\n";

        // Store gradients
        std::vector<std::vector<double>> dWxh = Wxh;
        std::vector<std::vector<double>> dWhh = Whh;
        std::vector<std::vector<double>> dWhy = Why;
        std::vector<double> dbh(hidden_size, 0.0);
        std::vector<double> dby(output_size, 0.0);
        std::vector<double> dh_next(hidden_size, 0.0);

        // Compute output error
        std::vector<double> output_error(output_size);
        for (int j = 0; j < output_size; ++j) {
            output_error[j] = (targets.back() - output[j]) * sigmoid_derivative(output[j]);
            std::cout << "  Output error[" << j << "] = " << output_error[j] << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        // Backpropagate through time (from last timestep to first)
        for (int t = sequence.size() - 1; t >= 0; --t) {
            std::cout << "  Timestep " << t << ":\n";

            // Hidden -> Output gradients
            for (int i = 0; i < hidden_size; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    if (t == sequence.size() - 1) { // Only update at last timestep
                        dWhy[i][j] += output_error[j] * hiddens[t][i];
                        dby[j] += output_error[j];
                    }
                }
            }

            // Hidden layer error
            std::vector<double> hidden_error(hidden_size, 0.0);
            for (int i = 0; i < hidden_size; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    hidden_error[i] += output_error[j] * Why[i][j];
                }
                for (int j = 0; j < hidden_size; ++j) {
                    hidden_error[i] += dh_next[j] * Whh[i][j];
                }
                hidden_error[i] *= sigmoid_derivative(hiddens[t][i]);
                std::cout << "    Hidden error[" << i << "] = " << hidden_error[i] << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }

            // Input -> Hidden and Hidden -> Hidden gradients
            for (int i = 0; i < input_size; ++i) {
                for (int j = 0; j < hidden_size; ++j) {
                    dWxh[i][j] += hidden_error[j] * sequence[t];
                }
            }
            if (t > 0) {
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < hidden_size; ++j) {
                        dWhh[i][j] += hidden_error[j] * hiddens[t - 1][i];
                    }
                }
            }
            for (int j = 0; j < hidden_size; ++j) {
                dbh[j] += hidden_error[j];
            }

            dh_next = hidden_error; // Pass error back in time
        }

        // Update weights and biases
        std::cout << "  Updating weights...\n";
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                Wxh[i][j] += learning_rate * dWxh[i][j];
            }
        }
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                Whh[i][j] += learning_rate * dWhh[i][j];
            }
            bh[i] += learning_rate * dbh[i];
        }
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                Why[i][j] += learning_rate * dWhy[i][j];
            }
            by[i] += learning_rate * dby[i];
        }
        std::cout << "Backward pass complete.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Train the network
    void train(const std::vector<double>& sequence, const std::vector<double>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "\nEpoch " << epoch << ":\n";
            std::vector<double> prediction = forward(sequence);
            double error = pow(targets.back() - prediction[0], 2);
            backward(sequence, targets);
            if (epoch % 2 == 0 || epoch == epochs - 1) {
                std::cout << "Error: " << error << ", Prediction: " << prediction[0] << "\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    // Predict the next value
    double predict(const std::vector<double>& sequence) {
        forward(sequence);
        return output[0];
    }
};

int main() {
    // Sequence: [0.1, 0.2, 0.3], Target: [0.4]
    std::vector<double> sequence = {0.1, 0.2, 0.3};
    std::vector<double> targets = {0.4};

    // Create RNN: 1 input, 4 hidden, 1 output
    RNN rnn(1, 4, 1, 0.1);
    rnn.train(sequence, targets, 10); // Train for 10 epochs for demo

    // Test prediction
    std::cout << "\nTesting prediction:\n";
    double prediction = rnn.predict(sequence);
    std::cout << "Sequence: [0.1, 0.2, 0.3] -> Predicted next: " << prediction << "\n";

    return 0;
}