# HARAYA Neural Network Playground

## Overview

The Neural Network Playground is an interactive component of the HARAYA ML Algorithm Explorer that allows users to train and experiment with cultural intelligence models using TensorFlow.js. This component specifically focuses on predicting trustworthiness scores based on Filipino cultural factors.

## Features

### 🧠 Interactive Neural Network Training
- **Real-time training visualization** with loss/accuracy curves
- **Configurable network architecture** (adjustable hidden layer sizes)
- **Live training progress** with epoch-by-epoch updates
- **Hyperparameter tuning** (learning rate, batch size)

### 🇵🇭 Filipino Cultural Intelligence Model
- **Input Features (7):**
  - `kapwa_score` - Shared identity and interconnectedness
  - `bayanihan_participation` - Community cooperation level
  - `utang_na_loob` - Debt of gratitude integrity
  - `community_standing` - Local reputation score
  - `income` - Economic status (normalized)
  - `family_size` - Household responsibility level
  - `digital_literacy` - Technology adoption rate

- **Output:** Trustworthiness prediction (0-1 scale)

### 📊 Advanced Analytics
- **Cultural factor importance analysis** using gradient-based methods
- **Overfitting detection** with validation monitoring
- **Regional cultural variation** consideration
- **Scammer pattern recognition** (5% synthetic adversarial examples)

### 🎯 Model Architecture
- **Default Architecture:** 7 → 64 → 32 → 1
- **Activation Functions:** ReLU (hidden layers), Sigmoid (output)
- **Regularization:** Dropout layers (20% and 10%)
- **Loss Function:** Binary cross-entropy
- **Optimizer:** Adam with configurable learning rate

## Technical Implementation

### Dataset Generation
The component generates synthetic Filipino cultural personas with:
- **1,000 samples** (800 training, 200 validation)
- **Regional variations** across Luzon, Visayas, and Mindanao
- **Authentic cultural patterns** based on Filipino cultural research
- **Scammer profiles** with inconsistent cultural indicators

### Training Process
1. **Data Preprocessing:** Feature normalization and tensor conversion
2. **Model Construction:** Dynamic architecture based on user inputs
3. **Training Loop:** Custom loop with real-time chart updates
4. **Evaluation:** Performance metrics and cultural insights
5. **Analysis:** Factor importance and model interpretability

### Performance Optimization
- **GPU Acceleration:** WebGL backend when available
- **Memory Management:** Proper tensor disposal
- **Batch Processing:** Configurable batch sizes
- **Early Stopping:** Overfitting prevention

## Usage Instructions

### 1. Accessing the Neural Network Playground
```javascript
// Navigate to the Neural Network tab in the ML Algorithm Explorer
// or directly access via URL parameter
```

### 2. Configuring the Model
- Adjust hidden layer sizes (16-256 neurons each)
- Set learning rate (0.001-0.1)
- Choose batch size (16, 32, 64, 128)

### 3. Training the Model
```javascript
// Click "Train Cultural Intelligence Model"
// Monitor real-time training progress
// View loss/accuracy curves
// Analyze cultural factor importance
```

### 4. Interpreting Results
- **Training Accuracy:** Model performance on training data
- **Validation Accuracy:** Model performance on unseen data
- **Cultural Insights:** Which Filipino cultural factors are most predictive
- **Overfitting Detection:** Validation loss trending

## Cultural Intelligence Insights

### Key Findings
Based on the model's analysis of Filipino cultural patterns:

1. **Kapwa Network Strength** - Often the most predictive factor
2. **Bayanihan Participation** - Strong indicator of community trustworthiness
3. **Utang na Loob Integrity** - Character integrity through gratitude fulfillment
4. **Community Standing** - Social validation importance
5. **Economic Factors** - Secondary but relevant considerations

### Regional Variations
- **Rural Areas:** Stronger traditional cultural values
- **Urban Areas:** Balanced cultural-modern factors
- **Metro Manila:** Higher digital literacy, potentially diluted traditional values

## Export and Integration

### Model Export
```javascript
// Models can be exported in TensorFlow.js format
// Includes architecture, weights, and training history
// Compatible with production deployment systems
```

### API Integration
The trained models can be integrated into the main HARAYA platform for:
- Real-time trustworthiness scoring
- Cultural authenticity verification
- Scam detection systems
- Regional adaptation algorithms

## Educational Value

### Learning Objectives
- Understanding neural network architecture design
- Filipino cultural factor analysis
- Machine learning model training process
- Overfitting detection and prevention
- Cultural bias in AI systems

### Interactive Elements
- Real-time visualization of training process
- Hyperparameter experimentation
- Cultural factor importance analysis
- Scammer vs. legitimate profile comparison

## Future Enhancements

### Planned Features
- **Advanced Architectures:** LSTM, Attention mechanisms
- **Transfer Learning:** Pre-trained cultural models
- **Ensemble Methods:** Multiple model combination
- **Adversarial Training:** Improved scam detection
- **Cross-validation:** Robust performance estimation

### Research Opportunities
- Comparative analysis across Southeast Asian cultures
- Temporal evolution of cultural patterns
- Multi-modal learning (text, behavior, social network)
- Fairness and bias mitigation in cultural AI

## Technical Requirements

### Browser Support
- Modern browsers with WebGL support
- TensorFlow.js 4.10.0+
- Chart.js for visualization
- Sufficient memory for model training (>1GB recommended)

### Performance Considerations
- Training time: 2-5 minutes for 100 epochs
- Memory usage: ~200MB during training
- GPU acceleration recommended for optimal performance

## Contributing

### Code Structure
```
neural-network.js
├── Initialization functions
├── Dataset generation
├── Model architecture
├── Training loop
├── Visualization updates
├── Cultural analysis
└── Export functionality
```

### Development Guidelines
- Follow Filipino cultural research standards
- Maintain educational value
- Ensure real-time performance
- Include comprehensive logging
- Test across different browsers

## Support and Documentation

For technical support or questions about the Neural Network Playground:
- Check browser console for detailed logs
- Verify TensorFlow.js compatibility
- Ensure stable internet connection for library loading
- Monitor memory usage during training

The Neural Network Playground represents a cutting-edge approach to combining machine learning with Filipino cultural intelligence, providing both educational value and practical applications for the HARAYA platform.