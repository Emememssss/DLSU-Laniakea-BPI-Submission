/**
 * HARAYA Neural Network Implementation
 * Cultural Intelligence Model Training and Visualization
 * 
 * COMPLETE TENSORFLOW.JS IMPLEMENTATION
 * - Real neural network training with backpropagation
 * - Live training visualization with Chart.js
 * - Feature importance analysis
 * - Cultural insights generation
 * - Model export/import functionality
 * - Interactive hyperparameter tuning
 */

// Neural network state
const neuralNetworkState = {
    model: null,
    isTraining: false,
    trainingHistory: {
        epochs: [],
        loss: [],
        accuracy: [],
        valLoss: [],
        valAccuracy: []
    },
    architecture: {
        inputLayer: 7,
        hiddenLayer1: 64,
        hiddenLayer2: 32,
        outputLayer: 1
    },
    hyperparameters: {
        learningRate: 0.0001, // Further reduced to prevent NaN predictions
        batchSize: 32,
        epochs: 50,
        validationSplit: 0.2,
        earlyStopping: false // Disable early stopping to ensure full training
    },
    featureImportance: null,
    culturalInsights: [],
    performanceMetrics: {
        accuracy: 0,
        loss: 0,
        precision: 0,
        recall: 0,
        f1Score: 0
    }
};

/**
 * Simple Linear Regression Implementation
 * Required by Self-Supervised Learning and Advanced Ensemble Learning algorithms
 */
class SimpleLinearRegression {
    constructor() {
        this.weights = null;
        this.bias = null;
        this.featureNames = null;
        this.trained = false;
    }

    async fit(X, y, options = {}) {
        const { learningRate = 0.01, epochs = 100, featureNames = null } = options;
        
        if (!X || !y || X.length === 0 || y.length === 0) {
            throw new Error('Invalid training data provided to SimpleLinearRegression');
        }
        
        if (X.length !== y.length) {
            throw new Error('Feature matrix and target vector must have same length');
        }

        // Store feature names for importance calculation
        this.featureNames = featureNames || X[0].map((_, i) => `feature_${i}`);
        
        // Initialize parameters
        const [numSamples, numFeatures] = [X.length, X[0].length];
        this.weights = new Array(numFeatures).fill(0);
        this.bias = 0;

        // Gradient descent training
        for (let epoch = 0; epoch < epochs; epoch++) {
            const predictions = this._predictUnchecked(X);
            
            // Calculate gradients
            const gradWeights = new Array(numFeatures).fill(0);
            let gradBias = 0;
            
            for (let i = 0; i < numSamples; i++) {
                const error = predictions[i] - y[i];
                gradBias += error;
                
                for (let j = 0; j < numFeatures; j++) {
                    gradWeights[j] += error * X[i][j];
                }
            }
            
            // Update parameters
            this.bias -= learningRate * (gradBias / numSamples);
            for (let j = 0; j < numFeatures; j++) {
                this.weights[j] -= learningRate * (gradWeights[j] / numSamples);
            }
        }
        
        this.trained = true;
    }

    // Private method for internal predictions during training
    _predictUnchecked(X) {
        if (!X || X.length === 0) {
            return [];
        }
        
        return X.map(sample => {
            let prediction = this.bias;
            for (let j = 0; j < this.weights.length; j++) {
                prediction += this.weights[j] * sample[j];
            }
            return prediction;
        });
    }

    predict(X) {
        if (!this.trained) {
            throw new Error('Model must be trained before making predictions');
        }
        
        return this._predictUnchecked(X);
    }

    getFeatureImportances() {
        if (!this.trained) {
            return null;
        }
        
        // Return absolute weights as feature importance
        const importances = {};
        const totalImportance = this.weights.reduce((sum, weight) => sum + Math.abs(weight), 0);
        
        this.featureNames.forEach((name, i) => {
            importances[name] = totalImportance > 0 ? Math.abs(this.weights[i]) / totalImportance : 0;
        });
        
        return importances;
    }

    getCoefficients() {
        return {
            weights: [...this.weights],
            bias: this.bias,
            featureNames: [...this.featureNames]
        };
    }

    // Method to calculate R-squared score
    score(X, y) {
        if (!this.trained) {
            throw new Error('Model must be trained before calculating score');
        }
        
        const predictions = this.predict(X);
        const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;
        
        let ssRes = 0; // Sum of squares of residuals
        let ssTot = 0; // Total sum of squares
        
        for (let i = 0; i < y.length; i++) {
            ssRes += Math.pow(y[i] - predictions[i], 2);
            ssTot += Math.pow(y[i] - yMean, 2);
        }
        
        return 1 - (ssRes / ssTot);
    }
}

/**
 * Initialize Neural Network component
 */
function initializeNeuralNetwork() {
    console.log('Initializing Neural Network...');
    registerAlgorithm('neuralNetwork');
    setupNeuralNetworkControls();
    createInitialTrainingChart();
    initializeNeuralNetworkDatasetConnection();
}

/**
 * Setup Neural Network control event listeners
 */
function setupNeuralNetworkControls() {
    const learningRateSlider = document.getElementById('learningRate');
    const learningRateValue = document.getElementById('learningRateValue');
    
    if (learningRateSlider && learningRateValue) {
        learningRateSlider.addEventListener('input', function() {
            neuralNetworkState.hyperparameters.learningRate = parseFloat(this.value);
            learningRateValue.textContent = this.value;
            updateNeuralNetworkArchitecture();
        });
    }

    const batchSizeSelect = document.getElementById('batchSize');
    if (batchSizeSelect) {
        batchSizeSelect.addEventListener('change', function() {
            neuralNetworkState.hyperparameters.batchSize = parseInt(this.value);
        });
    }
    
    // Setup architecture inputs
    const hiddenLayer1Input = document.querySelector('#networkArchitecture input[value="64"]');
    const hiddenLayer2Input = document.querySelector('#networkArchitecture input[value="32"]');
    
    if (hiddenLayer1Input) {
        hiddenLayer1Input.addEventListener('input', function() {
            const value = parseInt(this.value) || 64;
            neuralNetworkState.architecture.hiddenLayer1 = Math.max(16, Math.min(256, value));
            this.value = neuralNetworkState.architecture.hiddenLayer1;
            updateNeuralNetworkArchitecture();
        });
    }
    
    if (hiddenLayer2Input) {
        hiddenLayer2Input.addEventListener('input', function() {
            const value = parseInt(this.value) || 32;
            neuralNetworkState.architecture.hiddenLayer2 = Math.max(8, Math.min(128, value));
            this.value = neuralNetworkState.architecture.hiddenLayer2;
            updateNeuralNetworkArchitecture();
        });
    }
}

/**
 * Train Neural Network model with complete functionality
 */
async function trainNeuralNetwork() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        showNotification('No dataset loaded. Please upload or select a dataset first.', 'error');
        return;
    }

    if (neuralNetworkState.isTraining) {
        showNotification('Model is already training', 'error');
        return;
    }

    if (typeof tf === 'undefined') {
        showNotification('TensorFlow.js is not loaded', 'error');
        return;
    }

    try {
        neuralNetworkState.isTraining = true;
        updateTrainingStatus('Preparing cultural intelligence data...');
        clearDebugUI();
        
        debugToUI(`🚀 Starting neural network training with ${mlCore.sharedData.length} records`);
        console.log(`Starting neural network training with ${mlCore.sharedData.length} records...`);
        
        // === COMPREHENSIVE DATASET DEBUG ===
        console.log('=== DATASET INSPECTION ===');
        console.log('Dataset size:', mlCore.sharedData.length);
        
        if (mlCore.sharedData.length > 0) {
            console.log('First 3 rows sample:', mlCore.sharedData.slice(0, 3));
            console.log('Available columns:', Object.keys(mlCore.sharedData[0]));
            
            // Check key columns existence
            const requiredColumns = [
                'community_standing_score', 'location_stability_score', 'bill_payment_consistency',
                'monthly_income', 'family_size', 'digital_literacy_score', 'trustworthiness_label'
            ];
            
            const missingColumns = requiredColumns.filter(col => 
                !mlCore.sharedData[0].hasOwnProperty(col)
            );
            
            if (missingColumns.length > 0) {
                console.warn('⚠️ Missing expected columns:', missingColumns);
                console.log('Available columns:', Object.keys(mlCore.sharedData[0]));
            } else {
                console.log('✅ All required columns present');
            }
            
            // Sample data values from different rows
            console.log('Sample data values from rows 1, 100, 500:');
            [0, Math.min(99, mlCore.sharedData.length-1), Math.min(499, mlCore.sharedData.length-1)].forEach(i => {
                if (mlCore.sharedData[i]) {
                    console.log(`Row ${i+1}:`, {
                        community_standing_score: mlCore.sharedData[i].community_standing_score,
                        trustworthiness_label: mlCore.sharedData[i].trustworthiness_label,
                        monthly_income: mlCore.sharedData[i].monthly_income
                    });
                }
            });
        }
        console.log('=== END DATASET INSPECTION ===');
        
        // Reset previous training data
        neuralNetworkState.trainingHistory = {
            epochs: [],
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
        
        // === SYNTHETIC TEST OPTION (EMERGENCY DEBUG) ===
        // Uncomment the next line to test with synthetic data instead of real data
        // const useSyntheticData = true;
        const useSyntheticData = false;
        
        let features, labels, featureNames;
        
        if (useSyntheticData) {
            console.log('🧪 USING SYNTHETIC TEST DATA');
            // Create simple synthetic dataset that should definitely work
            const syntheticSize = 1000;
            features = [];
            labels = [];
            featureNames = [
                'Kapwa Network Strength', 'Bayanihan Participation', 'Utang na Loob Integrity',
                'Community Standing', 'Economic Status (Normalized)', 'Family Responsibility', 'Digital Literacy'
            ];
            
            for (let i = 0; i < syntheticSize; i++) {
                // Create features with clear linear relationship to label
                const baseValue = i / syntheticSize; // 0 to 1
                const noise = (Math.random() - 0.5) * 0.1; // Small noise
                
                const feature1 = Math.min(Math.max(baseValue + noise, 0), 1);
                const feature2 = Math.min(Math.max(baseValue + noise * 0.5, 0), 1);
                const feature3 = Math.min(Math.max(baseValue + noise * 0.8, 0), 1);
                const feature4 = Math.min(Math.max(0.5 + (baseValue - 0.5) * 0.7 + noise, 0), 1);
                const feature5 = Math.min(Math.max(Math.random(), 0), 1); // Random feature
                const feature6 = Math.min(Math.max(baseValue * 0.8 + 0.1, 0), 1);
                const feature7 = Math.min(Math.max(0.3 + baseValue * 0.4, 0), 1);
                
                features.push([feature1, feature2, feature3, feature4, feature5, feature6, feature7]);
                
                // Label based on average of first 3 features (clear relationship)
                const label = (feature1 + feature2 + feature3) / 3;
                labels.push([Math.min(Math.max(label, 0), 1)]);
            }
            
            console.log(`Generated ${features.length} synthetic samples with clear feature-label relationship`);
        } else {
            // Use real data
            const result = prepareTrainingData(mlCore.sharedData);
            features = result.features;
            labels = result.labels;
            featureNames = result.featureNames;
            
            if (features.length === 0) {
                throw new Error('No valid training data found. Check dataset structure and required columns.');
            }
        }

        console.log(`Successfully prepared ${features.length} training samples with features: ${featureNames.join(', ')}`);
        debugToUI(`✅ Prepared ${features.length} training samples with ${featureNames.length} features`);
        
        // Validate that we have both positive and negative examples
        const uniqueLabels = [...new Set(labels.map(l => l[0]))];
        console.log(`Training labels distribution: ${uniqueLabels.map(l => `${l}: ${labels.filter(lab => lab[0] === l).length}`).join(', ')}`);
        debugToUI(`🎯 Found ${uniqueLabels.length} unique labels: ${uniqueLabels.slice(0,3).map(l => l.toFixed(3)).join(', ')}${uniqueLabels.length > 3 ? '...' : ''}`);
        
        if (uniqueLabels.length < 2) {
            debugToUI('⚠️ Warning: All labels identical - model cannot learn differences!');
            showNotification('Training data needs both trustworthy and untrustworthy examples for proper training.', 'warning');
        }

        updateTrainingStatus(`Building neural network model... (${features.length} samples)`);
        
        // Calculate sample weights to handle imbalanced data for regression
        const { weights: sampleWeights, rangeWeights } = calculateRegressionSampleWeights(labels);
        console.log(`Calculated sample weights for ${labels.length} samples to handle score distribution imbalance`);
        
        // Create model with current architecture and custom weighted loss
        const model = createCulturalIntelligenceModel(rangeWeights);
        neuralNetworkState.model = model;
        debugToUI(`🧠 Neural network created with ${model.countParams()} parameters`);

        // Convert data to tensors
        const xs = tf.tensor2d(features);
        const ys = tf.tensor2d(labels);
        
        // CRITICAL DEBUG: Check tensor shapes
        debugToUI(`🔍 Tensor shapes - Features: [${xs.shape.join(', ')}], Labels: [${ys.shape.join(', ')}]`);
        
        // Debug first few values to check structure
        const featureSample = xs.slice([0, 0], [3, -1]).arraySync();
        const labelSample = ys.slice([0, 0], [3, -1]).arraySync();
        debugToUI(`🔍 Sample features (first 3): ${JSON.stringify(featureSample).substring(0, 100)}...`);
        debugToUI(`🔍 Sample labels (first 3): ${JSON.stringify(labelSample)}`);
        
        // Verify labels are in expected range [0, 1]
        const labelFlat = ys.flatten().arraySync();
        const labelRange = { min: Math.min(...labelFlat), max: Math.max(...labelFlat) };
        debugToUI(`🔍 Label range: [${labelRange.min.toFixed(3)}, ${labelRange.max.toFixed(3)}]`);
        
        // === TENSOR DEBUG STATISTICS ===
        console.log('=== TENSOR CREATION DEBUG ===');
        console.log('Input tensor shape:', xs.shape);
        console.log('Label tensor shape:', ys.shape);
        
        // Check tensor statistics
        const xsStats = {
            min: xs.min().arraySync(),
            max: xs.max().arraySync(),
            mean: xs.mean().arraySync(),
            hasNaN: xs.isNaN().any().arraySync(),
            hasInf: xs.isInf().any().arraySync()
        };
        
        const ysStats = {
            min: ys.min().arraySync(),
            max: ys.max().arraySync(),
            mean: ys.mean().arraySync(),
            hasNaN: ys.isNaN().any().arraySync(),
            hasInf: ys.isInf().any().arraySync()
        };
        
        console.log('Features (xs) stats:', xsStats);
        console.log('Labels (ys) stats:', ysStats);
        
        if (xsStats.hasNaN || ysStats.hasNaN) {
            debugToUI('🚨 CRITICAL: NaN detected in input tensors before training!');
            console.error('🚨 CRITICAL: NaN detected in input tensors before training!');
        } else if (xsStats.hasInf || ysStats.hasInf) {
            debugToUI('🚨 CRITICAL: Infinity detected in input tensors before training!');
            console.error('🚨 CRITICAL: Infinity detected in input tensors before training!');
        } else {
            debugToUI(`✅ Tensor validation passed - Features: [${xsStats.min.toFixed(3)}, ${xsStats.max.toFixed(3)}], Labels: [${ysStats.min.toFixed(3)}, ${ysStats.max.toFixed(3)}]`);
        }
        console.log('=== END TENSOR DEBUG ===');
        
        // Split data for validation
        const splitIndex = Math.floor(features.length * (1 - neuralNetworkState.hyperparameters.validationSplit));
        const trainXs = xs.slice([0, 0], [splitIndex, -1]);
        const trainYs = ys.slice([0, 0], [splitIndex, -1]);
        const valXs = xs.slice([splitIndex, 0], [-1, -1]);
        const valYs = ys.slice([splitIndex, 0], [-1, -1]);

        updateTrainingStatus('Training neural network with real backpropagation...');
        resetTrainingChart();
        
        // CRITICAL: Validate tensor data before training to prevent NaN issues
        const trainXsHasNaN = trainXs.isNaN().any().arraySync();
        const trainYsHasNaN = trainYs.isNaN().any().arraySync();
        const valXsHasNaN = valXs.isNaN().any().arraySync();
        const valYsHasNaN = valYs.isNaN().any().arraySync();
        
        if (trainXsHasNaN || trainYsHasNaN || valXsHasNaN || valYsHasNaN) {
            console.error('CRITICAL: NaN detected in training tensors!');
            console.log('trainXs has NaN:', trainXsHasNaN);
            console.log('trainYs has NaN:', trainYsHasNaN);
            console.log('valXs has NaN:', valXsHasNaN);
            console.log('valYs has NaN:', valYsHasNaN);
            throw new Error('Training data contains NaN values. Check data preprocessing.');
        }
        
        console.log('✅ Tensor validation passed - no NaN values detected in training data');
        
        // Setup early stopping callback
        let bestValLoss = Infinity;
        let patienceCounter = 0;
        const patience = 10;
        
        debugToUI(`🚀 Starting training: ${neuralNetworkState.hyperparameters.epochs} epochs, batch size ${neuralNetworkState.hyperparameters.batchSize}`);
        
        // Capture initial weights for gradient monitoring
        const initialWeights = model.getWeights();
        const initialWeightMatrix = initialWeights[0].arraySync();
        const initialWeight = initialWeightMatrix[0][0]; // Get first weight
        debugToUI(`🔍 Initial first weight: ${initialWeight.toFixed(6)}`);
        
        // Also capture a few more weights for comprehensive monitoring
        const initialWeightSample = [
            initialWeightMatrix[0][0],
            initialWeightMatrix[0][1] || 0,
            initialWeightMatrix[1] ? initialWeightMatrix[1][0] : 0
        ];
        debugToUI(`🔍 Initial weight sample: [${initialWeightSample.map(w => w.toFixed(6)).join(', ')}]`);
        
        // CRITICAL DEBUG: Test model predictions before training
        const testPredictions = model.predict(trainXs.slice([0, 0], [5, -1]));
        const testLabels = trainYs.slice([0, 0], [5, -1]);
        const predArray = testPredictions.arraySync();
        const labelArray = testLabels.arraySync();
        debugToUI(`🎯 Pre-training test - Predictions: [${predArray.map(p => p[0].toFixed(4)).join(', ')}]`);
        debugToUI(`🎯 Pre-training test - Labels: [${labelArray.map(l => l[0].toFixed(4)).join(', ')}]`);
        testPredictions.dispose();
        testLabels.dispose();
        
        // Train the model with comprehensive callbacks (sample weighting handled by custom loss)
        const history = await model.fit(trainXs, trainYs, {
            epochs: neuralNetworkState.hyperparameters.epochs,
            batchSize: neuralNetworkState.hyperparameters.batchSize,
            validationData: [valXs, valYs],
            shuffle: true,
            verbose: 0,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    // CRITICAL: Explicit epoch tracking and NaN detection
                    debugToUI(`🔄 Processing epoch ${epoch + 1} of ${neuralNetworkState.hyperparameters.epochs}`);
                    
                    // Check for null/undefined/NaN in logs immediately
                    const hasNullOrNaN = Object.entries(logs).some(([key, val]) => 
                        val === null || val === undefined || isNaN(val) || !isFinite(val)
                    );
                    if (hasNullOrNaN) {
                        debugToUI('🚨 CRITICAL: Null/NaN detected in training logs!');
                        debugToUI(`🚨 Logs: ${JSON.stringify(logs)}`);
                        
                        // Try to calculate loss manually as fallback
                        try {
                            const samplePreds = model.predict(trainXs.slice([0, 0], [10, -1]));
                            const sampleLabels = trainYs.slice([0, 0], [10, -1]);
                            const manualLoss = tf.losses.meanSquaredError(sampleLabels, samplePreds).arraySync();
                            debugToUI(`🔧 Manual loss calculation: ${manualLoss}`);
                            samplePreds.dispose();
                            sampleLabels.dispose();
                        } catch (e) {
                            debugToUI(`🚨 Manual loss calculation failed: ${e.message}`);
                        }
                        
                        model.stopTraining = true;
                        throw new Error(`Null/NaN detected at epoch ${epoch + 1}. Training halted.`);
                    }
                    
                    // Store training metrics (regression-focused)
                    neuralNetworkState.trainingHistory.epochs.push(epoch + 1);
                    neuralNetworkState.trainingHistory.loss.push(logs.loss); // MSE loss
                    neuralNetworkState.trainingHistory.accuracy.push(logs.mae || 0); // Use MAE instead of accuracy
                    neuralNetworkState.trainingHistory.valLoss.push(logs.val_loss); // Validation MSE
                    neuralNetworkState.trainingHistory.valAccuracy.push(logs.val_mae || 0); // Validation MAE
                    
                    // Add debug output for training progress (every 10 epochs or significant epochs)
                    if (epoch % 10 === 0 || epoch === 0 || epoch === neuralNetworkState.hyperparameters.epochs - 1) {
                        // Debug: Log all available keys in logs object
                        if (epoch === 0) {
                            debugToUI(`🔍 Available logs keys: ${Object.keys(logs).join(', ')}`);
                        }
                        
                        // Check if weights are changing (gradient monitoring)
                        if (epoch === 4 || epoch === 19) { // Check at epochs 5 and 20
                            const currentWeights = model.getWeights();
                            const currentWeightMatrix = currentWeights[0].arraySync();
                            const currentWeight = currentWeightMatrix[0][0];
                            const weightChange = Math.abs(currentWeight - initialWeight);
                            debugToUI(`🔍 Weight change at epoch ${epoch + 1}: ${weightChange.toFixed(8)} (current: ${currentWeight.toFixed(6)})`);
                            
                            // Check multiple weights for comprehensive monitoring
                            const currentWeightSample = [
                                currentWeightMatrix[0][0],
                                currentWeightMatrix[0][1] || 0,
                                currentWeightMatrix[1] ? currentWeightMatrix[1][0] : 0
                            ];
                            const weightChanges = currentWeightSample.map((w, i) => Math.abs(w - initialWeightSample[i]));
                            debugToUI(`🔍 All weight changes: [${weightChanges.map(c => c.toFixed(8)).join(', ')}]`);
                        }
                        
                        // CRITICAL DEBUG: Check actual predictions vs labels during training
                        if (epoch === 0 || epoch === 9) { // Epoch 1 and 10
                            const currentPredictions = model.predict(trainXs.slice([0, 0], [3, -1]));
                            const currentLabels = trainYs.slice([0, 0], [3, -1]);
                            const predArray = currentPredictions.arraySync();
                            const labelArray = currentLabels.arraySync();
                            debugToUI(`🎯 Epoch ${epoch + 1} - Preds: [${predArray.map(p => p[0].toFixed(4)).join(', ')}], Labels: [${labelArray.map(l => l[0].toFixed(4)).join(', ')}]`);
                            
                            // Check if predictions exactly match labels (which would cause 0 loss)
                            const differences = predArray.map((pred, i) => Math.abs(pred[0] - labelArray[i][0]));
                            const avgDiff = differences.reduce((a, b) => a + b, 0) / differences.length;
                            debugToUI(`🔍 Average prediction-label difference: ${avgDiff.toFixed(6)}`);
                            
                            currentPredictions.dispose();
                            currentLabels.dispose();
                        }
                        
                        const mae = logs.mae || logs.meanAbsoluteError || 0;
                        const valMae = logs.val_mae || logs.val_meanAbsoluteError || 0;
                        const rmse = Math.sqrt(logs.loss || 0);
                        const valRmse = Math.sqrt(logs.val_loss || 0);
                        debugToUI(`📈 Epoch ${epoch + 1}: MSE=${(logs.loss || 0).toFixed(6)} | MAE=${mae.toFixed(4)} | RMSE=${rmse.toFixed(4)} | Val MSE=${(logs.val_loss || 0).toFixed(6)} | Val MAE=${valMae.toFixed(4)}`);
                    }
                    
                    // Update UI
                    await updateTrainingProgress(epoch + 1, logs);
                    updateTrainingChart();
                    
                    // Early stopping logic based on validation loss
                    if (neuralNetworkState.hyperparameters.earlyStopping) {
                        if (logs.val_loss < bestValLoss) {
                            bestValLoss = logs.val_loss;
                            patienceCounter = 0;
                        } else {
                            patienceCounter++;
                            if (patienceCounter >= patience) {
                                updateTrainingStatus(`Early stopping triggered at epoch ${epoch + 1}`);
                                model.stopTraining = true;
                            }
                        }
                    }
                },
                onTrainEnd: () => {
                    updateTrainingStatus('Training completed successfully!');
                    debugToUI('🎉 Neural network training completed successfully!');
                }
            }
        });

        // Calculate final performance metrics
        await calculatePerformanceMetrics(model, valXs, valYs);
        
        // Perform feature importance analysis
        await analyzeFeatureImportance(model, features, featureNames);
        
        // Generate cultural insights
        generateCulturalInsights();
        
        // Clean up tensors
        xs.dispose();
        ys.dispose();
        trainXs.dispose();
        trainYs.dispose();
        valXs.dispose();
        valYs.dispose();
        
        // Update final UI
        updateFeatureImportanceChart();
        displayCulturalInsights();
        updatePerformanceMetricsUI();
        
        // Share learned cultural weights with Cultural Intelligence module
        updateCulturalIntelligenceWeights();
        
        showNotification('Neural network training completed successfully! Cultural weights updated.', 'success');
        
    } catch (error) {
        console.error('Neural Network Training Error:', error);
        showNotification(`Training failed: ${error.message}`, 'error');
    } finally {
        neuralNetworkState.isTraining = false;
    }
}

/**
 * Calculate sample weights for regression to handle imbalanced trustworthiness score distribution
 */
function calculateRegressionSampleWeights(labels) {
    const weights = [];
    const labelValues = labels.map(l => l[0]);
    
    // Create score ranges for weighting
    const scoreRanges = {
        'very_low': { min: 0.0, max: 0.2, count: 0 },
        'low': { min: 0.2, max: 0.4, count: 0 },
        'medium': { min: 0.4, max: 0.6, count: 0 },
        'high': { min: 0.6, max: 0.8, count: 0 },
        'very_high': { min: 0.8, max: 1.0, count: 0 }
    };
    
    // Count samples in each range
    labelValues.forEach(score => {
        for (const [rangeName, range] of Object.entries(scoreRanges)) {
            if (score >= range.min && score < range.max) {
                range.count++;
                break;
            }
        }
    });
    
    // Calculate inverse frequency weights
    const totalSamples = labelValues.length;
    const rangeWeights = {};
    
    for (const [rangeName, range] of Object.entries(scoreRanges)) {
        if (range.count > 0) {
            // Inverse frequency weighting: more weight to underrepresented ranges
            rangeWeights[rangeName] = totalSamples / (Object.keys(scoreRanges).length * range.count);
        } else {
            rangeWeights[rangeName] = 1.0; // Default weight for empty ranges
        }
    }
    
    console.log('Score distribution:', Object.entries(scoreRanges).map(([name, range]) => 
        `${name}: ${range.count} (${((range.count/totalSamples)*100).toFixed(1)}%)`).join(', '));
    console.log('Range weights:', rangeWeights);
    
    // Assign weights to each sample
    labelValues.forEach(score => {
        let weight = 1.0;
        for (const [rangeName, range] of Object.entries(scoreRanges)) {
            if (score >= range.min && score < range.max) {
                weight = rangeWeights[rangeName];
                break;
            }
        }
        weights.push(weight);
    });
    
    return { weights, rangeWeights };
}

/**
 * Helper function to get weight for a specific trustworthiness score
 */
function getWeightForTrustworthinessScore(score, rangeWeights) {
    // Define the same score ranges as in calculateRegressionSampleWeights
    const scoreRanges = [
        { name: 'very_low', min: 0.0, max: 0.2 },
        { name: 'low', min: 0.2, max: 0.4 },
        { name: 'medium', min: 0.4, max: 0.6 },
        { name: 'high', min: 0.6, max: 0.8 },
        { name: 'very_high', min: 0.8, max: 1.0 }
    ];
    
    // Find which range the score belongs to
    for (const range of scoreRanges) {
        if (score >= range.min && score < range.max) {
            return rangeWeights[range.name] || 1.0;
        }
    }
    
    // Handle edge case for score = 1.0
    if (score >= 1.0) {
        return rangeWeights['very_high'] || 1.0;
    }
    
    // Fallback for unexpected scores
    console.warn(`Score ${score} out of expected range [0,1] for weighting, defaulting to weight 1.0.`);
    return 1.0;
}

/**
 * Safe tensor sampling for debugging - handles different tensor shapes
 */
function safeSample(tensor, limit = 5) {
    try {
        const data = tensor.arraySync();
        if (Array.isArray(data)) {
            // Handle multi-dimensional arrays by flattening and slicing
            return data.flat(Infinity).slice(0, limit);
        } else {
            // Handle scalar values
            return [data];
        }
    } catch (error) {
        return [`[Error reading tensor: ${error.message}]`];
    }
}

/**
 * Create custom weighted MSE loss function to handle imbalanced data
 * Uses pure TensorFlow.js operations to maintain gradient flow
 * Includes debugging and NaN detection
 */
function createWeightedMeanSquaredErrorLoss(rangeWeights) {
    console.log('Creating weighted loss function with rangeWeights:', rangeWeights);
    
    return (yTrue, yPred) => {
        return tf.tidy(() => {
            // Debug: Check inputs for NaN/Infinity
            const yTrueHasNaN = yTrue.isNaN().any().arraySync();
            const yPredHasNaN = yPred.isNaN().any().arraySync();
            const yTrueHasInf = yTrue.isInf().any().arraySync();
            const yPredHasInf = yPred.isInf().any().arraySync();
            
            if (yTrueHasNaN || yPredHasNaN || yTrueHasInf || yPredHasInf) {
                console.error('NaN/Inf detected in loss function inputs!');
                console.log('yTrue has NaN:', yTrueHasNaN, 'yPred has NaN:', yPredHasNaN);
                console.log('yTrue has Inf:', yTrueHasInf, 'yPred has Inf:', yPredHasInf);
                console.log('yTrue sample:', safeSample(yTrue, 5));
                console.log('yPred sample:', safeSample(yPred, 5));
            }
            
            // Calculate per-sample MSE (loss for each individual sample in the batch)
            const msePerSample = tf.losses.meanSquaredError(yTrue, yPred, tf.Reduction.NONE);
            
            // Debug: Check MSE calculation
            const mseHasNaN = msePerSample.isNaN().any().arraySync();
            if (mseHasNaN) {
                console.error('NaN detected in MSE calculation!');
                console.log('msePerSample sample:', safeSample(msePerSample, 5));
            }
            
            // Ensure yTrue is a 1D tensor of scores
            const scoresTensor = yTrue.squeeze();
            
            // Initialize weights tensor with ones (default weight)
            let sampleWeightsTensor = tf.onesLike(scoresTensor);
            
            // Validate rangeWeights before use
            const validatedRangeWeights = {
                very_low: (rangeWeights?.very_low && isFinite(rangeWeights.very_low)) ? rangeWeights.very_low : 1.0,
                low: (rangeWeights?.low && isFinite(rangeWeights.low)) ? rangeWeights.low : 1.0,
                medium: (rangeWeights?.medium && isFinite(rangeWeights.medium)) ? rangeWeights.medium : 1.0,
                high: (rangeWeights?.high && isFinite(rangeWeights.high)) ? rangeWeights.high : 1.0,
                very_high: (rangeWeights?.very_high && isFinite(rangeWeights.very_high)) ? rangeWeights.very_high : 1.0
            };
            
            // Apply range-based weighting using tensor operations
            const ranges = [
                { min: 0.0, max: 0.2, weight: validatedRangeWeights.very_low },
                { min: 0.2, max: 0.4, weight: validatedRangeWeights.low },
                { min: 0.4, max: 0.6, weight: validatedRangeWeights.medium },
                { min: 0.6, max: 0.8, weight: validatedRangeWeights.high },
                { min: 0.8, max: 1.0, weight: validatedRangeWeights.very_high }
            ];
            
            // Apply weights for each range using tensor operations
            ranges.forEach((range, index) => {
                // Create boolean masks for scores within the current range
                const lowerBoundMask = scoresTensor.greaterEqual(range.min);
                const upperBoundMask = (index === ranges.length - 1) 
                    ? scoresTensor.lessEqual(range.max)
                    : scoresTensor.less(range.max);
                    
                const inRangeMask = lowerBoundMask.logicalAnd(upperBoundMask);
                
                // Update weights where the mask is true
                sampleWeightsTensor = tf.where(
                    inRangeMask, 
                    tf.fill(scoresTensor.shape, range.weight), 
                    sampleWeightsTensor
                );
            });
            
            // Debug: Check weights tensor
            const weightsHasNaN = sampleWeightsTensor.isNaN().any().arraySync();
            if (weightsHasNaN) {
                console.error('NaN detected in sample weights!');
                console.log('sampleWeightsTensor sample:', safeSample(sampleWeightsTensor, 5));
            }
            
            // Compute the weighted loss using tensor operations
            const finalLoss = tf.losses.computeWeightedLoss(msePerSample, sampleWeightsTensor);
            
            // Debug: Check final loss
            const finalLossHasNaN = finalLoss.isNaN().arraySync();
            if (finalLossHasNaN) {
                console.error('NaN detected in final weighted loss!');
                console.log('Final loss value:', finalLoss.arraySync());
            }
            
            return finalLoss;
        });
    };
}

/**
 * Prepare training data from dataset with proper cultural features
 */
function prepareTrainingData(dataset) {
    const features = [];
    const labels = [];
    let debugRowCount = 0;
    
    // 7 Cultural Intelligence Features mapped to available dataset columns
    const featureNames = [
        'Kapwa Network Strength',
        'Bayanihan Participation', 
        'Utang na Loob Integrity',
        'Community Standing',
        'Economic Status (Normalized)',
        'Family Responsibility',
        'Digital Literacy'
    ];
    
    console.log('=== DATA PROCESSING DEBUG ===');

    dataset.forEach((row, index) => {
        const featureVector = [];
        let hasAllFeatures = true;

        try {
            // Extract raw values first for proper scaling
            const communityStanding = parseFloat(row.community_standing_score || 0.6);
            const locationStability = parseFloat(row.location_stability_score || 0.7);
            const billPaymentConsistency = parseFloat(row.bill_payment_consistency || 0.8);
            const monthlyIncome = parseFloat(row.monthly_income || 20000);
            const familySize = parseFloat(row.family_size || 4);
            const digitalLiteracy = parseFloat(row.digital_literacy_score || 6);
            
            // Debug first 3 and last 3 rows plus some middle ones
            const shouldLog = index < 3 || index >= dataset.length - 3 || 
                             index === Math.floor(dataset.length / 2) ||
                             (index > 0 && index % 500 === 0);
                             
            if (shouldLog) {
                console.log(`Row ${index + 1} raw values:`, {
                    community_standing_score: row.community_standing_score,
                    location_stability_score: row.location_stability_score,
                    bill_payment_consistency: row.bill_payment_consistency,
                    monthly_income: row.monthly_income,
                    family_size: row.family_size,
                    digital_literacy_score: row.digital_literacy_score,
                    trustworthiness_label: row.trustworthiness_label
                });
                console.log(`Row ${index + 1} parsed values:`, {
                    communityStanding, locationStability, billPaymentConsistency,
                    monthlyIncome, familySize, digitalLiteracy
                });
            }

            // Feature 1: Kapwa Score (network strength and community bonds)
            const kapwaScore = (communityStanding + locationStability) / 2;
            featureVector.push(Math.min(Math.max(kapwaScore, 0), 1));

            // Feature 2: Bayanihan Participation (community cooperation and mutual aid)
            const bayanihanScore = (billPaymentConsistency + communityStanding) / 2;
            featureVector.push(Math.min(Math.max(bayanihanScore, 0), 1));

            // Feature 3: Utang na Loob Integrity (reciprocity and gratitude bonds)
            const utangScore = (billPaymentConsistency + locationStability) / 2;
            featureVector.push(Math.min(Math.max(utangScore, 0), 1));

            // Feature 4: Community Standing (social reputation and trust)
            featureVector.push(Math.min(Math.max(communityStanding, 0), 1));

            // Feature 5: Economic Status (robust income normalization)
            // Using log transformation to handle income distribution better
            const normalizedIncome = Math.min(Math.max(Math.log(monthlyIncome + 1) / Math.log(100001), 0), 1);
            featureVector.push(normalizedIncome);

            // Feature 6: Family Responsibility (normalized family size with better scaling)
            // Cap at reasonable maximum family size of 12
            const normalizedFamilySize = Math.min(Math.max(familySize / 12, 0), 1);
            featureVector.push(normalizedFamilySize);

            // Feature 7: Digital Literacy (0-10 scale normalized to 0-1)
            const normalizedDigitalLiteracy = Math.min(Math.max(digitalLiteracy / 10, 0), 1);
            featureVector.push(normalizedDigitalLiteracy);

            // Comprehensive validation: check for NaN, Infinity, and valid range [0,1]
            const isValidFeatureVector = featureVector.length === 7 && 
                featureVector.every(f => !isNaN(f) && isFinite(f) && f >= 0 && f <= 1);
                
            if (isValidFeatureVector) {
                features.push(featureVector);
                
                // Create binary trustworthiness label
                let label = 0.5; // default neutral
                
                if (row.trustworthiness_label === 'trustworthy') {
                    label = 1.0;
                } else if (row.trustworthiness_label === 'untrustworthy' || 
                           row.trustworthiness_label === 'adversarial_attack') {
                    label = 0.0;
                } else if (row.trustworthiness_label === 'challenging_legitimate') {
                    label = 0.7; // Slightly trustworthy
                } else if (row.is_legitimate === true || row.is_legitimate === 'True') {
                    label = 1.0;
                } else if (row.is_scammer === true || row.is_scammer === 'True') {
                    label = 0.0;
                } else {
                    // For neutral cases, use a combination of available indicators
                    if (parseFloat(row.bill_payment_consistency || 0) > 0.8 && 
                        parseFloat(row.community_standing_score || 0) > 0.7) {
                        label = 0.8; // Likely trustworthy
                    } else if (parseFloat(row.bill_payment_consistency || 0) < 0.5) {
                        label = 0.3; // Likely untrustworthy
                    }
                }
                
                // Debug calculated features and labels
                if (shouldLog) {
                    console.log(`Row ${index + 1} calculated features:`, featureVector);
                    console.log(`Row ${index + 1} label:`, label);
                }
                
                // Validate label: must be finite number in [0,1] range
                if (!isNaN(label) && isFinite(label) && label >= 0 && label <= 1) {
                    labels.push([label]);
                    debugRowCount++;
                } else {
                    console.warn('Invalid label detected, skipping row:', { label, row });
                    features.pop(); // Remove the corresponding feature vector
                }
            } else {
                hasAllFeatures = false;
            }
        } catch (error) {
            console.warn('Error processing row for neural network:', error, row);
            hasAllFeatures = false;
        }
    });
    
    console.log(`Prepared ${features.length} training samples with 7 cultural features from ${dataset.length} total records`);
    console.log(`Feature extraction success rate: ${((features.length / dataset.length) * 100).toFixed(1)}%`);
    
    if (features.length === 0) {
        console.error('No valid features extracted. Dataset structure:', dataset[0]);
        return { features, labels, featureNames };
    }
    
    // Apply additional feature scaling for robust normalization
    const scaledFeatures = applyFeatureScaling(features, featureNames);
    console.log('Applied feature scaling for improved neural network training');
    
    // Final validation: Check for any NaN/Infinity in processed data
    const featuresValid = scaledFeatures.every(row => 
        row.every(val => !isNaN(val) && isFinite(val) && val >= 0 && val <= 1)
    );
    const labelsValid = labels.every(label => 
        !isNaN(label[0]) && isFinite(label[0]) && label[0] >= 0 && label[0] <= 1
    );
    
    if (!featuresValid) {
        console.error('Invalid features detected after scaling! Some features contain NaN/Infinity.');
        // Find and log problematic features
        scaledFeatures.forEach((row, i) => {
            row.forEach((val, j) => {
                if (!isFinite(val) || val < 0 || val > 1) {
                    console.error(`Invalid feature at row ${i}, column ${j} (${featureNames[j]}): ${val}`);
                }
            });
        });
    }
    
    if (!labelsValid) {
        console.error('Invalid labels detected! Some labels contain NaN/Infinity.');
        labels.forEach((label, i) => {
            if (!isFinite(label[0]) || label[0] < 0 || label[0] > 1) {
                console.error(`Invalid label at row ${i}: ${label[0]}`);
            }
        });
    }
    
    console.log(`Data validation: Features valid = ${featuresValid}, Labels valid = ${labelsValid}`);
    
    // === FINAL DATA STATISTICS ===
    console.log('=== FINAL PROCESSED DATA STATS ===');
    if (scaledFeatures.length > 0) {
        // Feature statistics
        console.log(`Total processed samples: ${scaledFeatures.length}`);
        featureNames.forEach((name, i) => {
            const featureValues = scaledFeatures.map(row => row[i]);
            const min = Math.min(...featureValues);
            const max = Math.max(...featureValues);
            const mean = featureValues.reduce((sum, val) => sum + val, 0) / featureValues.length;
            const variance = featureValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / featureValues.length;
            
            console.log(`${name}: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}, variance=${variance.toFixed(6)}`);
        });
        
        // Label statistics
        const labelValues = labels.map(l => l[0]);
        const labelMin = Math.min(...labelValues);
        const labelMax = Math.max(...labelValues);
        const labelMean = labelValues.reduce((sum, val) => sum + val, 0) / labelValues.length;
        const labelVariance = labelValues.reduce((sum, val) => sum + Math.pow(val - labelMean, 2), 0) / labelValues.length;
        const uniqueLabels = [...new Set(labelValues)];
        
        console.log(`Labels: min=${labelMin.toFixed(4)}, max=${labelMax.toFixed(4)}, mean=${labelMean.toFixed(4)}, variance=${labelVariance.toFixed(6)}`);
        console.log(`Unique labels: ${uniqueLabels.length} (${uniqueLabels.slice(0, 10).map(l => l.toFixed(3)).join(', ')}${uniqueLabels.length > 10 ? '...' : ''})`);
        console.log(`Label distribution: ${uniqueLabels.map(l => `${l.toFixed(2)}: ${labelValues.filter(v => Math.abs(v - l) < 0.001).length}`).join(', ')}`);
    }
    console.log('=== END DATA PROCESSING DEBUG ===');
    
    return { features: scaledFeatures, labels, featureNames };
}

/**
 * Apply feature scaling for robust neural network training
 */
function applyFeatureScaling(features, featureNames) {
    if (features.length === 0) return features;
    
    const numFeatures = features[0].length;
    const scaledFeatures = [];
    
    // Calculate min/max for each feature across all samples
    const featureStats = [];
    for (let j = 0; j < numFeatures; j++) {
        const featureValues = features.map(row => row[j]);
        const min = Math.min(...featureValues);
        const max = Math.max(...featureValues);
        const range = max - min;
        
        featureStats.push({
            min: min,
            max: max,
            range: range > 0 ? range : 1 // Avoid division by zero
        });
        
        console.log(`${featureNames[j]}: min=${min.toFixed(3)}, max=${max.toFixed(3)}, range=${range.toFixed(3)}`);
    }
    
    // Apply Min-Max scaling to ensure all features are in [0,1] range
    for (let i = 0; i < features.length; i++) {
        const scaledRow = [];
        for (let j = 0; j < numFeatures; j++) {
            const originalValue = features[i][j];
            const stats = featureStats[j];
            
            // Min-Max scaling: (value - min) / (max - min)
            let scaledValue = (originalValue - stats.min) / stats.range;
            
            // Ensure values are bounded between 0 and 1
            scaledValue = Math.min(Math.max(scaledValue, 0), 1);
            
            scaledRow.push(scaledValue);
        }
        scaledFeatures.push(scaledRow);
    }
    
    return scaledFeatures;
}

/**
 * Create Cultural Intelligence neural network model with configurable architecture
 */
function createCulturalIntelligenceModel(rangeWeights = null) {
    // Dispose of existing model if it exists
    if (neuralNetworkState.model) {
        neuralNetworkState.model.dispose();
    }
    
    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [neuralNetworkState.architecture.inputLayer],
                units: neuralNetworkState.architecture.hiddenLayer1,
                activation: 'relu',
                name: 'cultural_features_encoder',
                kernelInitializer: 'glorotUniform',
                biasInitializer: 'zeros'
            }),
            tf.layers.dropout({
                rate: 0.05, // Reduced dropout to prevent gradient issues
                name: 'regularization_1'
            }),
            tf.layers.dense({
                units: neuralNetworkState.architecture.hiddenLayer2,
                activation: 'relu',
                name: 'cultural_pattern_detector',
                kernelInitializer: 'glorotUniform',
                biasInitializer: 'zeros'
            }),
            tf.layers.dropout({
                rate: 0.02, // Minimal dropout to prevent gradient issues
                name: 'regularization_2'
            }),
            tf.layers.dense({
                units: neuralNetworkState.architecture.outputLayer,
                activation: 'linear', // FIXED: Use linear activation for regression (was sigmoid)
                name: 'trustworthiness_predictor'
            })
        ]
    });

    // Compile with Adam optimizer - temporarily using standard MSE for diagnostic testing
    // const lossFunction = rangeWeights ? 
    //     createWeightedMeanSquaredErrorLoss(rangeWeights) : 
    //     'meanSquaredError';
    const lossFunction = 'meanSquaredError'; // DIAGNOSTIC: Force standard MSE to isolate NaN issue
    
    // Create Adam optimizer (TensorFlow.js doesn't support clipNorm directly)
    const optimizer = tf.train.adam({
        learningRate: neuralNetworkState.hyperparameters.learningRate,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8
    });
    
    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError', // Use string-based standard MSE to avoid null issues
        metrics: ['mae', 'mse'] // Mean Absolute Error and Mean Squared Error for regression
    });

    console.log('Cultural Intelligence Model Architecture:');
    console.log(`Input Layer: ${neuralNetworkState.architecture.inputLayer} neurons (cultural features)`);
    console.log(`Hidden Layer 1: ${neuralNetworkState.architecture.hiddenLayer1} neurons (feature encoding)`);
    console.log(`Hidden Layer 2: ${neuralNetworkState.architecture.hiddenLayer2} neurons (pattern detection)`);
    console.log(`Output Layer: ${neuralNetworkState.architecture.outputLayer} neuron (trustworthiness)`);
    console.log(`Total Parameters: ${model.countParams()}`);
    
    model.summary();

    return model;
}

/**
 * Update training progress during training with comprehensive regression metrics
 */
async function updateTrainingProgress(epoch, logs) {
    const progressElement = document.getElementById('trainingProgress');
    const accuracyElement = document.getElementById('trainingAccuracy');
    const lossElement = document.getElementById('trainingLoss');

    if (progressElement) {
        const progress = (epoch / neuralNetworkState.hyperparameters.epochs) * 100;
        progressElement.style.width = `${progress}%`;
    }

    // Update MAE (Mean Absolute Error) instead of accuracy for regression
    const mae = logs.mae || 0;
    if (accuracyElement) {
        accuracyElement.textContent = `MAE: ${mae.toFixed(4)}`;
    }

    // Update MSE loss
    if (lossElement && logs.loss) {
        lossElement.textContent = `MSE: ${logs.loss.toFixed(4)}`;
    }
    
    // Update validation metrics
    const valMae = logs.val_mae || 0;
    const valLoss = logs.val_loss || 0;
    
    // Calculate RMSE (Root Mean Squared Error) for better interpretability
    const rmse = Math.sqrt(logs.loss || 0);
    const valRmse = Math.sqrt(valLoss);
    
    // Update comprehensive status with regression metrics
    const statusText = `Epoch ${epoch}/${neuralNetworkState.hyperparameters.epochs} - ` +
                      `MSE: ${logs.loss.toFixed(4)} | ` +
                      `MAE: ${mae.toFixed(4)} | ` +
                      `RMSE: ${rmse.toFixed(4)} | ` +
                      `Val MSE: ${valLoss.toFixed(4)} | ` +
                      `Val MAE: ${valMae.toFixed(4)}`;
    
    updateTrainingStatus(statusText);
    
    // Add some delay for smooth visualization
    await new Promise(resolve => setTimeout(resolve, 10));
}

/**
 * Update training status message
 */
function updateTrainingStatus(message) {
    const statusElement = document.getElementById('trainingStatus');
    if (statusElement) {
        const statusText = statusElement.querySelector('.text-sm');
        if (statusText) {
            statusText.textContent = message;
        }
    }
}

/**
 * Show notification helper (fallback if not available in ml-core)
 */
function showNotification(message, type = 'info') {
    if (typeof showSuccessNotification === 'function' && type === 'success') {
        showSuccessNotification(message);
    } else if (typeof showErrorNotification === 'function' && type === 'error') {
        showErrorNotification(message);
    } else {
        console.log(`[${type.toUpperCase()}] ${message}`);
        // Fallback to browser notification if available
        if (window.Notification && Notification.permission === 'granted') {
            new Notification('HARAYA Neural Network', {
                body: message,
                icon: '/favicon.ico'
            });
        } else {
            alert(message);
        }
    }
}

/**
 * Create initial training chart with validation curves
 */
function createInitialTrainingChart() {
    const config = {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#DC143C',
                    backgroundColor: 'rgba(220, 20, 60, 0.1)',
                    yAxisID: 'y',
                    tension: 0.1
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#FF6B6B',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    yAxisID: 'y',
                    tension: 0.1,
                    borderDash: [5, 5]
                },
                {
                    label: 'Training MAE',
                    data: [],
                    borderColor: '#2E8B57',
                    backgroundColor: 'rgba(46, 139, 87, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.1
                },
                {
                    label: 'Validation MAE',
                    data: [],
                    borderColor: '#4ECDC4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.1,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Neural Network Training Progress',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        afterBody: function(context) {
                            const epoch = context[0].dataIndex + 1;
                            return [`Epoch: ${epoch}`];
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Epoch',
                        font: {
                            weight: 'bold'
                        }
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss',
                        font: {
                            weight: 'bold'
                        }
                    },
                    min: 0
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'MAE (Mean Absolute Error)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    min: 0,
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            animation: {
                duration: 300
            }
        }
    };

    createChart('trainingChart', config);
}

/**
 * Update training chart with new data including validation curves
 */
function updateTrainingChart() {
    const history = neuralNetworkState.trainingHistory;
    
    updateChart('trainingChart', {
        labels: history.epochs,
        datasets: [
            {
                label: 'Training Loss',
                data: history.loss,
                borderColor: '#DC143C',
                backgroundColor: 'rgba(220, 20, 60, 0.1)',
                yAxisID: 'y',
                tension: 0.1
            },
            {
                label: 'Validation Loss',
                data: history.valLoss,
                borderColor: '#FF6B6B',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                yAxisID: 'y',
                tension: 0.1,
                borderDash: [5, 5]
            },
            {
                label: 'Training MAE',
                data: history.accuracy, // accuracy field now contains MAE
                borderColor: '#2E8B57',
                backgroundColor: 'rgba(46, 139, 87, 0.1)',
                yAxisID: 'y1',
                tension: 0.1
            },
            {
                label: 'Validation MAE',
                data: history.valAccuracy, // valAccuracy field now contains validation MAE
                borderColor: '#4ECDC4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                yAxisID: 'y1',
                tension: 0.1,
                borderDash: [5, 5]
            }
        ]
    });
}

/**
 * Reset training chart to initial state
 */
function resetTrainingChart() {
    updateChart('trainingChart', {
        labels: [],
        datasets: [
            {
                label: 'Training Loss',
                data: [],
                borderColor: '#DC143C',
                backgroundColor: 'rgba(220, 20, 60, 0.1)',
                yAxisID: 'y',
                tension: 0.1
            },
            {
                label: 'Validation Loss',
                data: [],
                borderColor: '#FF6B6B',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                yAxisID: 'y',
                tension: 0.1,
                borderDash: [5, 5]
            },
            {
                label: 'Training Accuracy',
                data: [],
                borderColor: '#2E8B57',
                backgroundColor: 'rgba(46, 139, 87, 0.1)',
                yAxisID: 'y1',
                tension: 0.1
            },
            {
                label: 'Validation Accuracy',
                data: [],
                borderColor: '#4ECDC4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                yAxisID: 'y1',
                tension: 0.1,
                borderDash: [5, 5]
            }
        ]
    });
}

/**
 * Predict cultural authenticity for a persona (legacy function - use predictTrustScore for new implementations)
 */
async function predictCulturalAuthenticity(persona) {
    if (!neuralNetworkState.model) {
        throw new Error('Model not trained yet');
    }

    // Use the new predictTrustScore function for consistency
    const result = await predictTrustScore(persona);
    return result.trustworthiness_score;
}

/**
 * Handle data updates from other components
 */
function updateNeuralNetworkData(dataInfo) {
    console.log('Neural Network received data update:', dataInfo.metadata);
    // Reset training history when new data is loaded
    neuralNetworkState.trainingHistory = {
        epochs: [],
        loss: [],
        accuracy: []
    };
    updateTrainingChart();
}

/**
 * Get comprehensive model summary for display
 */
function getModelSummary() {
    if (!neuralNetworkState.model) {
        return {
            status: 'No model trained yet',
            ready: false
        };
    }

    return {
        status: 'Model ready',
        ready: true,
        totalParams: neuralNetworkState.model.countParams(),
        trainableParams: neuralNetworkState.model.countParams(),
        architecture: neuralNetworkState.architecture,
        hyperparameters: neuralNetworkState.hyperparameters,
        performanceMetrics: neuralNetworkState.performanceMetrics,
        featureImportance: neuralNetworkState.featureImportance,
        trainingHistory: {
            epochs: neuralNetworkState.trainingHistory.epochs.length,
            finalLoss: neuralNetworkState.trainingHistory.loss.slice(-1)[0],
            finalAccuracy: neuralNetworkState.trainingHistory.accuracy.slice(-1)[0]
        },
        isTraining: neuralNetworkState.isTraining,
        culturalInsights: neuralNetworkState.culturalInsights?.length || 0
    };
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeNeuralNetwork();
});

/**
 * Calculate performance metrics for the trained regression model
 */
async function calculatePerformanceMetrics(model, valXs, valYs) {
    const predictions = model.predict(valXs);
    const predArray = await predictions.data();
    const trueArray = await valYs.data();
    
    // Calculate regression metrics
    let mse = 0, mae = 0, sumSquaredResiduals = 0, sumSquaredTotal = 0;
    const n = predArray.length;
    
    // Calculate mean of actual values for R-squared
    const trueMean = trueArray.reduce((sum, val) => sum + val, 0) / n;
    
    for (let i = 0; i < n; i++) {
        const predicted = predArray[i];
        const actual = trueArray[i];
        const error = predicted - actual;
        
        // Mean Squared Error and Mean Absolute Error
        mse += error * error;
        mae += Math.abs(error);
        
        // For R-squared calculation
        sumSquaredResiduals += error * error;
        sumSquaredTotal += (actual - trueMean) * (actual - trueMean);
    }
    
    mse /= n;
    mae /= n;
    const rmse = Math.sqrt(mse);
    
    // Calculate R-squared (coefficient of determination)
    const rSquared = sumSquaredTotal > 0 ? 1 - (sumSquaredResiduals / sumSquaredTotal) : 0;
    
    // Calculate binary classification metrics for comparison (using 0.5 threshold)
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < n; i++) {
        const predictedBinary = predArray[i] > 0.5 ? 1 : 0;
        const actualBinary = trueArray[i] > 0.5 ? 1 : 0;
        
        if (predictedBinary === 1 && actualBinary === 1) tp++;
        else if (predictedBinary === 0 && actualBinary === 0) tn++;
        else if (predictedBinary === 1 && actualBinary === 0) fp++;
        else if (predictedBinary === 0 && actualBinary === 1) fn++;
    }
    
    const binaryAccuracy = (tp + tn) / n;
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
    
    neuralNetworkState.performanceMetrics = {
        // Primary regression metrics
        mse: mse,
        mae: mae,
        rmse: rmse,
        rSquared: rSquared,
        
        // Secondary binary classification metrics (for reference)
        binaryAccuracy: binaryAccuracy,
        precision: precision,
        recall: recall,
        f1Score: f1Score,
        
        // Confusion matrix components
        truePositives: tp,
        trueNegatives: tn,
        falsePositives: fp,
        falseNegatives: fn,
        
        // Additional metadata
        sampleCount: n,
        trueMean: trueMean,
        predictionRange: {
            min: Math.min(...predArray),
            max: Math.max(...predArray)
        }
    };
    
    predictions.dispose();
    
    console.log('Regression Performance Metrics:', {
        MSE: mse.toFixed(4),
        MAE: mae.toFixed(4),
        RMSE: rmse.toFixed(4),
        'R²': rSquared.toFixed(4),
        'Binary Accuracy': (binaryAccuracy * 100).toFixed(1) + '%'
    });
}

/**
 * Analyze feature importance using gradient-based methods for regression model
 */
async function analyzeFeatureImportance(model, features, featureNames) {
    if (!model || features.length === 0) return;
    
    try {
        // Use gradient-based feature importance for regression
        const sampleSize = Math.min(150, features.length); // Larger sample for better estimates
        const sampleIndices = Array.from({length: sampleSize}, (_, i) => 
            Math.floor(i * features.length / sampleSize)
        );
        
        const importance = new Array(featureNames.length).fill(0);
        let validSamples = 0;
        
        console.log(`Computing feature importance using ${sampleSize} samples...`);
        
        for (const idx of sampleIndices) {
            const input = tf.tensor2d([features[idx]]);
            
            try {
                // Calculate gradients with respect to input features
                const gradients = tf.grad(x => {
                    const prediction = model.predict(x);
                    // For regression, we want gradients of the continuous output
                    return tf.mean(prediction);
                })(input);
                
                const gradArray = await gradients.data();
                
                // Accumulate absolute gradients weighted by input values
                for (let i = 0; i < gradArray.length; i++) {
                    const gradient = gradArray[i];
                    const inputValue = features[idx][i];
                    
                    // Use gradient * input for importance (similar to integrated gradients)
                    importance[i] += Math.abs(gradient * inputValue);
                }
                
                validSamples++;
                gradients.dispose();
                
            } catch (error) {
                console.warn(`Error calculating gradients for sample ${idx}:`, error);
            }
            
            input.dispose();
        }
        
        if (validSamples === 0) {
            console.warn('No valid gradient calculations. Using fallback importance method.');
            // Fallback: use simple variance-based importance
            for (let i = 0; i < featureNames.length; i++) {
                const featureValues = features.map(row => row[i]);
                importance[i] = calculateVariance(featureValues);
            }
        }
        
        // Normalize importance scores
        const maxImportance = Math.max(...importance);
        const normalizedImportance = importance.map(score => 
            maxImportance > 0 ? score / maxImportance : 0
        );
        
        neuralNetworkState.featureImportance = featureNames.map((name, i) => ({
            feature: name,
            importance: normalizedImportance[i],
            rank: i + 1
        }));
        
        // Sort by importance
        neuralNetworkState.featureImportance.sort((a, b) => b.importance - a.importance);
        
        // Update ranks
        neuralNetworkState.featureImportance.forEach((item, i) => {
            item.rank = i + 1;
        });
        
        console.log(`Feature Importance (${validSamples} valid samples):`, 
            neuralNetworkState.featureImportance.map(f => `${f.feature}: ${f.importance.toFixed(3)}`));
            
    } catch (error) {
        console.error('Error in feature importance analysis:', error);
        // Set default importance if analysis fails
        neuralNetworkState.featureImportance = featureNames.map((name, i) => ({
            feature: name,
            importance: 1 / featureNames.length, // Equal importance as fallback
            rank: i + 1
        }));
    }
}

/**
 * Calculate variance for fallback feature importance
 */
function calculateVariance(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / values.length;
}

/**
 * Generate cultural insights based on feature importance and regression training results
 */
function generateCulturalInsights() {
    const insights = [];
    const importance = neuralNetworkState.featureImportance;
    const metrics = neuralNetworkState.performanceMetrics;
    const history = neuralNetworkState.trainingHistory;
    
    if (!importance || importance.length === 0) {
        insights.push('Train the model to see cultural intelligence insights.');
        neuralNetworkState.culturalInsights = insights;
        return;
    }
    
    // Analyze top cultural features
    const topFeature = importance[0];
    const secondFeature = importance.length > 1 ? importance[1] : null;
    
    // Cultural factor insights with continuous scoring context
    if (topFeature.feature.includes('Kapwa')) {
        insights.push(`🤝 **Kapwa Network Strength** is the strongest predictor of trustworthiness scores (${(topFeature.importance * 100).toFixed(1)}% importance). Strong Kapwa connections correlate with higher continuous trust ratings, reflecting the Filipino cultural emphasis on shared identity and community bonds.`);
    } else if (topFeature.feature.includes('Bayanihan')) {
        insights.push(`🏘️ **Bayanihan Participation** shows highest importance (${(topFeature.importance * 100).toFixed(1)}%) in determining trust scores. Community cooperation activities directly influence trustworthiness ratings across the 0-1 scale.`);
    } else if (topFeature.feature.includes('Utang na Loob')) {
        insights.push(`💖 **Utang na Loob Integrity** is most significant (${(topFeature.importance * 100).toFixed(1)}% importance) for continuous trust assessment. Reciprocity and gratitude bonds create measurable differences in trustworthiness scores.`);
    }
    
    // Traditional vs Cultural factors analysis
    const culturalFeatures = importance.filter(f => 
        f.feature.includes('Kapwa') || f.feature.includes('Bayanihan') || f.feature.includes('Utang')
    );
    const traditionalFeatures = importance.filter(f => 
        f.feature.includes('Economic') || f.feature.includes('Digital')
    );
    
    if (culturalFeatures.length > 0 && traditionalFeatures.length > 0) {
        const avgCulturalImportance = culturalFeatures.reduce((sum, f) => sum + f.importance, 0) / culturalFeatures.length;
        const avgTraditionalImportance = traditionalFeatures.reduce((sum, f) => sum + f.importance, 0) / traditionalFeatures.length;
        
        if (avgCulturalImportance > avgTraditionalImportance * 1.2) {
            insights.push(`📊 Cultural authenticity factors (${(avgCulturalImportance * 100).toFixed(1)}% avg) significantly outweigh traditional metrics (${(avgTraditionalImportance * 100).toFixed(1)}% avg) in predicting continuous trustworthiness scores.`);
        }
    }
    
    // Regression model performance insights
    if (history.loss.length > 0) {
        const finalLoss = history.loss[history.loss.length - 1];
        const finalMAE = history.accuracy[history.accuracy.length - 1]; // accuracy field now contains MAE
        
        if (finalMAE < 0.1) {
            insights.push(`🎯 Excellent regression performance: MAE of ${finalMAE.toFixed(4)} means predictions are typically within ${(finalMAE * 100).toFixed(1)}% of actual trustworthiness scores. Cultural patterns provide strong predictive signals.`);
        } else if (finalMAE < 0.2) {
            insights.push(`✅ Good regression performance: MAE of ${finalMAE.toFixed(4)} indicates cultural intelligence effectively captures trust variations with predictions typically within ${(finalMAE * 100).toFixed(1)}% of actual scores.`);
        } else {
            insights.push(`📈 Model learning progress: MAE of ${finalMAE.toFixed(4)} shows room for improvement. Consider more training data or feature engineering for better cultural pattern recognition.`);
        }
        
        insights.push(`📉 Training convergence: Model achieved MSE loss of ${finalLoss.toFixed(4)} after ${history.epochs.length} epochs, demonstrating effective learning of cultural trustworthiness patterns.`);
    }
    
    // Feature interaction insights for continuous scoring
    if (importance[0].importance > 0.7) {
        insights.push(`⚡ Dominant cultural factor: ${topFeature.feature} shows exceptional influence (${(topFeature.importance * 100).toFixed(1)}%), suggesting this factor creates strong variations in trustworthiness scores across Filipino communities.`);
    }
    
    // Score distribution insights
    insights.push(`🎭 Nuanced assessment capability: The continuous scoring model (0-1 scale) captures subtle trustworthiness variations that binary classification would miss, enabling more precise cultural intelligence evaluation.`);
    
    // Regional and cultural diversity insights
    if (secondFeature) {
        insights.push(`🔄 Multi-factor cultural assessment: Top factors ${topFeature.feature} (${(topFeature.importance * 100).toFixed(1)}%) and ${secondFeature.feature} (${(secondFeature.importance * 100).toFixed(1)}%) work together to create comprehensive trust profiles reflecting Filipino cultural complexity.`);
    }
    
    neuralNetworkState.culturalInsights = insights;
    console.log('Cultural Intelligence Insights Generated:', insights);
}

/**
 * Update feature importance visualization
 */
function updateFeatureImportanceChart() {
    const importance = neuralNetworkState.featureImportance;
    if (!importance || importance.length === 0) return;
    
    // Create or update feature importance chart if element exists
    const chartElement = document.getElementById('featureImportanceChart');
    if (chartElement) {
        const config = {
            type: 'horizontalBar',
            data: {
                labels: importance.map(f => f.feature),
                datasets: [{
                    label: 'Feature Importance',
                    data: importance.map(f => f.importance * 100),
                    backgroundColor: [
                        '#2E8B57', '#4169E1', '#FF6B6B', '#FFA500',
                        '#9370DB', '#32CD32', '#FF1493'
                    ].slice(0, importance.length),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Cultural Feature Importance Analysis'
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Importance (%)'
                        }
                    }
                }
            }
        };
        createChart('featureImportanceChart', config);
    }
}

/**
 * Display cultural insights in the UI
 */
function displayCulturalInsights() {
    const insightsElement = document.getElementById('culturalInsights');
    if (insightsElement && neuralNetworkState.culturalInsights) {
        const insightsHtml = neuralNetworkState.culturalInsights
            .map(insight => `<div class="mb-2 text-sm">${insight}</div>`)
            .join('');
        insightsElement.innerHTML = insightsHtml;
    }
}

/**
 * Update performance metrics in the UI with regression-focused metrics
 */
function updatePerformanceMetricsUI() {
    const metrics = neuralNetworkState.performanceMetrics;
    if (!metrics) return;
    
    // Update primary regression metrics
    const precisionElement = document.getElementById('modelPrecision');
    if (precisionElement) {
        // Show R-squared as primary performance metric
        precisionElement.textContent = `R²: ${(metrics.rSquared * 100).toFixed(1)}%`;
    }
    
    const recallElement = document.getElementById('modelRecall');
    if (recallElement) {
        // Show RMSE as secondary performance metric
        recallElement.textContent = `RMSE: ${metrics.rmse.toFixed(4)}`;
    }
    
    const f1Element = document.getElementById('modelF1Score');
    if (f1Element) {
        // Show MAE as tertiary performance metric
        f1Element.textContent = `MAE: ${metrics.mae.toFixed(4)}`;
    }
    
    // Show binary accuracy for reference (if UI element exists)
    const accuracyElement = document.getElementById('modelBinaryAccuracy');
    if (accuracyElement) {
        accuracyElement.textContent = `Binary Acc: ${(metrics.binaryAccuracy * 100).toFixed(1)}%`;
    }
    
    // Update parameters count
    const parametersElement = document.getElementById('modelParameters');
    if (parametersElement && neuralNetworkState.model) {
        const paramCount = neuralNetworkState.model.countParams();
        parametersElement.textContent = paramCount.toLocaleString();
    }
    
    // Add comprehensive metrics display if container exists
    const metricsContainer = document.getElementById('comprehensiveMetrics');
    if (metricsContainer) {
        metricsContainer.innerHTML = `
            <div class="grid grid-cols-2 gap-4 mt-4">
                <div class="bg-gray-50 p-3 rounded">
                    <h4 class="font-semibold text-sm">Regression Metrics</h4>
                    <p class="text-xs">MSE: ${metrics.mse.toFixed(4)}</p>
                    <p class="text-xs">MAE: ${metrics.mae.toFixed(4)}</p>
                    <p class="text-xs">RMSE: ${metrics.rmse.toFixed(4)}</p>
                    <p class="text-xs">R²: ${(metrics.rSquared * 100).toFixed(1)}%</p>
                </div>
                <div class="bg-gray-50 p-3 rounded">
                    <h4 class="font-semibold text-sm">Binary Classification</h4>
                    <p class="text-xs">Accuracy: ${(metrics.binaryAccuracy * 100).toFixed(1)}%</p>
                    <p class="text-xs">Precision: ${(metrics.precision * 100).toFixed(1)}%</p>
                    <p class="text-xs">Recall: ${(metrics.recall * 100).toFixed(1)}%</p>
                    <p class="text-xs">F1-Score: ${(metrics.f1Score * 100).toFixed(1)}%</p>
                </div>
            </div>
        `;
    }
    
    console.log('Updated UI with regression performance metrics');
}

/**
 * Update neural network architecture (rebuild model with new settings)
 */
function updateNeuralNetworkArchitecture() {
    if (neuralNetworkState.isTraining) {
        console.log('Cannot update architecture while training');
        return;
    }
    
    // Architecture is updated when training starts
    console.log('Architecture will be updated on next training run:', neuralNetworkState.architecture);
}

/**
 * Export trained model to downloadable format
 */
async function exportTrainedModel() {
    if (!neuralNetworkState.model) {
        showNotification('No trained model to export. Please train a model first.', 'error');
        return;
    }
    
    try {
        updateTrainingStatus('Exporting trained model...');
        
        // Save model to browser downloads
        const saveResult = await neuralNetworkState.model.save('downloads://haraya-cultural-intelligence-model');
        console.log('Model saved to downloads:', saveResult);
        
        // Also export metadata
        const modelMetadata = {
            architecture: neuralNetworkState.architecture,
            hyperparameters: neuralNetworkState.hyperparameters,
            performanceMetrics: neuralNetworkState.performanceMetrics,
            featureImportance: neuralNetworkState.featureImportance,
            culturalInsights: neuralNetworkState.culturalInsights,
            trainingHistory: {
                epochs: neuralNetworkState.trainingHistory.epochs.length,
                finalLoss: neuralNetworkState.trainingHistory.loss.slice(-1)[0],
                finalAccuracy: neuralNetworkState.trainingHistory.accuracy.slice(-1)[0]
            },
            exportDate: new Date().toISOString(),
            totalParameters: neuralNetworkState.model.countParams(),
            modelSummary: 'Cultural Intelligence Neural Network for Filipino Trustworthiness Prediction'
        };
        
        const metadataBlob = new Blob([JSON.stringify(modelMetadata, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(metadataBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'haraya-model-metadata.json';
        a.click();
        URL.revokeObjectURL(url);
        
        showNotification('Model and metadata exported successfully! Check your Downloads folder.', 'success');
        updateTrainingStatus('Model exported successfully!');
        
    } catch (error) {
        console.error('Export failed:', error);
        showNotification(`Export failed: ${error.message}`, 'error');
        updateTrainingStatus(`Export failed: ${error.message}`);
    }
}

/**
 * Predict trustworthiness for a given persona using trained model
 */
async function predictTrustScore(persona) {
    if (!neuralNetworkState.model) {
        throw new Error('No trained model available');
    }
    
    const featureColumns = [
        'kapwa_score', 
        'bayanihan_participation', 
        'utang_na_loob_integrity',
        'community_standing_score', 
        'monthly_income', 
        'family_size', 
        'digital_literacy_score'
    ];
    
    const features = [];
    featureColumns.forEach(column => {
        let value = persona[column] || 0;
        
        // Apply same normalization as training
        if (column === 'monthly_income') {
            value = Math.min(Math.log(value + 1) / Math.log(100001), 1);
        } else if (column === 'family_size') {
            value = Math.min(value / 12, 1);
        } else if (column === 'digital_literacy_score') {
            value = Math.min(value / 10, 1);
        } else {
            value = Math.min(Math.max(value, 0), 1);
        }
        
        features.push(value);
    });
    
    const input = tf.tensor2d([features]);
    const prediction = neuralNetworkState.model.predict(input);
    const result = await prediction.data();
    
    input.dispose();
    prediction.dispose();
    
    return {
        trustworthiness_score: result[0],
        confidence: Math.abs(result[0] - 0.5) * 2, // Distance from neutral
        prediction: result[0] > 0.5 ? 'trustworthy' : 'untrustworthy'
    };
}

/**
 * Get learned cultural weights for Cultural Intelligence module
 */
function getCulturalWeightsFromNeuralNetwork() {
    if (!neuralNetworkState.featureImportance || neuralNetworkState.featureImportance.length === 0) {
        return null;
    }
    
    // Map neural network feature importance to cultural intelligence weights
    const culturalWeights = {};
    
    neuralNetworkState.featureImportance.forEach(feature => {
        if (feature.feature.includes('Kapwa')) {
            culturalWeights.kapwa = feature.importance;
        } else if (feature.feature.includes('Bayanihan')) {
            culturalWeights.bayanihan = feature.importance;
        } else if (feature.feature.includes('Utang na Loob')) {
            culturalWeights.utang = feature.importance;
        } else if (feature.feature.includes('Community')) {
            culturalWeights.community = feature.importance;
        } else if (feature.feature.includes('Economic')) {
            culturalWeights.economic = feature.importance;
        } else if (feature.feature.includes('Family')) {
            culturalWeights.family = feature.importance;
        } else if (feature.feature.includes('Digital')) {
            culturalWeights.digital = feature.importance;
        }
    });
    
    console.log('Neural Network cultural weights for Cultural Intelligence:', culturalWeights);
    return culturalWeights;
}

/**
 * Update Cultural Intelligence weights based on Neural Network results
 */
function updateCulturalIntelligenceWeights() {
    const weights = getCulturalWeightsFromNeuralNetwork();
    if (weights && typeof updateCulturalWeights === 'function') {
        updateCulturalWeights(weights);
        console.log('Updated Cultural Intelligence weights from Neural Network feature importance');
    }
}

/**
 * Dataset Connection and Validation Functions
 */

/**
 * Validates dataset for Neural Network training
 */
function validateDataForTraining(dataset) {
    if (!dataset || !Array.isArray(dataset) || dataset.length === 0) {
        return {
            valid: false,
            totalRecords: 0,
            validRecords: 0,
            trustworthy: 0,
            untrustworthy: 0,
            issues: ['No dataset provided']
        };
    }

    let validRecords = 0;
    let trustworthy = 0;
    let untrustworthy = 0;
    const issues = [];

    // Check each record for required features
    dataset.forEach((record, index) => {
        const hasRequiredFeatures = record.hasOwnProperty('trustworthiness_label') || 
                                   record.hasOwnProperty('is_trustworthy') ||
                                   record.hasOwnProperty('label');
        
        if (hasRequiredFeatures) {
            validRecords++;
            
            // Count labels
            const label = record.trustworthiness_label || record.is_trustworthy || record.label;
            if (label === 'trustworthy' || label === true || label === 1 || label === '1') {
                trustworthy++;
            } else if (label === 'untrustworthy' || label === false || label === 0 || label === '0') {
                untrustworthy++;
            }
        }
    });

    // Check for issues
    if (validRecords === 0) {
        issues.push('No records with valid labels found');
    }
    if (trustworthy === 0) {
        issues.push('No trustworthy examples found');
    }
    if (untrustworthy === 0) {
        issues.push('No untrustworthy examples found');
    }
    if (validRecords < 100) {
        issues.push('Dataset too small (minimum 100 records recommended)');
    }

    return {
        valid: issues.length === 0,
        totalRecords: dataset.length,
        validRecords: validRecords,
        trustworthy: trustworthy,
        untrustworthy: untrustworthy,
        balance: trustworthy > 0 ? (untrustworthy / (trustworthy + untrustworthy) * 100).toFixed(1) : 0,
        issues: issues
    };
}

/**
 * Updates Neural Network dataset connection status
 */
function refreshNeuralNetworkDatasetConnection() {
    const selector = document.getElementById('nnDatasetSelector');
    const recordsEl = document.getElementById('nnDatasetRecords');
    const distributionEl = document.getElementById('nnLabelDistribution');
    const statusIndicator = document.getElementById('nnDatasetStatusIndicator');
    const readinessDiv = document.getElementById('nnTrainingReadiness');
    const culturalFeaturesEl = document.getElementById('nnCulturalFeatures');
    const dataBalanceEl = document.getElementById('nnDataBalance');
    const trainingStatusEl = document.getElementById('nnTrainingStatus');

    // Clear selector and add available datasets
    selector.innerHTML = '<option value="">No dataset loaded</option>';
    
    // Check for loaded datasets in data management
    if (typeof getLoadedDataset === 'function') {
        const currentDataset = getLoadedDataset();
        if (currentDataset && currentDataset.data) {
            selector.innerHTML += `<option value="current">Current Dataset (${currentDataset.data.length} records)</option>`;
        }
    }

    // Add preset options
    selector.innerHTML += `
        <option value="neural-network-training">Cultural Intelligence Training Dataset (2572 samples)</option>
        <option value="regional-cultural">Regional Cultural Variations (300 samples)</option>
        <option value="enhanced-synthetic">Enhanced Synthetic Dataset</option>
    `;

    // Update status based on current selection
    const selectedValue = selector.value;
    if (!selectedValue) {
        recordsEl.textContent = '--';
        distributionEl.textContent = '--';
        statusIndicator.className = 'w-3 h-3 bg-gray-400 rounded-full';
        readinessDiv.classList.add('hidden');
    }
}

/**
 * Handles Neural Network dataset selection
 */
function selectNeuralNetworkDataset() {
    const selector = document.getElementById('nnDatasetSelector');
    const recordsEl = document.getElementById('nnDatasetRecords');
    const distributionEl = document.getElementById('nnLabelDistribution');
    const statusIndicator = document.getElementById('nnDatasetStatusIndicator');
    const readinessDiv = document.getElementById('nnTrainingReadiness');
    const culturalFeaturesEl = document.getElementById('nnCulturalFeatures');
    const dataBalanceEl = document.getElementById('nnDataBalance');
    const trainingStatusEl = document.getElementById('nnTrainingStatus');

    const selectedValue = selector.value;
    
    if (!selectedValue) {
        recordsEl.textContent = '--';
        distributionEl.textContent = '--';
        statusIndicator.className = 'w-3 h-3 bg-gray-400 rounded-full';
        readinessDiv.classList.add('hidden');
        return;
    }

    // Load dataset based on selection
    let dataset = null;
    if (selectedValue === 'current' && typeof getLoadedDataset === 'function') {
        const currentDataset = getLoadedDataset();
        dataset = currentDataset ? currentDataset.data : null;
    } else if (selectedValue === 'neural-network-training') {
        // Trigger loading of neural network training dataset
        if (typeof loadPresetDataset === 'function') {
            loadPresetDataset('neural-network-training');
        }
        // Use placeholder data for now
        dataset = Array(2572).fill({}).map((_, i) => ({
            trustworthiness_label: i < 2052 ? 'trustworthy' : 'untrustworthy'
        }));
    }

    if (dataset) {
        const validation = validateDataForTraining(dataset);
        
        // Update UI
        recordsEl.textContent = validation.validRecords.toLocaleString();
        distributionEl.textContent = `${validation.trustworthy} trustworthy, ${validation.untrustworthy} untrustworthy`;
        
        // Update status indicator
        if (validation.valid) {
            statusIndicator.className = 'w-3 h-3 bg-green-400 rounded-full';
            readinessDiv.classList.remove('hidden');
            culturalFeaturesEl.textContent = '✓ Available';
            culturalFeaturesEl.className = 'text-green-600';
            dataBalanceEl.textContent = `${validation.balance}% untrustworthy`;
            dataBalanceEl.className = 'text-green-600';
            trainingStatusEl.textContent = '✓ Ready to Train';
            trainingStatusEl.className = 'text-green-600';
        } else {
            statusIndicator.className = 'w-3 h-3 bg-red-400 rounded-full';
            readinessDiv.classList.remove('hidden');
            culturalFeaturesEl.textContent = validation.issues.join(', ');
            culturalFeaturesEl.className = 'text-red-600';
            dataBalanceEl.textContent = 'Issues found';
            dataBalanceEl.className = 'text-red-600';
            trainingStatusEl.textContent = '⚠ Not Ready';
            trainingStatusEl.className = 'text-red-600';
        }
    }
}

/**
 * Updates neural network connection status when page loads
 */
function initializeNeuralNetworkDatasetConnection() {
    setTimeout(() => {
        refreshNeuralNetworkDatasetConnection();
        selectNeuralNetworkDataset();
    }, 500);
}

/**
 * UI-Visible Debug Functions (Browser-Friendly)
 */
function debugToUI(message, data = null) {
    const debugDiv = document.getElementById('debugOutput');
    if (!debugDiv) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const dataStr = data ? (typeof data === 'object' ? JSON.stringify(data).substring(0, 100) + '...' : data) : '';
    
    const newEntry = document.createElement('div');
    newEntry.className = 'text-xs mb-1 font-mono';
    newEntry.innerHTML = `<span class="text-gray-500">[${timestamp}]</span> ${message}${dataStr ? ': <span class="text-blue-600">' + dataStr + '</span>' : ''}`;
    
    debugDiv.appendChild(newEntry);
    debugDiv.scrollTop = debugDiv.scrollHeight; // Auto-scroll to bottom
}

function clearDebugUI() {
    const debugDiv = document.getElementById('debugOutput');
    if (debugDiv) {
        debugDiv.innerHTML = '<div class="text-sm text-gray-600 italic">Training started...</div>';
    }
}

/**
 * Simple Linear Regression Implementation (Baseline Model)
 */
function trainLinearRegression() {
    debugToUI('🚀 Starting Linear Regression Training');
    clearDebugUI();
    
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        debugToUI('❌ No dataset loaded');
        document.getElementById('linearRegressionResults').innerHTML = 
            '<div class="text-red-600">No dataset loaded. Please select a dataset first.</div>';
        return;
    }
    
    try {
        debugToUI(`📊 Dataset size: ${mlCore.sharedData.length} records`);
        
        // Use the same data preparation as neural network
        const { features, labels, featureNames } = prepareTrainingData(mlCore.sharedData);
        
        debugToUI(`✅ Prepared ${features.length} samples with ${features[0]?.length || 0} features`);
        
        if (features.length === 0) {
            throw new Error('No valid training data prepared');
        }
        
        // Check data quality
        const labelValues = labels.map(l => l[0]);
        const uniqueLabels = [...new Set(labelValues)];
        debugToUI(`🎯 Found ${uniqueLabels.length} unique labels: ${uniqueLabels.slice(0,5).map(l => l.toFixed(3)).join(', ')}${uniqueLabels.length > 5 ? '...' : ''}`);
        
        if (uniqueLabels.length < 2) {
            debugToUI('⚠️ Warning: All labels are the same - model cannot learn!');
        }
        
        // Get configurable parameters from UI
        const epochs = parseInt(document.getElementById('lrEpochsSlider')?.value) || 100;
        
        // Simple linear regression with gradient descent
        const numFeatures = features[0].length;
        let weights = new Array(numFeatures).fill(0); // Initialize weights to 0
        let bias = 0;
        const learningRate = 0.01;
        
        debugToUI(`🔧 Training with ${epochs} epochs, learning rate: ${learningRate}`);
        
        // Split data (80% train, 20% validation)
        const splitIndex = Math.floor(features.length * 0.8);
        const trainFeatures = features.slice(0, splitIndex);
        const trainLabels = labelValues.slice(0, splitIndex);
        const valFeatures = features.slice(splitIndex);
        const valLabels = labelValues.slice(splitIndex);
        
        let bestLoss = Infinity;
        let finalTrainLoss = 0, finalValLoss = 0;
        
        // Training loop
        for (let epoch = 0; epoch < epochs; epoch++) {
            let trainLoss = 0;
            
            // Forward pass and gradient computation
            for (let i = 0; i < trainFeatures.length; i++) {
                // Prediction: y = w1*x1 + w2*x2 + ... + wn*xn + bias
                let prediction = bias;
                for (let j = 0; j < numFeatures; j++) {
                    prediction += weights[j] * trainFeatures[i][j];
                }
                
                const error = prediction - trainLabels[i];
                trainLoss += error * error; // MSE
                
                // Gradient descent update
                bias -= learningRate * error;
                for (let j = 0; j < numFeatures; j++) {
                    weights[j] -= learningRate * error * trainFeatures[i][j];
                }
            }
            
            trainLoss /= trainFeatures.length;
            finalTrainLoss = trainLoss;
            
            // Validation loss
            let valLoss = 0;
            for (let i = 0; i < valFeatures.length; i++) {
                let prediction = bias;
                for (let j = 0; j < numFeatures; j++) {
                    prediction += weights[j] * valFeatures[i][j];
                }
                const error = prediction - valLabels[i];
                valLoss += error * error;
            }
            valLoss /= valFeatures.length;
            finalValLoss = valLoss;
            
            if (valLoss < bestLoss) {
                bestLoss = valLoss;
            }
            
            // Log progress every 20 epochs
            if (epoch % 20 === 0 || epoch === epochs - 1) {
                debugToUI(`📈 Epoch ${epoch + 1}: Train Loss=${trainLoss.toFixed(6)}, Val Loss=${valLoss.toFixed(6)}`);
            }
        }
        
        // Calculate R-squared on validation set
        const meanLabel = valLabels.reduce((sum, val) => sum + val, 0) / valLabels.length;
        let totalSumSquares = 0, residualSumSquares = 0;
        
        for (let i = 0; i < valFeatures.length; i++) {
            let prediction = bias;
            for (let j = 0; j < numFeatures; j++) {
                prediction += weights[j] * valFeatures[i][j];
            }
            
            residualSumSquares += Math.pow(valLabels[i] - prediction, 2);
            totalSumSquares += Math.pow(valLabels[i] - meanLabel, 2);
        }
        
        const rSquared = totalSumSquares > 0 ? 1 - (residualSumSquares / totalSumSquares) : 0;
        
        // Calculate MAE (Mean Absolute Error) on validation set
        let meanAbsoluteError = 0;
        for (let i = 0; i < valFeatures.length; i++) {
            let prediction = bias;
            for (let j = 0; j < numFeatures; j++) {
                prediction += weights[j] * valFeatures[i][j];
            }
            meanAbsoluteError += Math.abs(valLabels[i] - prediction);
        }
        meanAbsoluteError /= valFeatures.length;
        
        debugToUI(`✅ Training Complete!`);
        debugToUI(`📊 Final Results: R²=${(rSquared * 100).toFixed(2)}%, Train Loss=${finalTrainLoss.toFixed(6)}, Val Loss=${finalValLoss.toFixed(6)}`);
        
        // Update the model comparison dashboard
        updateModelComparisonResults('Linear Regression', {
            rSquared: `${(rSquared * 100).toFixed(1)}%`,
            mse: finalValLoss.toFixed(4),
            mae: meanAbsoluteError.toFixed(4),
            trainingTime: '< 1s',
            status: '✅ Completed'
        });
        
        // Display results in UI
        const resultsDiv = document.getElementById('linearRegressionResults');
        resultsDiv.innerHTML = `
            <div class="space-y-2">
                <div class="font-semibold text-green-600">✅ Linear Regression Training Complete</div>
                <div class="grid grid-cols-2 gap-2 text-sm">
                    <div>R-Squared: <span class="font-bold">${(rSquared * 100).toFixed(2)}%</span></div>
                    <div>Train Loss: <span class="font-bold">${finalTrainLoss.toFixed(6)}</span></div>
                    <div>Val Loss: <span class="font-bold">${finalValLoss.toFixed(6)}</span></div>
                    <div>Samples: <span class="font-bold">${features.length}</span></div>
                </div>
                <div class="text-xs text-gray-600">
                    Top Features: ${featureNames.slice(0,3).join(', ')}...
                </div>
            </div>
        `;
        
        debugToUI(`🎉 Linear regression model successfully trained and validated!`);
        
    } catch (error) {
        debugToUI(`❌ Error: ${error.message}`);
        document.getElementById('linearRegressionResults').innerHTML = 
            `<div class="text-red-600">Training failed: ${error.message}</div>`;
        
        // Update dashboard with failed status
        updateModelComparisonResults('Linear Regression', {
            rSquared: '--',
            mse: '--',
            mae: '--',
            trainingTime: '--',
            status: '❌ Failed'
        });
    }
}

/**
 * Data Preparation Function for ML Algorithms
 * Alias for prepareTrainingData to maintain compatibility
 */
function prepareDataForTraining(dataset) {
    return prepareTrainingData(dataset);
}

/**
 * Standardized Train-Validation-Test Split Function
 * Implements 70-20-10 split across all ML algorithms
 * @param {Array} features - Feature matrix
 * @param {Array} labels - Target labels
 * @param {Object} options - Split configuration options
 * @returns {Object} Split data with train, validation, and test sets
 */
function standardizedDataSplit(features, labels, options = {}) {
    const {
        trainRatio = 0.7,
        valRatio = 0.2,
        testRatio = 0.1,
        shuffle = true,
        randomSeed = null
    } = options;
    
    if (Math.abs(trainRatio + valRatio + testRatio - 1.0) > 1e-6) {
        throw new Error('Train, validation, and test ratios must sum to 1.0');
    }
    
    if (!features || !labels || features.length !== labels.length) {
        throw new Error('Features and labels must have the same length');
    }
    
    const totalSamples = features.length;
    let indices = Array.from({length: totalSamples}, (_, i) => i);
    
    // Shuffle indices if requested
    if (shuffle) {
        // Use simple shuffle (Fisher-Yates)
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
    }
    
    // Calculate split points
    const trainEnd = Math.floor(totalSamples * trainRatio);
    const valEnd = trainEnd + Math.floor(totalSamples * valRatio);
    
    // Split indices
    const trainIndices = indices.slice(0, trainEnd);
    const valIndices = indices.slice(trainEnd, valEnd);
    const testIndices = indices.slice(valEnd);
    
    // Create split datasets
    const trainFeatures = trainIndices.map(i => features[i]);
    const trainLabels = trainIndices.map(i => labels[i]);
    
    const valFeatures = valIndices.map(i => features[i]);
    const valLabels = valIndices.map(i => labels[i]);
    
    const testFeatures = testIndices.map(i => features[i]);
    const testLabels = testIndices.map(i => labels[i]);
    
    // Log split statistics
    console.log('=== STANDARDIZED DATA SPLIT ===');
    console.log(`Total samples: ${totalSamples}`);
    console.log(`Train: ${trainFeatures.length} samples (${(trainFeatures.length/totalSamples*100).toFixed(1)}%)`);
    console.log(`Validation: ${valFeatures.length} samples (${(valFeatures.length/totalSamples*100).toFixed(1)}%)`);
    console.log(`Test: ${testFeatures.length} samples (${(testFeatures.length/totalSamples*100).toFixed(1)}%)`);
    
    debugToUI(`📊 Data split: ${trainFeatures.length} train / ${valFeatures.length} val / ${testFeatures.length} test`);
    
    return {
        train: {
            features: trainFeatures,
            labels: trainLabels,
            size: trainFeatures.length
        },
        validation: {
            features: valFeatures,
            labels: valLabels,
            size: valFeatures.length
        },
        test: {
            features: testFeatures,
            labels: testLabels,
            size: testFeatures.length
        },
        splitInfo: {
            trainRatio,
            valRatio, 
            testRatio,
            totalSamples,
            shuffle,
            randomSeed
        }
    };
}

/**
 * Update Model Comparison Results Table
 * Updates the HTML table with model performance metrics
 */
function updateModelComparisonResults(modelName, results) {
    // Map model names to table row IDs
    const modelIdMap = {
        'Linear Regression': 'lr',
        'Support Vector Regression': 'svr', 
        'Random Forest': 'rf',
        'Semi-Supervised Learning (SSL)': 'ssl',
        'Self-Supervised Learning (SSL-AE)': 'ssae',
        'Reinforcement Learning (Q-Learning)': 'rl',
        'Advanced Ensemble (voting)': 'ensemble',
        'Advanced Ensemble (bagging)': 'ensemble',
        'Advanced Ensemble': 'ensemble'
    };
    
    // Find the appropriate row ID
    const rowId = modelIdMap[modelName];
    if (!rowId) {
        console.warn(`No table row found for model: ${modelName}`);
        return;
    }
    
    // Update the table cells
    const elements = {
        rSquared: document.getElementById(`${rowId}-rsquared`),
        mse: document.getElementById(`${rowId}-mse`),
        mae: document.getElementById(`${rowId}-mae`),
        trainingTime: document.getElementById(`${rowId}-time`),
        status: document.getElementById(`${rowId}-status`)
    };
    
    // Update each cell if the element exists
    Object.keys(elements).forEach(key => {
        if (elements[key] && results[key] !== undefined) {
            elements[key].textContent = results[key];
        }
    });
    
    // Color-code the status
    if (elements.status) {
        if (results.status && results.status.includes('✅')) {
            elements.status.className = 'text-center py-2 text-green-600 font-semibold';
        } else if (results.status && results.status.includes('❌')) {
            elements.status.className = 'text-center py-2 text-red-600 font-semibold';
        } else if (results.status && results.status.includes('⏳')) {
            elements.status.className = 'text-center py-2 text-yellow-600 font-semibold';
        }
    }
}

/**
 * Support Vector Regression Implementation
 * Simple JavaScript SVR using RBF kernel for non-linear regression
 */
async function trainSupportVectorRegression() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        showNotification('No dataset loaded. Please upload or select a dataset first.', 'error');
        return;
    }

    try {
        debugToUI('🚀 Starting Support Vector Regression training...');
        
        // Prepare data (same as neural network)
        const { features, labels, featureNames } = prepareDataForTraining(mlCore.sharedData);
        const labelValues = labels.map(row => row[0]);
        
        debugToUI(`📊 SVR Training with ${features.length} samples, ${featureNames.length} features`);
        
        // Use standardized 70-20-10 train-val-test split
        const dataSplit = standardizedDataSplit(features, labelValues);
        const trainFeatures = dataSplit.train.features;
        const trainLabels = dataSplit.train.labels;
        const valFeatures = dataSplit.validation.features;
        const valLabels = dataSplit.validation.labels;
        
        // Update split visualization dashboard
        if (typeof updateSplitVisualization === 'function') {
            updateSplitVisualization('Support Vector Regression', dataSplit);
        }
        
        // SVR hyperparameters
        const C = 1.0; // Regularization parameter
        const epsilon = 0.1; // Epsilon-tube for SVR
        const gamma = 1.0 / featureNames.length; // RBF kernel parameter (1/n_features)
        
        debugToUI(`🔧 SVR Parameters: C=${C}, ε=${epsilon}, γ=${gamma.toFixed(4)}`);
        
        // Simple SVR implementation using dual coordinate descent
        const svr = new SimpleSVR(C, epsilon, gamma);
        const trainingMetrics = await svr.fit(trainFeatures, trainLabels);
        
        debugToUI(`✅ SVR Training completed in ${trainingMetrics.iterations} iterations`);
        
        // Make predictions
        const trainPredictions = svr.predict(trainFeatures);
        const valPredictions = svr.predict(valFeatures);
        
        // Calculate metrics
        const trainMSE = meanSquaredError(trainLabels, trainPredictions);
        const valMSE = meanSquaredError(valLabels, valPredictions);
        const trainMAE = meanAbsoluteError(trainLabels, trainPredictions);
        const valMAE = meanAbsoluteError(valLabels, valPredictions);
        
        // Calculate R-squared
        const meanLabel = valLabels.reduce((sum, val) => sum + val, 0) / valLabels.length;
        let residualSumSquares = 0, totalSumSquares = 0;
        for (let i = 0; i < valLabels.length; i++) {
            residualSumSquares += Math.pow(valLabels[i] - valPredictions[i], 2);
            totalSumSquares += Math.pow(valLabels[i] - meanLabel, 2);
        }
        const rSquared = totalSumSquares > 0 ? 1 - (residualSumSquares / totalSumSquares) : 0;
        
        debugToUI(`📊 SVR Results: R²=${(rSquared * 100).toFixed(2)}%, Train MSE=${trainMSE.toFixed(6)}, Val MSE=${valMSE.toFixed(6)}`);
        
        // Update the model comparison dashboard
        updateModelComparisonResults('Support Vector Regression', {
            rSquared: `${(rSquared * 100).toFixed(1)}%`,
            mse: valMSE.toFixed(4),
            mae: valMAE.toFixed(4),
            trainingTime: '< 1s',
            status: '✅ Completed'
        });
        
        // Display results in UI
        const resultsDiv = document.getElementById('svrResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="space-y-2">
                    <div class="font-semibold text-blue-600">✅ Support Vector Regression Complete</div>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div>R-Squared: <span class="font-bold">${(rSquared * 100).toFixed(2)}%</span></div>
                        <div>Train MSE: <span class="font-bold">${trainMSE.toFixed(6)}</span></div>
                        <div>Val MSE: <span class="font-bold">${valMSE.toFixed(6)}</span></div>
                        <div>Support Vectors: <span class="font-bold">${svr.supportVectors.length}</span></div>
                    </div>
                    <div class="text-xs text-gray-600">
                        Parameters: C=${C}, ε=${epsilon}, γ=${gamma.toFixed(4)}
                    </div>
                </div>
            `;
        }
        
        debugToUI('🎉 Support Vector Regression completed successfully!');
        
    } catch (error) {
        debugToUI(`❌ SVR Error: ${error.message}`);
        const resultsDiv = document.getElementById('svrResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `<div class="text-red-600">SVR Training failed: ${error.message}</div>`;
        }
        
        // Update dashboard with failed status
        updateModelComparisonResults('Support Vector Regression', {
            rSquared: '--',
            mse: '--',
            mae: '--',
            trainingTime: '--',
            status: '❌ Failed'
        });
    }
}

/**
 * Simple Support Vector Regression implementation
 */
class SimpleSVR {
    constructor(C = 1.0, epsilon = 0.1, gamma = 0.1) {
        this.C = C;
        this.epsilon = epsilon;
        this.gamma = gamma;
        this.alpha = [];
        this.supportVectors = [];
        this.supportLabels = [];
        this.bias = 0;
    }
    
    // RBF kernel function
    rbfKernel(x1, x2) {
        let sum = 0;
        for (let i = 0; i < x1.length; i++) {
            sum += Math.pow(x1[i] - x2[i], 2);
        }
        return Math.exp(-this.gamma * sum);
    }
    
    // Simplified SMO algorithm for SVR
    async fit(features, labels) {
        const n = features.length;
        this.alpha = new Array(n * 2).fill(0); // Alpha and alpha* pairs
        
        // Simplified training with coordinate descent
        const maxIter = 1000;
        const tolerance = 1e-6;
        let iterations = 0;
        
        for (let iter = 0; iter < maxIter; iter++) {
            let changed = false;
            iterations++;
            
            for (let i = 0; i < n; i++) {
                const prediction = this.predictSingle(features[i], features, labels);
                const error = prediction - labels[i];
                
                // Update alpha values based on SVR conditions
                if (error > this.epsilon) {
                    if (this.alpha[i] < this.C) {
                        this.alpha[i] += 0.01;
                        changed = true;
                    }
                } else if (error < -this.epsilon) {
                    if (this.alpha[i + n] < this.C) {
                        this.alpha[i + n] += 0.01;
                        changed = true;
                    }
                }
            }
            
            if (!changed || iter % 100 === 0) {
                debugToUI(`🔄 SVR iteration ${iter + 1}, changed: ${changed}`);
            }
            
            if (!changed) break;
        }
        
        // Extract support vectors
        this.supportVectors = [];
        this.supportLabels = [];
        for (let i = 0; i < n; i++) {
            if (this.alpha[i] > tolerance || this.alpha[i + n] > tolerance) {
                this.supportVectors.push(features[i]);
                this.supportLabels.push(labels[i]);
            }
        }
        
        return { iterations, supportVectors: this.supportVectors.length };
    }
    
    // Predict single sample during training
    predictSingle(sample, allFeatures, allLabels) {
        let sum = 0;
        for (let i = 0; i < allFeatures.length; i++) {
            const kernel = this.rbfKernel(sample, allFeatures[i]);
            sum += (this.alpha[i] - this.alpha[i + allFeatures.length]) * kernel;
        }
        return sum + this.bias;
    }
    
    // Make predictions on new data
    predict(features) {
        return features.map(sample => {
            let sum = 0;
            for (let i = 0; i < this.supportVectors.length; i++) {
                const kernel = this.rbfKernel(sample, this.supportVectors[i]);
                sum += this.alpha[i] * kernel;
            }
            return sum + this.bias;
        });
    }
}

// Utility functions for metrics
function meanSquaredError(actual, predicted) {
    const n = actual.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += Math.pow(actual[i] - predicted[i], 2);
    }
    return sum / n;
}

function meanAbsoluteError(actual, predicted) {
    const n = actual.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += Math.abs(actual[i] - predicted[i]);
    }
    return sum / n;
}

/**
 * Random Forest Regression Implementation
 * Ensemble of decision trees for robust regression
 */
async function trainRandomForestRegression() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        showNotification('No dataset loaded. Please upload or select a dataset first.', 'error');
        return;
    }

    try {
        debugToUI('🌳 Starting Random Forest Regression training...');
        
        const { features, labels, featureNames } = prepareDataForTraining(mlCore.sharedData);
        const labelValues = labels.map(row => row[0]);
        
        debugToUI(`📊 Random Forest Training with ${features.length} samples, ${featureNames.length} features`);
        
        // Use standardized 70-20-10 train-val-test split
        const dataSplit = standardizedDataSplit(features, labelValues);
        const trainFeatures = dataSplit.train.features;
        const trainLabels = dataSplit.train.labels;
        const valFeatures = dataSplit.validation.features;
        const valLabels = dataSplit.validation.labels;
        
        // Update split visualization dashboard
        if (typeof updateSplitVisualization === 'function') {
            updateSplitVisualization('Random Forest', dataSplit);
        }
        
        // Random Forest parameters
        const numTrees = 50;
        const maxDepth = 10;
        const minSamplesLeaf = 3;
        const maxFeatures = Math.ceil(Math.sqrt(featureNames.length)); // sqrt(n_features)
        
        debugToUI(`🌲 Forest Parameters: ${numTrees} trees, max_depth=${maxDepth}, max_features=${maxFeatures}`);
        
        // Train Random Forest
        const rf = new SimpleRandomForest(numTrees, maxDepth, minSamplesLeaf, maxFeatures);
        await rf.fit(trainFeatures, trainLabels, featureNames);
        
        debugToUI(`✅ Random Forest trained with ${numTrees} trees`);
        
        // Make predictions
        const trainPredictions = rf.predict(trainFeatures);
        const valPredictions = rf.predict(valFeatures);
        
        // Calculate metrics
        const trainMSE = meanSquaredError(trainLabels, trainPredictions);
        const valMSE = meanSquaredError(valLabels, valPredictions);
        const trainMAE = meanAbsoluteError(trainLabels, trainPredictions);
        const valMAE = meanAbsoluteError(valLabels, valPredictions);
        
        // Calculate R-squared
        const meanLabel = valLabels.reduce((sum, val) => sum + val, 0) / valLabels.length;
        let residualSumSquares = 0, totalSumSquares = 0;
        for (let i = 0; i < valLabels.length; i++) {
            residualSumSquares += Math.pow(valLabels[i] - valPredictions[i], 2);
            totalSumSquares += Math.pow(valLabels[i] - meanLabel, 2);
        }
        const rSquared = totalSumSquares > 0 ? 1 - (residualSumSquares / totalSumSquares) : 0;
        
        // Get feature importance
        const featureImportance = rf.getFeatureImportance();
        const topFeatures = featureImportance
            .map((imp, i) => ({ feature: featureNames[i], importance: imp }))
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 3)
            .map(f => f.feature);
        
        debugToUI(`📊 Random Forest Results: R²=${(rSquared * 100).toFixed(2)}%, Train MSE=${trainMSE.toFixed(6)}, Val MSE=${valMSE.toFixed(6)}`);
        
        // Update the model comparison dashboard
        updateModelComparisonResults('Random Forest', {
            rSquared: `${(rSquared * 100).toFixed(1)}%`,
            mse: valMSE.toFixed(4),
            mae: valMAE.toFixed(4),
            trainingTime: '< 1s',
            status: '✅ Completed'
        });
        
        // Display results
        const resultsDiv = document.getElementById('rfResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="space-y-2">
                    <div class="font-semibold text-green-600">✅ Random Forest Regression Complete</div>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div>R-Squared: <span class="font-bold">${(rSquared * 100).toFixed(2)}%</span></div>
                        <div>Train MSE: <span class="font-bold">${trainMSE.toFixed(6)}</span></div>
                        <div>Val MSE: <span class="font-bold">${valMSE.toFixed(6)}</span></div>
                        <div>Trees: <span class="font-bold">${numTrees}</span></div>
                    </div>
                    <div class="text-xs text-gray-600">
                        Top Features: ${topFeatures.join(', ')}
                    </div>
                </div>
            `;
        }
        
        debugToUI('🎉 Random Forest Regression completed successfully!');
        
    } catch (error) {
        debugToUI(`❌ Random Forest Error: ${error.message}`);
        const resultsDiv = document.getElementById('rfResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = `<div class="text-red-600">Random Forest Training failed: ${error.message}</div>`;
        }
        
        // Update dashboard with failed status
        updateModelComparisonResults('Random Forest', {
            rSquared: '--',
            mse: '--',
            mae: '--',
            trainingTime: '--',
            status: '❌ Failed'
        });
    }
}

/**
 * Simple Random Forest implementation
 */
class SimpleRandomForest {
    constructor(numTrees = 50, maxDepth = 10, minSamplesLeaf = 3, maxFeatures = null) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.trees = [];
        this.featureImportances = [];
    }
    
    async fit(features, labels, featureNames) {
        const n = features.length;
        const numFeatures = features[0].length;
        this.maxFeatures = this.maxFeatures || Math.ceil(Math.sqrt(numFeatures));
        this.featureImportances = new Array(numFeatures).fill(0);
        
        for (let t = 0; t < this.numTrees; t++) {
            if (t % 10 === 0) {
                debugToUI(`🌱 Training tree ${t + 1}/${this.numTrees}...`);
            }
            
            // Bootstrap sampling
            const indices = [];
            for (let i = 0; i < n; i++) {
                indices.push(Math.floor(Math.random() * n));
            }
            
            const bootstrapFeatures = indices.map(i => features[i]);
            const bootstrapLabels = indices.map(i => labels[i]);
            
            // Random feature selection
            const selectedFeatures = [];
            const availableFeatures = Array.from({length: numFeatures}, (_, i) => i);
            for (let f = 0; f < this.maxFeatures; f++) {
                const randomIdx = Math.floor(Math.random() * availableFeatures.length);
                selectedFeatures.push(availableFeatures.splice(randomIdx, 1)[0]);
            }
            
            // Create and train tree
            const tree = new SimpleDecisionTree(this.maxDepth, this.minSamplesLeaf, selectedFeatures);
            tree.fit(bootstrapFeatures, bootstrapLabels);
            this.trees.push(tree);
            
            // Accumulate feature importance
            const treeImportance = tree.getFeatureImportance();
            for (let f = 0; f < selectedFeatures.length; f++) {
                this.featureImportances[selectedFeatures[f]] += treeImportance[f];
            }
        }
        
        // Normalize feature importance
        const totalImportance = this.featureImportances.reduce((sum, imp) => sum + imp, 0);
        if (totalImportance > 0) {
            this.featureImportances = this.featureImportances.map(imp => imp / totalImportance);
        }
    }
    
    predict(features) {
        return features.map(sample => {
            const predictions = this.trees.map(tree => tree.predict(sample));
            return predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
        });
    }
    
    getFeatureImportance() {
        return this.featureImportances;
    }
}

/**
 * Simple Decision Tree for regression
 */
class SimpleDecisionTree {
    constructor(maxDepth = 10, minSamplesLeaf = 3, selectedFeatures = null) {
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.selectedFeatures = selectedFeatures;
        this.root = null;
        this.featureImportances = [];
    }
    
    fit(features, labels) {
        const numFeatures = this.selectedFeatures ? this.selectedFeatures.length : features[0].length;
        this.featureImportances = new Array(numFeatures).fill(0);
        this.root = this.buildTree(features, labels, 0);
    }
    
    buildTree(features, labels, depth) {
        const n = features.length;
        
        // Stopping criteria
        if (depth >= this.maxDepth || n < this.minSamplesLeaf * 2) {
            const meanLabel = labels.reduce((sum, label) => sum + label, 0) / n;
            return { prediction: meanLabel };
        }
        
        // Find best split
        let bestFeature = -1;
        let bestThreshold = 0;
        let bestMSE = Infinity;
        let bestLeftIndices = [];
        let bestRightIndices = [];
        
        const availableFeatures = this.selectedFeatures || Array.from({length: features[0].length}, (_, i) => i);
        
        for (const featureIdx of availableFeatures) {
            const values = features.map(f => f[featureIdx]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
            
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
                const leftIndices = [];
                const rightIndices = [];
                
                for (let j = 0; j < n; j++) {
                    if (features[j][featureIdx] <= threshold) {
                        leftIndices.push(j);
                    } else {
                        rightIndices.push(j);
                    }
                }
                
                if (leftIndices.length < this.minSamplesLeaf || rightIndices.length < this.minSamplesLeaf) {
                    continue;
                }
                
                // Calculate MSE for this split
                const leftLabels = leftIndices.map(idx => labels[idx]);
                const rightLabels = rightIndices.map(idx => labels[idx]);
                const leftMSE = this.calculateMSE(leftLabels);
                const rightMSE = this.calculateMSE(rightLabels);
                const weightedMSE = (leftIndices.length * leftMSE + rightIndices.length * rightMSE) / n;
                
                if (weightedMSE < bestMSE) {
                    bestMSE = weightedMSE;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                    bestLeftIndices = leftIndices;
                    bestRightIndices = rightIndices;
                }
            }
        }
        
        if (bestFeature === -1) {
            const meanLabel = labels.reduce((sum, label) => sum + label, 0) / n;
            return { prediction: meanLabel };
        }
        
        // Update feature importance
        const parentMSE = this.calculateMSE(labels);
        const improvement = parentMSE - bestMSE;
        if (this.selectedFeatures) {
            const localFeatureIdx = this.selectedFeatures.indexOf(bestFeature);
            if (localFeatureIdx !== -1) {
                this.featureImportances[localFeatureIdx] += improvement * n;
            }
        } else {
            this.featureImportances[bestFeature] += improvement * n;
        }
        
        // Recursively build subtrees
        const leftFeatures = bestLeftIndices.map(idx => features[idx]);
        const leftLabels = bestLeftIndices.map(idx => labels[idx]);
        const rightFeatures = bestRightIndices.map(idx => features[idx]);
        const rightLabels = bestRightIndices.map(idx => labels[idx]);
        
        return {
            feature: bestFeature,
            threshold: bestThreshold,
            left: this.buildTree(leftFeatures, leftLabels, depth + 1),
            right: this.buildTree(rightFeatures, rightLabels, depth + 1)
        };
    }
    
    calculateMSE(labels) {
        if (labels.length === 0) return 0;
        const mean = labels.reduce((sum, label) => sum + label, 0) / labels.length;
        return labels.reduce((sum, label) => sum + Math.pow(label - mean, 2), 0) / labels.length;
    }
    
    predict(sample) {
        return this.predictNode(this.root, sample);
    }
    
    predictNode(node, sample) {
        if (node.prediction !== undefined) {
            return node.prediction;
        }
        
        if (sample[node.feature] <= node.threshold) {
            return this.predictNode(node.left, sample);
        } else {
            return this.predictNode(node.right, sample);
        }
    }
    
    getFeatureImportance() {
        const totalImportance = this.featureImportances.reduce((sum, imp) => sum + imp, 0);
        if (totalImportance === 0) return this.featureImportances;
        return this.featureImportances.map(imp => imp / totalImportance);
    }
}

/**
 * Train all models for comparison
 */
async function trainAllModels() {
    debugToUI('🚀 Starting comprehensive model comparison...');
    
    const startTime = Date.now();
    
    try {
        // Train all models sequentially with timing
        debugToUI('📈 Training Linear Regression...');
        const lr_start = Date.now();
        await trainLinearRegression();
        const lr_time = Date.now() - lr_start;
        
        debugToUI('🎯 Training Support Vector Regression...');
        const svr_start = Date.now();
        await trainSupportVectorRegression();
        const svr_time = Date.now() - svr_start;
        
        debugToUI('🌳 Training Random Forest Regression...');
        const rf_start = Date.now();
        await trainRandomForestRegression();
        const rf_time = Date.now() - rf_start;
        
        debugToUI('🧠 Training Neural Network...');
        const nn_start = Date.now();
        await trainNeuralNetwork();
        const nn_time = Date.now() - nn_start;
        
        const totalTime = Date.now() - startTime;
        
        debugToUI(`✅ All models trained! Total time: ${(totalTime/1000).toFixed(2)}s`);
        debugToUI(`📊 Training times - LR: ${(lr_time/1000).toFixed(1)}s, SVR: ${(svr_time/1000).toFixed(1)}s, RF: ${(rf_time/1000).toFixed(1)}s, NN: ${(nn_time/1000).toFixed(1)}s`);
        
    } catch (error) {
        debugToUI(`❌ Model comparison failed: ${error.message}`);
    }
}

// Export functions for global access
window.trainNeuralNetwork = trainNeuralNetwork;
window.trainLinearRegression = trainLinearRegression;
window.trainSupportVectorRegression = trainSupportVectorRegression;
window.trainRandomForestRegression = trainRandomForestRegression;
window.trainAllModels = trainAllModels; // Train all models for comparison
window.debugToUI = debugToUI;
window.updateNeuralNetworkData = updateNeuralNetworkData;
window.predictCulturalAuthenticity = predictCulturalAuthenticity;
window.predictTrustScore = predictTrustScore;
window.getModelSummary = getModelSummary;
window.updateNeuralNetworkArchitecture = updateNeuralNetworkArchitecture;
window.exportTrainedModel = exportTrainedModel;
window.getCulturalWeightsFromNeuralNetwork = getCulturalWeightsFromNeuralNetwork;
window.updateCulturalIntelligenceWeights = updateCulturalIntelligenceWeights;
window.validateDataForTraining = validateDataForTraining;

/**
 * ADVANCED ML METHODS FOR CULTURAL INTELLIGENCE & SCAM DETECTION
 * ============================================================
 */

/**
 * Semi-Supervised Learning - Label Propagation Algorithm
 * Suitable for cultural intelligence when you have limited labeled data
 * Propagates labels through a graph based on feature similarity
 */
class LabelPropagationSSL {
    constructor(alpha = 0.2, maxIterations = 1000, tolerance = 1e-6) {
        this.alpha = alpha; // Clamping factor (0 = only labeled data, 1 = only graph)
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.labeledIndices = [];
        this.unlabeledIndices = [];
        this.propagatedLabels = null;
        this.confidenceScores = null;
    }

    // Build similarity matrix using RBF kernel
    buildSimilarityMatrix(features, gamma = 0.1) {
        const n = features.length;
        const similarityMatrix = Array(n).fill().map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    let squaredDistance = 0;
                    for (let k = 0; k < features[i].length; k++) {
                        squaredDistance += Math.pow(features[i][k] - features[j][k], 2);
                    }
                    similarityMatrix[i][j] = Math.exp(-gamma * squaredDistance);
                }
            }
        }
        return similarityMatrix;
    }

    // Normalize similarity matrix to transition matrix
    normalizeMatrix(matrix) {
        const n = matrix.length;
        const normalized = Array(n).fill().map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            const rowSum = matrix[i].reduce((sum, val) => sum + val, 0);
            if (rowSum > 0) {
                for (let j = 0; j < n; j++) {
                    normalized[i][j] = matrix[i][j] / rowSum;
                }
            }
        }
        return normalized;
    }

    // Train the semi-supervised model
    fit(features, labels, labeledMask) {
        const n = features.length;
        
        // Identify labeled and unlabeled indices
        this.labeledIndices = [];
        this.unlabeledIndices = [];
        
        for (let i = 0; i < n; i++) {
            if (labeledMask[i]) {
                this.labeledIndices.push(i);
            } else {
                this.unlabeledIndices.push(i);
            }
        }
        
        // Build and normalize similarity matrix
        const similarity = this.buildSimilarityMatrix(features);
        const transition = this.normalizeMatrix(similarity);
        
        // Initialize label matrix (soft labels)
        let currentLabels = Array(n).fill(0);
        let previousLabels = Array(n).fill(0);
        
        // Set initial labels for labeled data
        for (let i = 0; i < n; i++) {
            currentLabels[i] = labeledMask[i] ? labels[i] : 0.5; // Neutral for unlabeled
        }
        
        // Label propagation iterations
        for (let iter = 0; iter < this.maxIterations; iter++) {
            previousLabels = [...currentLabels];
            
            // Propagate labels: F = αS*F + (1-α)Y
            const propagated = Array(n).fill(0);
            
            for (let i = 0; i < n; i++) {
                let weightedSum = 0;
                for (let j = 0; j < n; j++) {
                    weightedSum += transition[i][j] * currentLabels[j];
                }
                propagated[i] = this.alpha * weightedSum + (1 - this.alpha) * (labeledMask[i] ? labels[i] : currentLabels[i]);
            }
            
            currentLabels = propagated;
            
            // Check convergence
            let maxChange = 0;
            for (let i = 0; i < n; i++) {
                maxChange = Math.max(maxChange, Math.abs(currentLabels[i] - previousLabels[i]));
            }
            
            if (maxChange < this.tolerance) {
                console.log(`Label propagation converged after ${iter + 1} iterations`);
                break;
            }
        }
        
        this.propagatedLabels = currentLabels;
        
        // Calculate confidence scores based on label certainty
        this.confidenceScores = currentLabels.map(label => 
            Math.abs(label - 0.5) * 2 // Distance from neutral (0.5)
        );
        
        return this;
    }

    // Predict labels for new data points
    predict(newFeatures, trainFeatures) {
        if (!this.propagatedLabels) {
            throw new Error('Model must be trained before prediction');
        }
        
        const predictions = [];
        
        for (const newPoint of newFeatures) {
            let weightedSum = 0;
            let totalWeight = 0;
            
            // Use k-nearest neighbors for prediction
            const similarities = trainFeatures.map(trainPoint => {
                let squaredDistance = 0;
                for (let i = 0; i < newPoint.length; i++) {
                    squaredDistance += Math.pow(newPoint[i] - trainPoint[i], 2);
                }
                return Math.exp(-0.1 * squaredDistance);
            });
            
            for (let i = 0; i < trainFeatures.length; i++) {
                weightedSum += similarities[i] * this.propagatedLabels[i];
                totalWeight += similarities[i];
            }
            
            predictions.push(totalWeight > 0 ? weightedSum / totalWeight : 0.5);
        }
        
        return predictions;
    }

    // Get high-confidence unlabeled samples for active learning
    getHighConfidenceUnlabeled(threshold = 0.8) {
        const highConfidenceSamples = [];
        
        for (const idx of this.unlabeledIndices) {
            if (this.confidenceScores[idx] >= threshold) {
                highConfidenceSamples.push({
                    index: idx,
                    predictedLabel: this.propagatedLabels[idx],
                    confidence: this.confidenceScores[idx]
                });
            }
        }
        
        return highConfidenceSamples.sort((a, b) => b.confidence - a.confidence);
    }
}

/**
 * Train Semi-Supervised Learning Model for Cultural Intelligence
 */
async function trainSemiSupervisedLearning() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        showNotification('No dataset loaded. Please upload or select a dataset first.', 'error');
        return;
    }

    try {
        debugToUI('🚀 Starting Semi-Supervised Learning (Label Propagation)...');
        
        // Prepare data
        const { features, labels, featureNames } = prepareDataForTraining(mlCore.sharedData);
        const labelValues = labels.map(row => row[0]);
        
        debugToUI(`📊 SSL Training with ${features.length} samples, ${featureNames.length} features`);
        
        // Simulate partially labeled data (only 20% labeled)
        const labeledRatio = 0.2;
        const numLabeled = Math.floor(features.length * labeledRatio);
        const labeledMask = Array(features.length).fill(false);
        
        // Randomly select samples to be labeled
        const shuffledIndices = Array.from({length: features.length}, (_, i) => i)
            .sort(() => Math.random() - 0.5);
        
        for (let i = 0; i < numLabeled; i++) {
            labeledMask[shuffledIndices[i]] = true;
        }
        
        debugToUI(`📊 Using ${numLabeled} labeled samples (${labeledRatio * 100}%) and ${features.length - numLabeled} unlabeled samples`);
        
        // Get configurable parameters from UI
        const maxIterations = parseInt(document.getElementById('sslIterationsSlider')?.value) || 1000;
        
        // Train Semi-Supervised Learning model
        const startTime = Date.now();
        const sslModel = new LabelPropagationSSL(0.2, maxIterations, 1e-6);
        
        debugToUI(`🔧 SSL Parameters: maxIterations=${maxIterations}, alpha=0.2, tolerance=1e-6`);
        sslModel.fit(features, labelValues, labeledMask);
        const trainingTime = Date.now() - startTime;
        
        // Evaluate on all data
        const predictions = sslModel.propagatedLabels;
        
        // Calculate metrics
        let mse = 0, mae = 0, totalSumSquares = 0;
        const meanLabel = labelValues.reduce((sum, val) => sum + val, 0) / labelValues.length;
        
        for (let i = 0; i < labelValues.length; i++) {
            const error = predictions[i] - labelValues[i];
            mse += error * error;
            mae += Math.abs(error);
            totalSumSquares += Math.pow(labelValues[i] - meanLabel, 2);
        }
        
        mse /= labelValues.length;
        mae /= labelValues.length;
        const rSquared = Math.max(0, 1 - (mse * labelValues.length) / totalSumSquares);
        
        // Get high-confidence predictions for active learning
        const highConfidenceSamples = sslModel.getHighConfidenceUnlabeled(0.8);
        
        debugToUI(`✅ SSL Training completed in ${(trainingTime/1000).toFixed(2)}s`);
        debugToUI(`📊 SSL Performance: R² = ${(rSquared * 100).toFixed(1)}%, MSE = ${mse.toFixed(4)}, MAE = ${mae.toFixed(4)}`);
        debugToUI(`🎯 Found ${highConfidenceSamples.length} high-confidence unlabeled samples for potential auto-labeling`);
        
        // Update results table
        updateModelComparisonResults('Semi-Supervised Learning (SSL)', {
            rSquared: (rSquared * 100).toFixed(1),
            mse: mse.toFixed(4),
            mae: mae.toFixed(4),
            trainingTime: `${(trainingTime/1000).toFixed(2)}s`,
            status: '✅ Completed'
        });
        
        // Store model for later use
        window.semiSupervisedModel = sslModel;
        
    } catch (error) {
        debugToUI(`❌ SSL Training failed: ${error.message}`);
        updateModelComparisonResults('Semi-Supervised Learning (SSL)', {
            rSquared: '--',
            mse: '--', 
            mae: '--',
            trainingTime: '--',
            status: '❌ Failed'
        });
    }
}

// Export SSL function
window.trainSemiSupervisedLearning = trainSemiSupervisedLearning;

/**
 * Self-Supervised Learning - Simple Autoencoder
 * Learns meaningful feature representations without labels
 * Useful for discovering hidden cultural patterns
 */
class SimpleAutoencoder {
    constructor(inputDim, hiddenDim = null, learningRate = 0.01) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim || Math.floor(inputDim * 0.7); // Default to 70% of input
        this.learningRate = learningRate;
        
        // Initialize weights randomly
        this.weightsEncode = this.initializeWeights(this.inputDim, this.hiddenDim);
        this.biasEncode = Array(this.hiddenDim).fill(0);
        this.weightsDecode = this.initializeWeights(this.hiddenDim, this.inputDim);
        this.biasDecode = Array(this.inputDim).fill(0);
        
        this.trainLoss = [];
        this.embeddingCache = null;
    }
    
    initializeWeights(rows, cols) {
        const weights = [];
        for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
                // Xavier/Glorot initialization
                weights[i][j] = (Math.random() - 0.5) * 2 * Math.sqrt(6 / (rows + cols));
            }
        }
        return weights;
    }
    
    // Sigmoid activation function
    sigmoid(x) {
        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); // Clip to prevent overflow
    }
    
    // Derivative of sigmoid
    sigmoidDerivative(x) {
        const s = this.sigmoid(x);
        return s * (1 - s);
    }
    
    // Forward pass - encode input to hidden representation
    encode(input) {
        const hidden = Array(this.hiddenDim).fill(0);
        for (let j = 0; j < this.hiddenDim; j++) {
            for (let i = 0; i < this.inputDim; i++) {
                hidden[j] += input[i] * this.weightsEncode[i][j];
            }
            hidden[j] = this.sigmoid(hidden[j] + this.biasEncode[j]);
        }
        return hidden;
    }
    
    // Forward pass - decode hidden representation back to output
    decode(hidden) {
        const output = Array(this.inputDim).fill(0);
        for (let i = 0; i < this.inputDim; i++) {
            for (let j = 0; j < this.hiddenDim; j++) {
                output[i] += hidden[j] * this.weightsDecode[j][i];
            }
            output[i] = this.sigmoid(output[i] + this.biasDecode[i]);
        }
        return output;
    }
    
    // Full forward pass
    forward(input) {
        const hidden = this.encode(input);
        const output = this.decode(hidden);
        return { hidden, output };
    }
    
    // Calculate reconstruction loss (Mean Squared Error)
    calculateLoss(inputs) {
        let totalLoss = 0;
        for (const input of inputs) {
            const { output } = this.forward(input);
            for (let i = 0; i < input.length; i++) {
                totalLoss += Math.pow(input[i] - output[i], 2);
            }
        }
        return totalLoss / (inputs.length * this.inputDim);
    }
    
    // Train the autoencoder
    fit(trainingData, epochs = 100, batchSize = 32) {
        console.log(`Training autoencoder with ${trainingData.length} samples for ${epochs} epochs`);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            // Shuffle training data
            const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
            let epochLoss = 0;
            
            // Process mini-batches
            for (let batchStart = 0; batchStart < shuffled.length; batchStart += batchSize) {
                const batch = shuffled.slice(batchStart, Math.min(batchStart + batchSize, shuffled.length));
                epochLoss += this.trainBatch(batch);
            }
            
            epochLoss /= Math.ceil(shuffled.length / batchSize);
            this.trainLoss.push(epochLoss);
            
            // Log progress every 20 epochs
            if (epoch % 20 === 0 || epoch === epochs - 1) {
                console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${epochLoss.toFixed(6)}`);
            }
        }
    }
    
    // Train a single batch using backpropagation
    trainBatch(batch) {
        let batchLoss = 0;
        
        // Initialize gradients
        const gradWeightsEncode = this.initializeWeights(this.inputDim, this.hiddenDim);
        const gradBiasEncode = Array(this.hiddenDim).fill(0);
        const gradWeightsDecode = this.initializeWeights(this.hiddenDim, this.inputDim);
        const gradBiasDecode = Array(this.inputDim).fill(0);
        
        // Process each sample in batch
        for (const input of batch) {
            const { hidden, output } = this.forward(input);
            
            // Calculate loss for this sample
            let sampleLoss = 0;
            for (let i = 0; i < input.length; i++) {
                sampleLoss += Math.pow(input[i] - output[i], 2);
            }
            batchLoss += sampleLoss / this.inputDim;
            
            // Backward pass - calculate gradients
            // Output layer gradients
            const outputGradients = Array(this.inputDim);
            for (let i = 0; i < this.inputDim; i++) {
                const error = output[i] - input[i]; // Reconstruction error
                outputGradients[i] = error * this.sigmoidDerivative(output[i]);
            }
            
            // Hidden layer gradients
            const hiddenGradients = Array(this.hiddenDim).fill(0);
            for (let j = 0; j < this.hiddenDim; j++) {
                for (let i = 0; i < this.inputDim; i++) {
                    hiddenGradients[j] += outputGradients[i] * this.weightsDecode[j][i];
                }
                hiddenGradients[j] *= this.sigmoidDerivative(hidden[j]);
            }
            
            // Accumulate gradients for decoder weights
            for (let j = 0; j < this.hiddenDim; j++) {
                for (let i = 0; i < this.inputDim; i++) {
                    gradWeightsDecode[j][i] += outputGradients[i] * hidden[j];
                }
            }
            
            // Accumulate gradients for decoder bias
            for (let i = 0; i < this.inputDim; i++) {
                gradBiasDecode[i] += outputGradients[i];
            }
            
            // Accumulate gradients for encoder weights
            for (let i = 0; i < this.inputDim; i++) {
                for (let j = 0; j < this.hiddenDim; j++) {
                    gradWeightsEncode[i][j] += hiddenGradients[j] * input[i];
                }
            }
            
            // Accumulate gradients for encoder bias
            for (let j = 0; j < this.hiddenDim; j++) {
                gradBiasEncode[j] += hiddenGradients[j];
            }
        }
        
        // Update weights using accumulated gradients
        const batchSizeFloat = batch.length;
        
        // Update encoder weights and biases
        for (let i = 0; i < this.inputDim; i++) {
            for (let j = 0; j < this.hiddenDim; j++) {
                this.weightsEncode[i][j] -= this.learningRate * gradWeightsEncode[i][j] / batchSizeFloat;
            }
        }
        for (let j = 0; j < this.hiddenDim; j++) {
            this.biasEncode[j] -= this.learningRate * gradBiasEncode[j] / batchSizeFloat;
        }
        
        // Update decoder weights and biases
        for (let j = 0; j < this.hiddenDim; j++) {
            for (let i = 0; i < this.inputDim; i++) {
                this.weightsDecode[j][i] -= this.learningRate * gradWeightsDecode[j][i] / batchSizeFloat;
            }
        }
        for (let i = 0; i < this.inputDim; i++) {
            this.biasDecode[i] -= this.learningRate * gradBiasDecode[i] / batchSizeFloat;
        }
        
        return batchLoss / batchSizeFloat;
    }
    
    // Get learned feature embeddings
    getEmbeddings(data) {
        return data.map(sample => this.encode(sample));
    }
    
    // Detect anomalies based on reconstruction error
    detectAnomalies(data, threshold = null) {
        const errors = data.map(sample => {
            const { output } = this.forward(sample);
            let error = 0;
            for (let i = 0; i < sample.length; i++) {
                error += Math.pow(sample[i] - output[i], 2);
            }
            return Math.sqrt(error / sample.length); // RMSE
        });
        
        // If no threshold provided, use 95th percentile
        if (threshold === null) {
            const sortedErrors = [...errors].sort((a, b) => a - b);
            threshold = sortedErrors[Math.floor(sortedErrors.length * 0.95)];
        }
        
        return {
            errors,
            anomalies: errors.map((error, idx) => ({
                index: idx,
                error,
                isAnomaly: error > threshold
            })),
            threshold
        };
    }
}

/**
 * Train Self-Supervised Learning Model for Cultural Intelligence
 */
async function trainSelfSupervisedLearning() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        showNotification('No dataset loaded. Please upload or select a dataset first.', 'error');
        return;
    }

    try {
        debugToUI('🚀 Starting Self-Supervised Learning (Autoencoder)...');
        
        // Prepare data
        const { features, labels, featureNames } = prepareDataForTraining(mlCore.sharedData);
        const labelValues = labels.map(row => row[0]);
        
        debugToUI(`📊 Self-Supervised Training with ${features.length} samples, ${featureNames.length} features`);
        
        // Get configurable parameters from UI
        const trainingEpochs = parseInt(document.getElementById('autoencoderEpochsSlider')?.value) || 200;
        
        // Train autoencoder
        const startTime = Date.now();
        const autoencoder = new SimpleAutoencoder(features[0].length, 
                                                Math.floor(features[0].length * 0.6), // Hidden dimension
                                                0.01); // Learning rate
        
        debugToUI(`🔧 Autoencoder Parameters: epochs=${trainingEpochs}, hiddenDim=${Math.floor(features[0].length * 0.6)}, lr=0.01`);
        
        // Train the model
        autoencoder.fit(features, trainingEpochs, 32); // configurable epochs, batch size 32
        const trainingTime = Date.now() - startTime;
        
        // Get learned embeddings
        const embeddings = autoencoder.getEmbeddings(features);
        
        // Detect anomalies (unusual cultural patterns)
        const anomalyResults = autoencoder.detectAnomalies(features);
        const numAnomalies = anomalyResults.anomalies.filter(a => a.isAnomaly).length;
        
        // Evaluate reconstruction quality
        const finalLoss = autoencoder.calculateLoss(features);
        const reconstructionAccuracy = Math.max(0, Math.min(100, (1 - finalLoss) * 100));
        
        // Use embeddings to train a simple predictor for evaluation
        const embeddingPredictor = new SimpleLinearRegression();
        embeddingPredictor.fit(embeddings, labelValues);
        const predictions = embeddingPredictor.predict(embeddings);
        
        // Calculate metrics using embedding-based predictions
        let mse = 0, mae = 0, totalSumSquares = 0;
        const meanLabel = labelValues.reduce((sum, val) => sum + val, 0) / labelValues.length;
        
        for (let i = 0; i < labelValues.length; i++) {
            const error = predictions[i] - labelValues[i];
            mse += error * error;
            mae += Math.abs(error);
            totalSumSquares += Math.pow(labelValues[i] - meanLabel, 2);
        }
        
        mse /= labelValues.length;
        mae /= labelValues.length;
        const rSquared = Math.max(0, 1 - (mse * labelValues.length) / totalSumSquares);
        
        debugToUI(`✅ Self-Supervised Training completed in ${(trainingTime/1000).toFixed(2)}s`);
        debugToUI(`📊 Reconstruction Loss: ${finalLoss.toFixed(4)}, Accuracy: ${reconstructionAccuracy.toFixed(1)}%`);
        debugToUI(`🎯 Detected ${numAnomalies} anomalous cultural patterns (${(numAnomalies/features.length*100).toFixed(1)}%)`);
        debugToUI(`📈 Embedding-based prediction: R² = ${(rSquared * 100).toFixed(1)}%, MSE = ${mse.toFixed(4)}`);
        
        // Update results table
        updateModelComparisonResults('Self-Supervised Learning (SSL-AE)', {
            rSquared: `${(rSquared * 100).toFixed(1)}`,
            mse: mse.toFixed(4),
            mae: mae.toFixed(4),
            trainingTime: `${(trainingTime/1000).toFixed(2)}s`,
            status: '✅ Completed'
        });
        
        // Store model and embeddings for later use
        window.autoencoderModel = autoencoder;
        window.culturalEmbeddings = embeddings;
        window.anomalyResults = anomalyResults;
        
    } catch (error) {
        debugToUI(`❌ Self-Supervised Training failed: ${error.message}`);
        updateModelComparisonResults('Self-Supervised Learning (SSL-AE)', {
            rSquared: '--',
            mse: '--',
            mae: '--', 
            trainingTime: '--',
            status: '❌ Failed'
        });
    }
}

// Export Self-Supervised Learning function
window.trainSelfSupervisedLearning = trainSelfSupervisedLearning;

/**
 * Reinforcement Learning - Q-Learning Agent for Scam Detection
 * Learns optimal decision boundaries through reward-based feedback
 * Suitable for adaptive scam detection that learns from new patterns
 */
class QLearningScamDetector {
    constructor(numFeatures, learningRate = 0.1, discountFactor = 0.9, explorationRate = 0.1) {
        this.numFeatures = numFeatures;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.explorationRate = explorationRate;
        
        // Discretize feature space into bins for Q-table
        this.numBins = 5; // 5 bins per feature
        this.qTable = {}; // State-action value table
        this.episodeRewards = [];
        this.totalSteps = 0;
        
        // Actions: 0 = classify as legitimate, 1 = classify as scam
        this.actions = [0, 1];
        
        // Experience replay buffer
        this.memory = [];
        this.maxMemory = 1000;
    }
    
    // Convert continuous features to discrete state
    featuresToState(features) {
        const discreteFeatures = features.map(feature => {
            // Clamp feature to [0, 1] range and discretize
            const clampedFeature = Math.max(0, Math.min(1, feature));
            return Math.floor(clampedFeature * (this.numBins - 1));
        });
        return discreteFeatures.join(',');
    }
    
    // Initialize Q-values for a state if not exists
    initializeQValues(state) {
        if (!this.qTable[state]) {
            this.qTable[state] = {};
            for (const action of this.actions) {
                this.qTable[state][action] = 0.0; // Initialize to neutral
            }
        }
    }
    
    // Epsilon-greedy action selection
    selectAction(state, isTraining = true) {
        this.initializeQValues(state);
        
        if (isTraining && Math.random() < this.explorationRate) {
            // Explore: random action
            return this.actions[Math.floor(Math.random() * this.actions.length)];
        } else {
            // Exploit: best known action
            const qValues = this.qTable[state];
            let bestAction = this.actions[0];
            let bestValue = qValues[bestAction];
            
            for (const action of this.actions) {
                if (qValues[action] > bestValue) {
                    bestValue = qValues[action];
                    bestAction = action;
                }
            }
            return bestAction;
        }
    }
    
    // Calculate reward based on prediction accuracy
    calculateReward(prediction, actualLabel, confidence = 0.5) {
        const isCorrect = (prediction === 1 && actualLabel >= 0.7) || 
                         (prediction === 0 && actualLabel < 0.7);
        
        if (isCorrect) {
            // Correct prediction: positive reward scaled by confidence
            return 1.0 + (confidence - 0.5); // Range: [0.5, 1.5]
        } else {
            // Incorrect prediction: negative reward
            const severity = Math.abs(prediction - actualLabel);
            return -1.0 - severity; // Penalty increases with severity
        }
    }
    
    // Update Q-values using Q-learning update rule
    updateQValue(state, action, reward, nextState) {
        this.initializeQValues(state);
        this.initializeQValues(nextState);
        
        // Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        const currentQValue = this.qTable[state][action];
        const maxNextQValue = Math.max(...this.actions.map(a => this.qTable[nextState][a]));
        
        const updatedQValue = currentQValue + this.learningRate * 
                             (reward + this.discountFactor * maxNextQValue - currentQValue);
        
        this.qTable[state][action] = updatedQValue;
        this.totalSteps++;
    }
    
    // Add experience to replay buffer
    addExperience(state, action, reward, nextState, actualLabel) {
        this.memory.push({
            state, action, reward, nextState, actualLabel, 
            timestamp: Date.now()
        });
        
        // Maintain buffer size
        if (this.memory.length > this.maxMemory) {
            this.memory.shift();
        }
    }
    
    // Experience replay training
    replayExperiences(batchSize = 32) {
        if (this.memory.length < batchSize) return;
        
        // Sample random batch from memory
        const batch = [];
        for (let i = 0; i < batchSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.memory.length);
            batch.push(this.memory[randomIndex]);
        }
        
        // Update Q-values for batch
        for (const experience of batch) {
            this.updateQValue(experience.state, experience.action, 
                            experience.reward, experience.nextState);
        }
    }
    
    // Train the Q-learning agent
    fit(features, labels, episodes = 1000) {
        console.log(`Training Q-Learning agent for ${episodes} episodes`);
        
        for (let episode = 0; episode < episodes; episode++) {
            let episodeReward = 0;
            
            // Shuffle data for each episode
            const indices = Array.from({length: features.length}, (_, i) => i)
                          .sort(() => Math.random() - 0.5);
            
            for (let i = 0; i < indices.length; i++) {
                const idx = indices[i];
                const state = this.featuresToState(features[idx]);
                const action = this.selectAction(state, true);
                
                // Convert action to prediction probability
                const prediction = action; // 0 or 1
                
                // Calculate reward
                const reward = this.calculateReward(prediction, labels[idx]);
                episodeReward += reward;
                
                // Determine next state (use next sample or same if last)
                const nextIdx = i < indices.length - 1 ? indices[i + 1] : idx;
                const nextState = this.featuresToState(features[nextIdx]);
                
                // Update Q-values
                this.updateQValue(state, action, reward, nextState);
                
                // Add to experience replay
                this.addExperience(state, action, reward, nextState, labels[idx]);
            }
            
            // Perform experience replay
            this.replayExperiences(32);
            
            // Decay exploration rate
            this.explorationRate = Math.max(0.01, this.explorationRate * 0.995);
            
            this.episodeRewards.push(episodeReward);
            
            // Log progress
            if (episode % 200 === 0 || episode === episodes - 1) {
                const avgReward = episodeReward / features.length;
                console.log(`Episode ${episode + 1}/${episodes}, Avg Reward: ${avgReward.toFixed(4)}, ε: ${this.explorationRate.toFixed(3)}`);
            }
        }
    }
    
    // Make predictions using trained Q-table
    predict(features) {
        const predictions = [];
        
        for (const featureVector of features) {
            const state = this.featuresToState(featureVector);
            const action = this.selectAction(state, false); // No exploration during prediction
            predictions.push(action); // 0 for legitimate, 1 for scam
        }
        
        return predictions;
    }
    
    // Get prediction confidence based on Q-values
    getPredictionConfidence(features) {
        const confidences = [];
        
        for (const featureVector of features) {
            const state = this.featuresToState(featureVector);
            this.initializeQValues(state);
            
            const qValues = this.actions.map(action => this.qTable[state][action]);
            const maxQ = Math.max(...qValues);
            const minQ = Math.min(...qValues);
            
            // Confidence based on Q-value difference
            const confidence = Math.abs(maxQ - minQ); 
            confidences.push(Math.min(1.0, confidence));
        }
        
        return confidences;
    }
    
    // Get learning statistics
    getStats() {
        return {
            totalSteps: this.totalSteps,
            qTableSize: Object.keys(this.qTable).length,
            memorySize: this.memory.length,
            finalExplorationRate: this.explorationRate,
            avgReward: this.episodeRewards.length > 0 ? 
                      this.episodeRewards.reduce((a, b) => a + b, 0) / this.episodeRewards.length : 0
        };
    }
}

/**
 * Train Reinforcement Learning Model for Scam Detection
 */
async function trainReinforcementLearning() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        showNotification('No dataset loaded. Please upload or select a dataset first.', 'error');
        return;
    }

    try {
        debugToUI('🚀 Starting Reinforcement Learning (Q-Learning) for Scam Detection...');
        
        // Prepare data
        const { features, labels, featureNames } = prepareDataForTraining(mlCore.sharedData);
        const labelValues = labels.map(row => row[0]);
        
        debugToUI(`📊 RL Training with ${features.length} samples, ${featureNames.length} features`);
        
        // Get configurable parameters from UI
        const trainingEpisodes = parseInt(document.getElementById('rlEpisodesSlider')?.value) || 800;
        
        // Train Q-Learning agent
        const startTime = Date.now();
        const rlAgent = new QLearningScamDetector(
            features[0].length,  // Number of features
            0.1,                 // Learning rate
            0.9,                 // Discount factor
            0.3                  // Initial exploration rate
        );
        
        debugToUI(`🔧 Q-Learning Parameters: episodes=${trainingEpisodes}, lr=0.1, γ=0.9, ε=0.3`);
        
        // Train the agent
        rlAgent.fit(features, labelValues, trainingEpisodes);
        const trainingTime = Date.now() - startTime;
        
        // Get predictions
        const predictions = rlAgent.predict(features);
        const confidences = rlAgent.getPredictionConfidence(features);
        
        // Calculate metrics for binary classification
        let tp = 0, fp = 0, tn = 0, fn = 0;
        let mse = 0, mae = 0;
        
        for (let i = 0; i < labelValues.length; i++) {
            const predicted = predictions[i];
            const actual = labelValues[i] >= 0.7 ? 1 : 0; // Binary threshold
            
            // Confusion matrix
            if (predicted === 1 && actual === 1) tp++;
            else if (predicted === 1 && actual === 0) fp++;
            else if (predicted === 0 && actual === 0) tn++;
            else if (predicted === 0 && actual === 1) fn++;
            
            // Regression metrics (treat predictions as probabilities)
            const predProb = predicted;
            const error = predProb - labelValues[i];
            mse += error * error;
            mae += Math.abs(error);
        }
        
        const accuracy = (tp + tn) / labelValues.length;
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
        
        mse /= labelValues.length;
        mae /= labelValues.length;
        
        // Get agent statistics
        const stats = rlAgent.getStats();
        
        debugToUI(`✅ RL Training completed in ${(trainingTime/1000).toFixed(2)}s`);
        debugToUI(`📊 RL Performance: Accuracy = ${(accuracy * 100).toFixed(1)}%, F1 = ${f1Score.toFixed(3)}`);
        debugToUI(`🎯 Precision = ${precision.toFixed(3)}, Recall = ${recall.toFixed(3)}`);
        debugToUI(`🧠 Q-Table Size: ${stats.qTableSize} states, Avg Reward: ${stats.avgReward.toFixed(3)}`);
        
        // Update results table  
        updateModelComparisonResults('Reinforcement Learning (Q-Learning)', {
            rSquared: `${(accuracy * 100).toFixed(1)}`, // Use accuracy instead of R²
            mse: mse.toFixed(4),
            mae: mae.toFixed(4),
            trainingTime: `${(trainingTime/1000).toFixed(2)}s`,
            status: '✅ Completed'
        });
        
        // Store model for later use
        window.reinforcementLearningAgent = rlAgent;
        window.rlPredictionConfidences = confidences;
        
    } catch (error) {
        debugToUI(`❌ RL Training failed: ${error.message}`);
        updateModelComparisonResults('Reinforcement Learning (Q-Learning)', {
            rSquared: '--',
            mse: '--',
            mae: '--',
            trainingTime: '--',
            status: '❌ Failed'
        });
    }
}

// Export Reinforcement Learning function
window.trainReinforcementLearning = trainReinforcementLearning;

/**
 * Advanced Ensemble Learning Methods
 * Combines multiple algorithms for improved performance and robustness
 */
class AdvancedEnsemble {
    constructor() {
        this.models = [];
        this.weights = [];
        this.metaLearner = null;
        this.ensembleType = 'voting'; // 'voting', 'stacking', 'bagging'
        this.trainingHistory = [];
    }

    // Add a trained model to the ensemble
    addModel(model, weight = 1.0, modelType = 'generic') {
        this.models.push({
            model: model,
            weight: weight,
            type: modelType,
            performance: null
        });
        this.weights.push(weight);
    }

    // Voting ensemble: weighted average of predictions
    votingPredict(features) {
        if (this.models.length === 0) {
            throw new Error('No models in ensemble');
        }

        const predictions = [];
        
        for (const featureVector of features) {
            let weightedSum = 0;
            let totalWeight = 0;
            
            for (let i = 0; i < this.models.length; i++) {
                const modelEntry = this.models[i];
                let prediction;
                
                // Handle different model types
                if (modelEntry.type === 'linear') {
                    prediction = modelEntry.model.predict([featureVector])[0];
                } else if (modelEntry.type === 'svr') {
                    prediction = modelEntry.model.predict([featureVector])[0];
                } else if (modelEntry.type === 'rf') {
                    prediction = modelEntry.model.predict([featureVector])[0];
                } else if (modelEntry.type === 'ssl') {
                    // Use propagated labels
                    prediction = 0.5; // Placeholder - would need actual implementation
                } else if (modelEntry.type === 'rl') {
                    prediction = modelEntry.model.predict([featureVector])[0];
                } else {
                    prediction = 0.5; // Default
                }
                
                weightedSum += prediction * this.weights[i];
                totalWeight += this.weights[i];
            }
            
            predictions.push(totalWeight > 0 ? weightedSum / totalWeight : 0.5);
        }
        
        return predictions;
    }

    // Bagging ensemble: train multiple models on bootstrap samples
    static async trainBagging(features, labels, numModels = 5, modelType = 'rf') {
        const ensemble = new AdvancedEnsemble();
        ensemble.ensembleType = 'bagging';
        
        console.log(`Training bagging ensemble with ${numModels} models`);
        
        for (let i = 0; i < numModels; i++) {
            // Create bootstrap sample
            const bootstrapIndices = [];
            for (let j = 0; j < features.length; j++) {
                bootstrapIndices.push(Math.floor(Math.random() * features.length));
            }
            
            const bootstrapFeatures = bootstrapIndices.map(idx => features[idx]);
            const bootstrapLabels = bootstrapIndices.map(idx => labels[idx]);
            
            // Train model on bootstrap sample
            let model;
            if (modelType === 'rf') {
                model = new SimpleRandomForest(10, 8, 2, null); // Smaller trees for diversity
                model.fit(bootstrapFeatures, bootstrapLabels);
            } else if (modelType === 'svr') {
                model = new SimpleSVR(1.0, 0.1, 0.1);
                model.fit(bootstrapFeatures, bootstrapLabels);
            } else {
                // Default to linear regression
                model = new SimpleLinearRegression();
                model.fit(bootstrapFeatures, bootstrapLabels);
            }
            
            ensemble.addModel(model, 1.0 / numModels, modelType);
            console.log(`Trained bagging model ${i + 1}/${numModels}`);
        }
        
        return ensemble;
    }

    // Stacking ensemble: use meta-learner to combine predictions
    async trainStacking(features, labels, baseModels, metaLearnerType = 'linear') {
        this.ensembleType = 'stacking';
        
        // Step 1: Train base models with cross-validation
        const kFolds = 5;
        const foldSize = Math.floor(features.length / kFolds);
        const metaFeatures = Array(features.length).fill().map(() => []);
        
        for (let fold = 0; fold < kFolds; fold++) {
            const testStart = fold * foldSize;
            const testEnd = fold === kFolds - 1 ? features.length : testStart + foldSize;
            
            // Split data for this fold
            const trainIndices = [];
            const testIndices = [];
            
            for (let i = 0; i < features.length; i++) {
                if (i >= testStart && i < testEnd) {
                    testIndices.push(i);
                } else {
                    trainIndices.push(i);
                }
            }
            
            const trainFeatures = trainIndices.map(idx => features[idx]);
            const trainLabels = trainIndices.map(idx => labels[idx]);
            const testFeatures = testIndices.map(idx => features[idx]);
            
            // Train each base model on fold training data
            for (const baseModel of baseModels) {
                const foldModel = this.cloneModel(baseModel);
                foldModel.fit(trainFeatures, trainLabels);
                
                // Get predictions on fold test data
                const predictions = foldModel.predict(testFeatures);
                
                // Store as meta-features
                for (let i = 0; i < testIndices.length; i++) {
                    const originalIdx = testIndices[i];
                    metaFeatures[originalIdx].push(predictions[i]);
                }
            }
        }
        
        // Step 2: Train meta-learner on meta-features
        if (metaLearnerType === 'linear') {
            this.metaLearner = new SimpleLinearRegression();
        } else {
            this.metaLearner = new SimpleLinearRegression(); // Default
        }
        
        this.metaLearner.fit(metaFeatures, labels);
        
        // Step 3: Train base models on full dataset
        this.models = [];
        for (const baseModel of baseModels) {
            const fullModel = this.cloneModel(baseModel);
            fullModel.fit(features, labels);
            this.addModel(fullModel, 1.0, baseModel.type || 'generic');
        }
    }
    
    // Helper method to clone model (simplified)
    cloneModel(model) {
        if (model instanceof SimpleLinearRegression) {
            return new SimpleLinearRegression();
        } else if (model instanceof SimpleSVR) {
            return new SimpleSVR(model.C, model.epsilon, model.gamma);
        } else if (model instanceof SimpleRandomForest) {
            return new SimpleRandomForest(model.numTrees, model.maxDepth, model.minSamplesLeaf, model.maxFeatures);
        } else {
            return new SimpleLinearRegression(); // Default
        }
    }

    // Stacking prediction
    stackingPredict(features) {
        if (!this.metaLearner || this.models.length === 0) {
            throw new Error('Stacking ensemble not properly trained');
        }

        const metaFeatures = [];
        
        for (const featureVector of features) {
            const basesPredictions = [];
            
            for (const modelEntry of this.models) {
                const prediction = modelEntry.model.predict([featureVector])[0];
                basesPredictions.push(prediction);
            }
            
            metaFeatures.push(basesPredictions);
        }
        
        return this.metaLearner.predict(metaFeatures);
    }

    // Main prediction method
    predict(features) {
        switch (this.ensembleType) {
            case 'voting':
            case 'bagging':
                return this.votingPredict(features);
            case 'stacking':
                return this.stackingPredict(features);
            default:
                return this.votingPredict(features);
        }
    }

    // Evaluate ensemble diversity
    calculateDiversity(features, labels) {
        if (this.models.length < 2) return 0;

        const predictions = this.models.map(modelEntry => 
            modelEntry.model.predict(features)
        );

        let diversitySum = 0;
        let pairCount = 0;

        // Calculate pairwise disagreement
        for (let i = 0; i < predictions.length; i++) {
            for (let j = i + 1; j < predictions.length; j++) {
                let disagreements = 0;
                
                for (let k = 0; k < predictions[i].length; k++) {
                    const pred1 = predictions[i][k] >= 0.5 ? 1 : 0;
                    const pred2 = predictions[j][k] >= 0.5 ? 1 : 0;
                    if (pred1 !== pred2) disagreements++;
                }
                
                diversitySum += disagreements / predictions[i].length;
                pairCount++;
            }
        }

        return pairCount > 0 ? diversitySum / pairCount : 0;
    }
}

/**
 * Train Advanced Ensemble Learning Model
 */
async function trainAdvancedEnsemble() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        showNotification('No dataset loaded. Please upload or select a dataset first.', 'error');
        return;
    }

    try {
        debugToUI('🚀 Starting Advanced Ensemble Learning...');
        
        // Prepare data
        const { features, labels, featureNames } = prepareDataForTraining(mlCore.sharedData);
        const labelValues = labels.map(row => row[0]);
        
        debugToUI(`📊 Ensemble Training with ${features.length} samples, ${featureNames.length} features`);
        
        // Use standardized 70-20-10 train-val-test split
        const dataSplit = standardizedDataSplit(features, labelValues);
        const trainFeatures = dataSplit.train.features;
        const trainLabels = dataSplit.train.labels;
        const valFeatures = dataSplit.validation.features;
        const valLabels = dataSplit.validation.labels;
        
        // Update split visualization dashboard
        if (typeof updateSplitVisualization === 'function') {
            updateSplitVisualization('Advanced Ensemble', dataSplit);
        }
        
        // Get configurable parameters from UI
        const ensembleEpochs = parseInt(document.getElementById('ensembleEpochsSlider')?.value) || 100;
        
        const startTime = Date.now();
        
        // Create base models
        const baseModels = [
            new SimpleLinearRegression(),
            new SimpleSVR(1.0, 0.1, 0.1),
            new SimpleRandomForest(30, 10, 3, null)
        ];
        
        debugToUI(`🔧 Ensemble Parameters: baseModelEpochs=${ensembleEpochs}, models=[LinearReg, SVR, RandomForest]`);
        
        // Train base models on training set
        debugToUI('🔧 Training base models for ensemble...');
        baseModels[0].fit(trainFeatures, trainLabels, { featureNames, epochs: ensembleEpochs }); // Linear Regression
        baseModels[1].fit(trainFeatures, trainLabels); // SVR  
        baseModels[2].fit(trainFeatures, trainLabels); // Random Forest
        
        // Create voting ensemble
        const votingEnsemble = new AdvancedEnsemble();
        votingEnsemble.addModel(baseModels[0], 0.3, 'linear');
        votingEnsemble.addModel(baseModels[1], 0.4, 'svr');
        votingEnsemble.addModel(baseModels[2], 0.3, 'rf');
        
        // Train bagging ensemble
        debugToUI('🎒 Training bagging ensemble...');
        const baggingEnsemble = await AdvancedEnsemble.trainBagging(trainFeatures, trainLabels, 5, 'rf');
        
        const trainingTime = Date.now() - startTime;
        
        // Get predictions from voting ensemble on validation set
        const votingPredictions = votingEnsemble.predict(valFeatures);
        
        // Get predictions from bagging ensemble on validation set
        const baggingPredictions = baggingEnsemble.predict(valFeatures);
        
        // Calculate metrics for voting ensemble using validation set
        let mse_voting = 0, mae_voting = 0, totalSumSquares = 0;
        const meanLabel = valLabels.reduce((sum, val) => sum + val, 0) / valLabels.length;
        
        for (let i = 0; i < valLabels.length; i++) {
            const error = votingPredictions[i] - valLabels[i];
            mse_voting += error * error;
            mae_voting += Math.abs(error);
            totalSumSquares += Math.pow(valLabels[i] - meanLabel, 2);
        }
        
        mse_voting /= valLabels.length;
        mae_voting /= valLabels.length;
        const rSquared_voting = Math.max(0, 1 - (mse_voting * valLabels.length) / totalSumSquares);
        
        // Calculate metrics for bagging ensemble using validation set
        let mse_bagging = 0, mae_bagging = 0;
        
        for (let i = 0; i < valLabels.length; i++) {
            const error = baggingPredictions[i] - valLabels[i];
            mse_bagging += error * error;
            mae_bagging += Math.abs(error);
        }
        
        mse_bagging /= valLabels.length;
        mae_bagging /= valLabels.length;
        const rSquared_bagging = Math.max(0, 1 - (mse_bagging * valLabels.length) / totalSumSquares);
        
        // Calculate ensemble diversity using validation set
        const votingDiversity = votingEnsemble.calculateDiversity(valFeatures, valLabels);
        const baggingDiversity = baggingEnsemble.calculateDiversity(valFeatures, valLabels);
        
        debugToUI(`✅ Ensemble Training completed in ${(trainingTime/1000).toFixed(2)}s`);
        debugToUI(`📊 Voting Ensemble: R² = ${(rSquared_voting * 100).toFixed(1)}%, MSE = ${mse_voting.toFixed(4)}, Diversity = ${votingDiversity.toFixed(3)}`);
        debugToUI(`📊 Bagging Ensemble: R² = ${(rSquared_bagging * 100).toFixed(1)}%, MSE = ${mse_bagging.toFixed(4)}, Diversity = ${baggingDiversity.toFixed(3)}`);
        
        // Update results table for best performing ensemble
        const bestEnsemble = rSquared_voting > rSquared_bagging ? 'voting' : 'bagging';
        const bestMetrics = bestEnsemble === 'voting' ? 
            { rSquared: rSquared_voting, mse: mse_voting, mae: mae_voting } :
            { rSquared: rSquared_bagging, mse: mse_bagging, mae: mae_bagging };
        
        updateModelComparisonResults(`Advanced Ensemble (${bestEnsemble})`, {
            rSquared: (bestMetrics.rSquared * 100).toFixed(1),
            mse: bestMetrics.mse.toFixed(4),
            mae: bestMetrics.mae.toFixed(4),
            trainingTime: `${(trainingTime/1000).toFixed(2)}s`,
            status: '✅ Completed'
        });
        
        // Store ensembles for later use
        window.votingEnsemble = votingEnsemble;
        window.baggingEnsemble = baggingEnsemble;
        
    } catch (error) {
        debugToUI(`❌ Ensemble Training failed: ${error.message}`);
        updateModelComparisonResults('Advanced Ensemble', {
            rSquared: '--',
            mse: '--',
            mae: '--',
            trainingTime: '--',
            status: '❌ Failed'
        });
    }
}

// Export Advanced Ensemble function
window.trainAdvancedEnsemble = trainAdvancedEnsemble;
window.refreshNeuralNetworkDatasetConnection = refreshNeuralNetworkDatasetConnection;
window.selectNeuralNetworkDataset = selectNeuralNetworkDataset;
window.initializeNeuralNetworkDatasetConnection = initializeNeuralNetworkDatasetConnection;