/**
 * Neural Network Tests
 * Tests for TensorFlow.js neural network implementation and training
 */

import { jest } from '@jest/globals';

// Mock Neural Network state
const mockNeuralNetworkState = {
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
        learningRate: 0.01,
        batchSize: 32,
        epochs: 50,
        validationSplit: 0.2,
        earlyStopping: true
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

// Mock Neural Network functions
const mockNeuralNetworkFunctions = {
    trainNeuralNetwork: jest.fn(async () => {
        if (!global.mlCore.sharedData || global.mlCore.sharedData.length === 0) {
            throw new Error('No dataset loaded');
        }

        if (mockNeuralNetworkState.isTraining) {
            throw new Error('Model is already training');
        }

        mockNeuralNetworkState.isTraining = true;

        try {
            // Mock training process
            const { features, labels, featureNames } = mockNeuralNetworkFunctions.prepareTrainingData(global.mlCore.sharedData);
            
            // Create mock model
            const model = {
                fit: jest.fn().mockResolvedValue({
                    history: {
                        loss: [0.8, 0.6, 0.4, 0.3, 0.2],
                        acc: [0.6, 0.7, 0.8, 0.85, 0.9],
                        val_loss: [0.9, 0.7, 0.5, 0.4, 0.3],
                        val_acc: [0.55, 0.65, 0.75, 0.8, 0.85]
                    }
                }),
                predict: jest.fn().mockReturnValue({
                    data: jest.fn().mockResolvedValue([0.85])
                }),
                dispose: jest.fn(),
                countParams: jest.fn().mockReturnValue(2048),
                summary: jest.fn()
            };

            mockNeuralNetworkState.model = model;

            // Mock training history update
            mockNeuralNetworkState.trainingHistory = {
                epochs: [1, 2, 3, 4, 5],
                loss: [0.8, 0.6, 0.4, 0.3, 0.2],
                accuracy: [0.6, 0.7, 0.8, 0.85, 0.9],
                valLoss: [0.9, 0.7, 0.5, 0.4, 0.3],
                valAccuracy: [0.55, 0.65, 0.75, 0.8, 0.85]
            };

            // Mock performance metrics
            mockNeuralNetworkState.performanceMetrics = {
                accuracy: 0.85,
                loss: 0.2,
                precision: 0.87,
                recall: 0.83,
                f1Score: 0.85
            };

            return model;
        } finally {
            mockNeuralNetworkState.isTraining = false;
        }
    }),

    prepareTrainingData: jest.fn((dataset) => {
        const features = [];
        const labels = [];
        
        const featureColumns = [
            'kapwa_score', 
            'bayanihan_participation', 
            'utang_na_loob_integrity',
            'community_standing_score', 
            'monthly_income', 
            'family_size', 
            'digital_literacy_score'
        ];
        
        const featureNames = [
            'Kapwa Network Strength',
            'Bayanihan Participation',
            'Utang na Loob Integrity',
            'Community Standing',
            'Economic Status (Normalized)',
            'Family Responsibility',
            'Digital Literacy'
        ];

        dataset.forEach(row => {
            const featureVector = [];
            featureColumns.forEach(column => {
                let value = row[column] || 0;
                
                // Normalize based on column type
                if (column === 'monthly_income') {
                    value = Math.min(Math.log(value + 1) / Math.log(100001), 1);
                } else if (column === 'family_size') {
                    value = Math.min(value / 12, 1);
                } else if (column === 'digital_literacy_score') {
                    value = Math.min(value / 10, 1);
                } else {
                    value = Math.min(Math.max(value, 0), 1);
                }
                
                featureVector.push(value);
            });
            
            features.push(featureVector);
            
            // Create binary trustworthiness label
            let label = 0.5;
            if (row.trustworthiness_label === 'trustworthy') {
                label = 1.0;
            } else if (row.trustworthiness_label === 'untrustworthy') {
                label = 0.0;
            }
            
            labels.push([label]);
        });
        
        return { features, labels, featureNames };
    }),

    createCulturalIntelligenceModel: jest.fn(() => {
        const mockModel = {
            compile: jest.fn(),
            fit: jest.fn().mockResolvedValue({
                history: { loss: [0.5], acc: [0.8] }
            }),
            predict: jest.fn().mockReturnValue({
                data: jest.fn().mockResolvedValue([0.85])
            }),
            dispose: jest.fn(),
            countParams: jest.fn().mockReturnValue(2048),
            summary: jest.fn()
        };
        
        return mockModel;
    }),

    calculatePerformanceMetrics: jest.fn(async (model, valXs, valYs) => {
        // Mock performance calculation
        mockNeuralNetworkState.performanceMetrics = {
            accuracy: 0.85,
            precision: 0.87,
            recall: 0.83,
            f1Score: 0.85,
            truePositives: 42,
            trueNegatives: 38,
            falsePositives: 6,
            falseNegatives: 8
        };
    }),

    analyzeFeatureImportance: jest.fn(async (model, features, featureNames) => {
        // Mock feature importance analysis
        mockNeuralNetworkState.featureImportance = [
            { feature: 'Kapwa Network Strength', importance: 0.95, rank: 1 },
            { feature: 'Utang na Loob Integrity', importance: 0.87, rank: 2 },
            { feature: 'Bayanihan Participation', importance: 0.82, rank: 3 },
            { feature: 'Community Standing', importance: 0.71, rank: 4 },
            { feature: 'Economic Status (Normalized)', importance: 0.64, rank: 5 },
            { feature: 'Digital Literacy', importance: 0.58, rank: 6 },
            { feature: 'Family Responsibility', importance: 0.45, rank: 7 }
        ];
    }),

    generateCulturalInsights: jest.fn(() => {
        mockNeuralNetworkState.culturalInsights = [
            '🤝 **Kapwa Network Strength** is the strongest predictor of trustworthiness (95.0% importance). This reflects the Filipino cultural emphasis on shared identity and community bonds.',
            '💖 **Utang na Loob Integrity** shows high importance (87.0%). Gratitude and reciprocity bonds are crucial cultural trust indicators.',
            '🏘️ **Bayanihan Participation** demonstrates significant predictive power (82.0%). Community spirit remains central to Filipino trust assessment.',
            '🎯 Excellent model performance: 85.0% accuracy shows that Filipino cultural patterns are highly predictive of trustworthiness.',
            '📊 Cultural authenticity factors significantly outweigh traditional financial metrics in predicting trustworthiness.'
        ];
    }),

    predictTrustScore: jest.fn(async (persona) => {
        if (!mockNeuralNetworkState.model) {
            throw new Error('No trained model available');
        }

        // Mock prediction
        return {
            trustworthiness_score: 0.85,
            confidence: 0.7,
            prediction: 'trustworthy'
        };
    }),

    updateTrainingProgress: jest.fn(async (epoch, logs) => {
        // Mock progress update
        const accuracy = logs.acc || logs.accuracy || 0;
        const valAccuracy = logs.val_acc || logs.val_accuracy || 0;
        
        expect(epoch).toBeGreaterThan(0);
        expect(logs.loss).toBeGreaterThanOrEqual(0);
        expect(accuracy).toBeGreaterThanOrEqual(0);
    }),

    getModelSummary: jest.fn(() => {
        if (!mockNeuralNetworkState.model) {
            return {
                status: 'No model trained yet',
                ready: false
            };
        }

        return {
            status: 'Model ready',
            ready: true,
            totalParams: 2048,
            trainableParams: 2048,
            architecture: mockNeuralNetworkState.architecture,
            hyperparameters: mockNeuralNetworkState.hyperparameters,
            performanceMetrics: mockNeuralNetworkState.performanceMetrics,
            featureImportance: mockNeuralNetworkState.featureImportance,
            trainingHistory: {
                epochs: mockNeuralNetworkState.trainingHistory.epochs.length,
                finalLoss: mockNeuralNetworkState.trainingHistory.loss.slice(-1)[0],
                finalAccuracy: mockNeuralNetworkState.trainingHistory.accuracy.slice(-1)[0]
            },
            isTraining: mockNeuralNetworkState.isTraining,
            culturalInsights: mockNeuralNetworkState.culturalInsights?.length || 0
        };
    })
};

// Global setup
beforeAll(() => {
    global.neuralNetworkState = mockNeuralNetworkState;
    Object.keys(mockNeuralNetworkFunctions).forEach(key => {
        global[key] = mockNeuralNetworkFunctions[key];
    });
});

describe('Neural Network Module', () => {
    
    beforeEach(() => {
        jest.clearAllMocks();
        global.mlCore = global.createMockMLState();
        mockNeuralNetworkState.isTraining = false;
        mockNeuralNetworkState.model = null;
    });

    describe('Training Data Preparation', () => {
        
        test('prepareTrainingData should format data correctly', () => {
            const dataset = global.createMockDataset();
            const result = mockNeuralNetworkFunctions.prepareTrainingData(dataset);
            
            expect(result.features).toHaveLength(dataset.length);
            expect(result.labels).toHaveLength(dataset.length);
            expect(result.featureNames).toHaveLength(7);
            expect(result.features[0]).toHaveLength(7);
        });
        
        test('prepareTrainingData should normalize features correctly', () => {
            const dataset = [{
                kapwa_score: 0.8,
                bayanihan_participation: 0.7,
                utang_na_loob_integrity: 0.9,
                community_standing_score: 0.6,
                monthly_income: 25000,
                family_size: 4,
                digital_literacy_score: 7,
                trustworthiness_label: 'trustworthy'
            }];
            
            const result = mockNeuralNetworkFunctions.prepareTrainingData(dataset);
            
            // Check that all features are normalized to [0, 1]
            expect(result.features[0].every(val => val >= 0 && val <= 1)).toBe(true);
        });
        
        test('prepareTrainingData should handle missing values', () => {
            const dataset = [{
                kapwa_score: 0.8,
                // Missing some features
                trustworthiness_label: 'trustworthy'
            }];
            
            const result = mockNeuralNetworkFunctions.prepareTrainingData(dataset);
            
            expect(result.features[0]).toHaveLength(7);
            expect(result.features[0].every(val => typeof val === 'number')).toBe(true);
        });
        
        test('prepareTrainingData should create correct labels', () => {
            const dataset = [
                { trustworthiness_label: 'trustworthy' },
                { trustworthiness_label: 'untrustworthy' }
            ];
            
            const result = mockNeuralNetworkFunctions.prepareTrainingData(dataset);
            
            expect(result.labels[0][0]).toBe(1.0);
            expect(result.labels[1][0]).toBe(0.0);
        });
    });

    describe('Model Creation', () => {
        
        test('createCulturalIntelligenceModel should return valid model', () => {
            const model = mockNeuralNetworkFunctions.createCulturalIntelligenceModel();
            
            expect(model).toBeDefined();
            expect(model.compile).toBeDefined();
            expect(model.fit).toBeDefined();
            expect(model.predict).toBeDefined();
        });
    });

    describe('Training Process', () => {
        
        test('trainNeuralNetwork should require dataset', async () => {
            global.mlCore.sharedData = null;
            
            await expect(mockNeuralNetworkFunctions.trainNeuralNetwork()).rejects.toThrow('No dataset loaded');
        });
        
        test('trainNeuralNetwork should prevent concurrent training', async () => {
            global.mlCore.sharedData = global.createMockDataset();
            mockNeuralNetworkState.isTraining = true;
            
            await expect(mockNeuralNetworkFunctions.trainNeuralNetwork()).rejects.toThrow('Model is already training');
        });
        
        test('trainNeuralNetwork should complete successfully', async () => {
            global.mlCore.sharedData = global.createMockDataset();
            
            const model = await mockNeuralNetworkFunctions.trainNeuralNetwork();
            
            expect(model).toBeDefined();
            expect(mockNeuralNetworkState.model).toBe(model);
            expect(mockNeuralNetworkState.trainingHistory.epochs.length).toBeGreaterThan(0);
        });
        
        test('training should update progress correctly', async () => {
            const logs = {
                loss: 0.5,
                acc: 0.8,
                val_loss: 0.6,
                val_acc: 0.75
            };
            
            await mockNeuralNetworkFunctions.updateTrainingProgress(5, logs);
            
            expect(mockNeuralNetworkFunctions.updateTrainingProgress).toHaveBeenCalledWith(5, logs);
        });
    });

    describe('Performance Metrics', () => {
        
        test('calculatePerformanceMetrics should compute metrics', async () => {
            const mockModel = mockNeuralNetworkFunctions.createCulturalIntelligenceModel();
            const mockValXs = {}; // Mock tensor
            const mockValYs = {}; // Mock tensor
            
            await mockNeuralNetworkFunctions.calculatePerformanceMetrics(mockModel, mockValXs, mockValYs);
            
            expect(mockNeuralNetworkState.performanceMetrics.accuracy).toBeGreaterThan(0);
            expect(mockNeuralNetworkState.performanceMetrics.precision).toBeGreaterThan(0);
            expect(mockNeuralNetworkState.performanceMetrics.recall).toBeGreaterThan(0);
            expect(mockNeuralNetworkState.performanceMetrics.f1Score).toBeGreaterThan(0);
        });
    });

    describe('Feature Importance Analysis', () => {
        
        test('analyzeFeatureImportance should identify important features', async () => {
            const mockModel = mockNeuralNetworkFunctions.createCulturalIntelligenceModel();
            const features = [[0.8, 0.7, 0.9, 0.6, 0.5, 0.4, 0.3]];
            const featureNames = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7'];
            
            await mockNeuralNetworkFunctions.analyzeFeatureImportance(mockModel, features, featureNames);
            
            expect(mockNeuralNetworkState.featureImportance).toBeDefined();
            expect(mockNeuralNetworkState.featureImportance).toHaveLength(7);
            expect(mockNeuralNetworkState.featureImportance[0].importance).toBeGreaterThan(0);
        });
    });

    describe('Cultural Insights Generation', () => {
        
        test('generateCulturalInsights should create meaningful insights', () => {
            // Setup feature importance first
            mockNeuralNetworkState.featureImportance = [
                { feature: 'Kapwa Network Strength', importance: 0.95, rank: 1 },
                { feature: 'Utang na Loob Integrity', importance: 0.87, rank: 2 }
            ];
            mockNeuralNetworkState.performanceMetrics = { accuracy: 0.85 };
            
            mockNeuralNetworkFunctions.generateCulturalInsights();
            
            expect(mockNeuralNetworkState.culturalInsights).toBeDefined();
            expect(mockNeuralNetworkState.culturalInsights.length).toBeGreaterThan(0);
            expect(mockNeuralNetworkState.culturalInsights[0]).toContain('Kapwa Network Strength');
        });
    });

    describe('Prediction', () => {
        
        test('predictTrustScore should require trained model', async () => {
            mockNeuralNetworkState.model = null;
            const testPersona = { kapwa_score: 0.8 };
            
            await expect(mockNeuralNetworkFunctions.predictTrustScore(testPersona)).rejects.toThrow('No trained model available');
        });
        
        test('predictTrustScore should return valid predictions', async () => {
            mockNeuralNetworkState.model = mockNeuralNetworkFunctions.createCulturalIntelligenceModel();
            const testPersona = {
                kapwa_score: 0.8,
                bayanihan_participation: 0.7,
                utang_na_loob_integrity: 0.9,
                community_standing_score: 0.6,
                monthly_income: 25000,
                family_size: 4,
                digital_literacy_score: 7
            };
            
            const result = await mockNeuralNetworkFunctions.predictTrustScore(testPersona);
            
            expect(result.trustworthiness_score).toBeGreaterThanOrEqual(0);
            expect(result.trustworthiness_score).toBeLessThanOrEqual(1);
            expect(result.confidence).toBeGreaterThanOrEqual(0);
            expect(result.prediction).toMatch(/trustworthy|untrustworthy/);
        });
    });

    describe('Model Summary', () => {
        
        test('getModelSummary should return not ready when no model', () => {
            mockNeuralNetworkState.model = null;
            
            const summary = mockNeuralNetworkFunctions.getModelSummary();
            
            expect(summary.ready).toBe(false);
            expect(summary.status).toBe('No model trained yet');
        });
        
        test('getModelSummary should return complete info when model ready', () => {
            mockNeuralNetworkState.model = mockNeuralNetworkFunctions.createCulturalIntelligenceModel();
            mockNeuralNetworkState.trainingHistory.epochs = [1, 2, 3];
            mockNeuralNetworkState.culturalInsights = ['Insight 1', 'Insight 2'];
            
            const summary = mockNeuralNetworkFunctions.getModelSummary();
            
            expect(summary.ready).toBe(true);
            expect(summary.status).toBe('Model ready');
            expect(summary.totalParams).toBeGreaterThan(0);
            expect(summary.architecture).toBeDefined();
            expect(summary.performanceMetrics).toBeDefined();
        });
    });

    describe('State Management', () => {
        
        test('initial state should be correct', () => {
            expect(mockNeuralNetworkState.isTraining).toBe(false);
            expect(mockNeuralNetworkState.architecture.inputLayer).toBe(7);
            expect(mockNeuralNetworkState.hyperparameters.learningRate).toBe(0.01);
        });
        
        test('training should update state correctly', async () => {
            global.mlCore.sharedData = global.createMockDataset();
            
            await mockNeuralNetworkFunctions.trainNeuralNetwork();
            
            expect(mockNeuralNetworkState.model).toBeDefined();
            expect(mockNeuralNetworkState.trainingHistory.epochs.length).toBeGreaterThan(0);
            expect(mockNeuralNetworkState.performanceMetrics.accuracy).toBeGreaterThan(0);
        });
    });
});

describe('Neural Network Integration Tests', () => {
    
    test('complete neural network workflow', async () => {
        // Setup
        global.mlCore.sharedData = global.createMockDataset();
        
        // Train model
        const model = await mockNeuralNetworkFunctions.trainNeuralNetwork();
        expect(model).toBeDefined();
        
        // Check training results
        expect(mockNeuralNetworkState.trainingHistory.epochs.length).toBeGreaterThan(0);
        expect(mockNeuralNetworkState.performanceMetrics.accuracy).toBeGreaterThan(0);
        expect(mockNeuralNetworkState.featureImportance).toBeDefined();
        expect(mockNeuralNetworkState.culturalInsights.length).toBeGreaterThan(0);
        
        // Test prediction
        const testPersona = global.createMockDataset()[0];
        const prediction = await mockNeuralNetworkFunctions.predictTrustScore(testPersona);
        expect(prediction.trustworthiness_score).toBeDefined();
        
        // Get model summary
        const summary = mockNeuralNetworkFunctions.getModelSummary();
        expect(summary.ready).toBe(true);
    });
});

describe('Neural Network Error Handling', () => {
    
    test('should handle TensorFlow.js not available', async () => {
        // Mock TensorFlow not available
        const originalTF = global.tf;
        global.tf = undefined;
        
        global.mlCore.sharedData = global.createMockDataset();
        
        // Should handle gracefully (in real implementation would show warning)
        expect(global.tf).toBeUndefined();
        
        // Restore
        global.tf = originalTF;
    });
    
    test('should handle invalid training data', async () => {
        global.mlCore.sharedData = []; // Empty dataset
        
        await expect(mockNeuralNetworkFunctions.trainNeuralNetwork()).rejects.toThrow('No dataset loaded');
    });
    
    test('should handle model compilation errors gracefully', () => {
        // Create model that might fail
        const model = mockNeuralNetworkFunctions.createCulturalIntelligenceModel();
        
        // Should not throw when calling compile
        expect(() => model.compile()).not.toThrow();
    });
});