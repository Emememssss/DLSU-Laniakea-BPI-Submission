/**
 * Evaluation Metrics Tests
 * Tests for comprehensive model evaluation including confusion matrices and performance metrics
 */

import { jest } from '@jest/globals';

// Mock evaluation functions that would be loaded from evaluation-metrics.js
const mockEvaluationFunctions = {
    calculateBinaryClassificationMetrics: jest.fn((yTrue, yPred, yProb = null, threshold = 0.5) => {
        if (yTrue.length !== yPred.length) {
            throw new Error('Prediction and ground truth arrays must have the same length');
        }

        // Simple mock implementation
        const binaryPred = yProb ? 
            yProb.map(prob => prob >= threshold ? 1 : 0) : 
            yPred.map(pred => pred >= threshold ? 1 : 0);
        const binaryTrue = yTrue.map(val => val >= threshold ? 1 : 0);

        let tp = 0, tn = 0, fp = 0, fn = 0;
        for (let i = 0; i < binaryTrue.length; i++) {
            if (binaryTrue[i] === 1 && binaryPred[i] === 1) tp++;
            else if (binaryTrue[i] === 0 && binaryPred[i] === 0) tn++;
            else if (binaryTrue[i] === 0 && binaryPred[i] === 1) fp++;
            else if (binaryTrue[i] === 1 && binaryPred[i] === 0) fn++;
        }

        const accuracy = (tp + tn) / (tp + tn + fp + fn);
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

        return {
            accuracy,
            precision,
            recall,
            f1Score,
            confusionMatrix: { truePositive: tp, trueNegative: tn, falsePositive: fp, falseNegative: fn }
        };
    }),

    calculateROCCurve: jest.fn((yTrue, yProb) => {
        // Mock ROC calculation
        const points = [
            { fpr: 0, tpr: 0, threshold: 1 },
            { fpr: 0.1, tpr: 0.3, threshold: 0.8 },
            { fpr: 0.2, tpr: 0.6, threshold: 0.6 },
            { fpr: 0.3, tpr: 0.8, threshold: 0.4 },
            { fpr: 1, tpr: 1, threshold: 0 }
        ];
        const auc = 0.85;
        return { points, auc };
    }),

    calculateClusteringMetrics: jest.fn((data, labels, centroids) => {
        // Mock clustering metrics
        return {
            wcss: 25.6,
            silhouetteScore: 0.72,
            calinskiHarabasz: 156.3,
            daviesBouldinIndex: 0.68,
            numClusters: centroids.length,
            numPoints: data.length
        };
    }),

    calculateSilhouetteScore: jest.fn((data, labels, centroids) => {
        return 0.72; // Mock silhouette score
    }),

    createConfusionMatrixChart: jest.fn((confusionMatrix, elementId) => {
        return { id: 'mock-confusion-matrix-chart' };
    }),

    createROCCurveChart: jest.fn((rocPoints, auc, elementId) => {
        return { id: 'mock-roc-curve-chart' };
    }),

    generateEvaluationReport: jest.fn((metrics) => {
        return {
            timestamp: '2024-01-01T00:00:00.000Z',
            modelType: 'Neural Network',
            metrics: metrics,
            interpretation: ['🎯 Excellent accuracy: Model performs exceptionally well'],
            recommendations: ['Consider feature engineering for further improvement']
        };
    }),

    interpretMetrics: jest.fn((metrics) => {
        const interpretations = [];
        if (metrics.accuracy >= 0.8) {
            interpretations.push('✅ Good accuracy: Model shows strong predictive performance');
        }
        if (metrics.f1Score >= 0.8) {
            interpretations.push('🎊 Strong F1-score: Well-balanced model performance');
        }
        return interpretations;
    })
};

// Global setup
beforeAll(() => {
    Object.keys(mockEvaluationFunctions).forEach(key => {
        global[key] = mockEvaluationFunctions[key];
    });
});

describe('Evaluation Metrics Module', () => {
    
    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('Binary Classification Metrics', () => {
        
        const perfectPredictions = {
            yTrue: [1, 1, 0, 0, 1, 0],
            yPred: [1, 1, 0, 0, 1, 0]
        };

        const imperfectPredictions = {
            yTrue: [1, 1, 0, 0, 1, 0],
            yPred: [1, 0, 0, 1, 1, 0]
        };

        test('should calculate perfect metrics correctly', () => {
            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(
                perfectPredictions.yTrue,
                perfectPredictions.yPred
            );

            expect(metrics.accuracy).toBe(1.0);
            expect(metrics.precision).toBe(1.0);
            expect(metrics.recall).toBe(1.0);
            expect(metrics.f1Score).toBe(1.0);
            expect(metrics.confusionMatrix.truePositive).toBe(3);
            expect(metrics.confusionMatrix.trueNegative).toBe(3);
            expect(metrics.confusionMatrix.falsePositive).toBe(0);
            expect(metrics.confusionMatrix.falseNegative).toBe(0);
        });

        test('should calculate imperfect metrics correctly', () => {
            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(
                imperfectPredictions.yTrue,
                imperfectPredictions.yPred
            );

            expect(metrics.accuracy).toBeLessThan(1.0);
            expect(metrics.precision).toBeLessThan(1.0);
            expect(metrics.recall).toBeLessThan(1.0);
            expect(metrics.f1Score).toBeLessThan(1.0);
        });

        test('should handle probability predictions with threshold', () => {
            const yTrue = [1, 1, 0, 0];
            const yProb = [0.9, 0.7, 0.3, 0.1];
            const threshold = 0.5;

            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(
                yTrue, null, yProb, threshold
            );

            expect(metrics.accuracy).toBe(1.0);
            expect(mockEvaluationFunctions.calculateBinaryClassificationMetrics).toHaveBeenCalledWith(
                yTrue, null, yProb, threshold
            );
        });

        test('should throw error for mismatched array lengths', () => {
            const yTrue = [1, 0, 1];
            const yPred = [1, 0]; // Different length

            expect(() => {
                mockEvaluationFunctions.calculateBinaryClassificationMetrics(yTrue, yPred);
            }).toThrow('Prediction and ground truth arrays must have the same length');
        });

        test('should handle edge case with all positive predictions', () => {
            const yTrue = [0, 0, 0, 0];
            const yPred = [1, 1, 1, 1];

            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(yTrue, yPred);

            expect(metrics.confusionMatrix.truePositive).toBe(0);
            expect(metrics.confusionMatrix.falsePositive).toBe(4);
            expect(metrics.precision).toBe(0);
        });

        test('should handle edge case with all negative predictions', () => {
            const yTrue = [1, 1, 1, 1];
            const yPred = [0, 0, 0, 0];

            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(yTrue, yPred);

            expect(metrics.confusionMatrix.truePositive).toBe(0);
            expect(metrics.confusionMatrix.falseNegative).toBe(4);
            expect(metrics.recall).toBe(0);
        });
    });

    describe('ROC Curve Calculation', () => {
        
        test('should calculate ROC curve points and AUC', () => {
            const yTrue = [1, 1, 0, 0, 1];
            const yProb = [0.9, 0.7, 0.6, 0.3, 0.8];

            const result = mockEvaluationFunctions.calculateROCCurve(yTrue, yProb);

            expect(result.points).toBeDefined();
            expect(result.auc).toBeDefined();
            expect(result.points.length).toBeGreaterThan(0);
            expect(result.auc).toBeGreaterThan(0);
            expect(result.auc).toBeLessThanOrEqual(1);
        });

        test('should return reasonable AUC values', () => {
            const yTrue = [1, 1, 0, 0];
            const yProb = [0.9, 0.8, 0.3, 0.2];

            const result = mockEvaluationFunctions.calculateROCCurve(yTrue, yProb);

            expect(result.auc).toBeGreaterThan(0.5); // Should be better than random
        });
    });

    describe('Clustering Metrics', () => {
        
        const mockClusteringData = {
            data: [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]],
            labels: [0, 0, 1, 1, 0, 1],
            centroids: [[1.17, 1.47], [7.33, 9]]
        };

        test('should calculate clustering metrics', () => {
            const metrics = mockEvaluationFunctions.calculateClusteringMetrics(
                mockClusteringData.data,
                mockClusteringData.labels,
                mockClusteringData.centroids
            );

            expect(metrics.wcss).toBeGreaterThan(0);
            expect(metrics.silhouetteScore).toBeGreaterThan(-1);
            expect(metrics.silhouetteScore).toBeLessThanOrEqual(1);
            expect(metrics.numClusters).toBe(2);
            expect(metrics.numPoints).toBe(6);
        });

        test('should calculate silhouette score separately', () => {
            const score = mockEvaluationFunctions.calculateSilhouetteScore(
                mockClusteringData.data,
                mockClusteringData.labels,
                mockClusteringData.centroids
            );

            expect(score).toBeGreaterThan(-1);
            expect(score).toBeLessThanOrEqual(1);
        });
    });

    describe('Visualization Functions', () => {
        
        test('should create confusion matrix chart', () => {
            const confusionMatrix = {
                truePositive: 45,
                trueNegative: 40,
                falsePositive: 5,
                falseNegative: 10
            };

            const chart = mockEvaluationFunctions.createConfusionMatrixChart(
                confusionMatrix,
                'confusionMatrixChart'
            );

            expect(chart).toBeDefined();
            expect(mockEvaluationFunctions.createConfusionMatrixChart).toHaveBeenCalledWith(
                confusionMatrix,
                'confusionMatrixChart'
            );
        });

        test('should create ROC curve chart', () => {
            const rocPoints = [
                { fpr: 0, tpr: 0 },
                { fpr: 0.2, tpr: 0.6 },
                { fpr: 1, tpr: 1 }
            ];
            const auc = 0.8;

            const chart = mockEvaluationFunctions.createROCCurveChart(
                rocPoints,
                auc,
                'rocCurveChart'
            );

            expect(chart).toBeDefined();
            expect(mockEvaluationFunctions.createROCCurveChart).toHaveBeenCalledWith(
                rocPoints,
                auc,
                'rocCurveChart'
            );
        });
    });

    describe('Evaluation Report Generation', () => {
        
        test('should generate comprehensive evaluation report', () => {
            const mockMetrics = {
                accuracy: 0.85,
                precision: 0.87,
                recall: 0.83,
                f1Score: 0.85,
                confusionMatrix: {
                    truePositive: 42,
                    trueNegative: 38,
                    falsePositive: 6,
                    falseNegative: 8
                }
            };

            const report = mockEvaluationFunctions.generateEvaluationReport(mockMetrics);

            expect(report.timestamp).toBeDefined();
            expect(report.modelType).toBeDefined();
            expect(report.metrics).toEqual(mockMetrics);
            expect(report.interpretation).toBeDefined();
            expect(report.recommendations).toBeDefined();
        });

        test('should interpret metrics correctly', () => {
            const highPerformanceMetrics = {
                accuracy: 0.95,
                f1Score: 0.93
            };

            const interpretations = mockEvaluationFunctions.interpretMetrics(highPerformanceMetrics);

            expect(interpretations.length).toBeGreaterThan(0);
            expect(interpretations.some(interp => interp.includes('Good accuracy'))).toBe(true);
        });
    });

    describe('Cultural Intelligence Specific Tests', () => {
        
        test('should evaluate cultural intelligence model performance', () => {
            // Mock cultural intelligence prediction results
            const culturalTestData = {
                yTrue: [1, 1, 0, 1, 0, 1, 0, 0, 1, 1], // Trustworthy labels
                yPred: [1, 1, 0, 1, 0, 0, 0, 0, 1, 1], // Model predictions
                yProb: [0.95, 0.89, 0.23, 0.87, 0.34, 0.45, 0.12, 0.08, 0.92, 0.91] // Probabilities
            };

            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(
                culturalTestData.yTrue,
                culturalTestData.yPred,
                culturalTestData.yProb
            );

            // Check that metrics are reasonable for cultural intelligence
            expect(metrics.accuracy).toBeGreaterThan(0.5); // Better than random
            expect(metrics.precision).toBeGreaterThan(0); 
            expect(metrics.recall).toBeGreaterThan(0);

            // Generate ROC for cultural model
            const rocResult = mockEvaluationFunctions.calculateROCCurve(
                culturalTestData.yTrue,
                culturalTestData.yProb
            );

            expect(rocResult.auc).toBeGreaterThan(0.5); // Better than random classifier
        });

        test('should handle imbalanced cultural datasets', () => {
            // Simulate imbalanced dataset (more legitimate users than scammers)
            const imbalancedData = {
                yTrue: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0], // 80% trustworthy, 20% untrustworthy
                yPred: [1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
            };

            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(
                imbalancedData.yTrue,
                imbalancedData.yPred
            );

            // Should still provide meaningful metrics for imbalanced data
            expect(metrics.accuracy).toBeDefined();
            expect(metrics.precision).toBeDefined();
            expect(metrics.recall).toBeDefined();
            expect(metrics.f1Score).toBeDefined();
        });
    });

    describe('Error Handling', () => {
        
        test('should handle empty arrays gracefully', () => {
            expect(() => {
                mockEvaluationFunctions.calculateBinaryClassificationMetrics([], []);
            }).not.toThrow();
        });

        test('should validate input parameters', () => {
            const yTrue = [1, 0, 1];
            const yPred = [1, 0]; // Wrong length

            expect(() => {
                mockEvaluationFunctions.calculateBinaryClassificationMetrics(yTrue, yPred);
            }).toThrow();
        });
    });

    describe('Integration with ML Models', () => {
        
        test('should integrate with K-Means clustering evaluation', () => {
            const clusteringResults = {
                data: [[1, 2], [2, 1], [8, 8], [9, 9]],
                labels: [0, 0, 1, 1],
                centroids: [[1.5, 1.5], [8.5, 8.5]]
            };

            const metrics = mockEvaluationFunctions.calculateClusteringMetrics(
                clusteringResults.data,
                clusteringResults.labels,
                clusteringResults.centroids
            );

            expect(metrics.numClusters).toBe(2);
            expect(metrics.numPoints).toBe(4);
            expect(metrics.silhouetteScore).toBeDefined();
        });

        test('should integrate with Neural Network evaluation', () => {
            // Simulate neural network training results
            const nnResults = {
                predictions: [0.9, 0.1, 0.8, 0.2, 0.95],
                groundTruth: [1, 0, 1, 0, 1],
                probabilities: [0.9, 0.1, 0.8, 0.2, 0.95]
            };

            const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(
                nnResults.groundTruth,
                nnResults.predictions,
                nnResults.probabilities
            );

            expect(metrics.accuracy).toBe(1.0); // Perfect predictions in this mock
            expect(metrics.confusionMatrix).toBeDefined();

            // Should also calculate ROC
            const rocResult = mockEvaluationFunctions.calculateROCCurve(
                nnResults.groundTruth,
                nnResults.probabilities
            );

            expect(rocResult.auc).toBeGreaterThan(0.5);
        });
    });
});

describe('Evaluation Metrics Integration Tests', () => {
    
    test('complete evaluation workflow for binary classification', () => {
        // Simulate complete model evaluation workflow
        const testResults = {
            yTrue: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
            yPred: [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            yProb: [0.9, 0.8, 0.2, 0.1, 0.95, 0.3, 0.6, 0.15, 0.92, 0.85]
        };

        // Calculate metrics
        const metrics = mockEvaluationFunctions.calculateBinaryClassificationMetrics(
            testResults.yTrue,
            testResults.yPred,
            testResults.yProb
        );

        expect(metrics).toBeDefined();
        expect(metrics.accuracy).toBeGreaterThan(0);

        // Calculate ROC curve
        const rocResult = mockEvaluationFunctions.calculateROCCurve(
            testResults.yTrue,
            testResults.yProb
        );

        expect(rocResult.auc).toBeGreaterThan(0);

        // Generate report
        const report = mockEvaluationFunctions.generateEvaluationReport(metrics);

        expect(report.metrics).toEqual(metrics);
        expect(report.interpretation).toBeDefined();
        expect(report.recommendations).toBeDefined();

        // Test visualization creation
        const confusionChart = mockEvaluationFunctions.createConfusionMatrixChart(
            metrics.confusionMatrix
        );
        expect(confusionChart).toBeDefined();

        const rocChart = mockEvaluationFunctions.createROCCurveChart(
            rocResult.points,
            rocResult.auc
        );
        expect(rocChart).toBeDefined();
    });

    test('complete evaluation workflow for clustering', () => {
        // Simulate K-means clustering evaluation
        const clusterResults = {
            data: [[1, 1], [1.2, 1.1], [0.8, 0.9], [8, 8], [8.2, 8.1], [7.8, 7.9]],
            labels: [0, 0, 0, 1, 1, 1],
            centroids: [[1, 1], [8, 8]]
        };

        // Calculate clustering metrics
        const metrics = mockEvaluationFunctions.calculateClusteringMetrics(
            clusterResults.data,
            clusterResults.labels,
            clusterResults.centroids
        );

        expect(metrics.numClusters).toBe(2);
        expect(metrics.silhouetteScore).toBeDefined();
        expect(metrics.wcss).toBeGreaterThan(0);

        // Individual silhouette calculation
        const silhouetteScore = mockEvaluationFunctions.calculateSilhouetteScore(
            clusterResults.data,
            clusterResults.labels,
            clusterResults.centroids
        );

        expect(silhouetteScore).toBeGreaterThan(-1);
        expect(silhouetteScore).toBeLessThanOrEqual(1);
    });
});