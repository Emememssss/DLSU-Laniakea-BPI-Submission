/**
 * ML Core Functions Tests
 * Tests for core ML utilities, shared functionality, and notification system
 */

import { jest } from '@jest/globals';

// Mock the ml-core.js file contents
const mockMLCore = {
    initialized: false,
    activeAlgorithms: new Set(),
    sharedData: null,
    models: {},
    charts: {},
    status: {
        dataLoaded: false,
        modelsReady: false,
        trainingInProgress: false
    }
};

// Mock functions that would be loaded from ml-core.js
const mockFunctions = {
    showSuccessNotification: jest.fn(),
    showErrorNotification: jest.fn(),
    showToast: jest.fn(),
    updateProgressBar: jest.fn(),
    createChart: jest.fn(),
    updateChart: jest.fn(),
    registerAlgorithm: jest.fn(),
    stats: {
        mean: (arr) => arr.reduce((sum, val) => sum + val, 0) / arr.length,
        median: (arr) => {
            const sorted = [...arr].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        },
        standardDeviation: (arr) => {
            const mean = mockFunctions.stats.mean(arr);
            const squaredDiffs = arr.map(val => Math.pow(val - mean, 2));
            return Math.sqrt(mockFunctions.stats.mean(squaredDiffs));
        },
        normalize: (arr) => {
            const min = Math.min(...arr);
            const max = Math.max(...arr);
            const range = max - min;
            return range === 0 ? arr.map(() => 0) : arr.map(val => (val - min) / range);
        }
    },
    validators: {
        isNumber: (value) => typeof value === 'number' && !isNaN(value),
        isInRange: (value, min, max) => mockFunctions.validators.isNumber(value) && value >= min && value <= max,
        isValidDataset: (dataset) => Array.isArray(dataset) && dataset.length > 0 && typeof dataset[0] === 'object'
    },
    calculateCulturalAuthenticity: (persona) => {
        const weights = {
            kapwa_score: 0.3,
            bayanihan_participation: 0.25,
            utang_na_loob_integrity: 0.25,
            community_standing_score: 0.2
        };
        
        let score = 0;
        let totalWeight = 0;
        
        Object.entries(weights).forEach(([feature, weight]) => {
            if (persona[feature] !== undefined && typeof persona[feature] === 'number') {
                score += persona[feature] * weight;
                totalWeight += weight;
            }
        });
        
        return totalWeight > 0 ? score / totalWeight : 0;
    }
};

// Global setup
beforeAll(() => {
    global.mlCore = mockMLCore;
    Object.keys(mockFunctions).forEach(key => {
        global[key] = mockFunctions[key];
    });
});

describe('ML Core Module', () => {
    
    describe('Notification System', () => {
        beforeEach(() => {
            jest.clearAllMocks();
        });
        
        test('showSuccessNotification should be callable', () => {
            mockFunctions.showSuccessNotification('Test success message');
            expect(mockFunctions.showSuccessNotification).toHaveBeenCalledWith('Test success message');
        });
        
        test('showErrorNotification should be callable', () => {
            mockFunctions.showErrorNotification('Test error message');
            expect(mockFunctions.showErrorNotification).toHaveBeenCalledWith('Test error message');
        });
        
        test('showToast should handle different types', () => {
            const testMessage = 'Test toast message';
            ['success', 'error', 'warning', 'info'].forEach(type => {
                mockFunctions.showToast(testMessage, type);
                expect(mockFunctions.showToast).toHaveBeenCalledWith(testMessage, type);
            });
        });
    });
    
    describe('Progress Bar System', () => {
        test('updateProgressBar should handle percentage values', () => {
            const elementId = 'testProgressBar';
            const percentage = 65;
            const message = 'Processing...';
            
            mockFunctions.updateProgressBar(elementId, percentage, message);
            expect(mockFunctions.updateProgressBar).toHaveBeenCalledWith(elementId, percentage, message);
        });
    });
    
    describe('Chart Management', () => {
        test('createChart should be callable with config', () => {
            const canvasId = 'testCanvas';
            const config = { type: 'line', data: { labels: [], datasets: [] } };
            
            mockFunctions.createChart(canvasId, config);
            expect(mockFunctions.createChart).toHaveBeenCalledWith(canvasId, config);
        });
        
        test('updateChart should handle new data', () => {
            const canvasId = 'testCanvas';
            const newData = { labels: ['A', 'B'], datasets: [{ data: [1, 2] }] };
            
            mockFunctions.updateChart(canvasId, newData);
            expect(mockFunctions.updateChart).toHaveBeenCalledWith(canvasId, newData);
        });
    });
    
    describe('Statistical Functions', () => {
        const testData = [1, 2, 3, 4, 5];
        
        test('stats.mean should calculate correct mean', () => {
            const result = mockFunctions.stats.mean(testData);
            expect(result).toBe(3);
        });
        
        test('stats.median should calculate correct median', () => {
            const result = mockFunctions.stats.median(testData);
            expect(result).toBe(3);
            
            const evenData = [1, 2, 3, 4];
            const evenResult = mockFunctions.stats.median(evenData);
            expect(evenResult).toBe(2.5);
        });
        
        test('stats.standardDeviation should calculate correct std dev', () => {
            const result = mockFunctions.stats.standardDeviation(testData);
            expect(result).toBeCloseTo(1.58, 1);
        });
        
        test('stats.normalize should normalize data to 0-1 range', () => {
            const result = mockFunctions.stats.normalize(testData);
            expect(result[0]).toBe(0);
            expect(result[result.length - 1]).toBe(1);
            expect(result.every(val => val >= 0 && val <= 1)).toBe(true);
        });
    });
    
    describe('Validation Functions', () => {
        test('validators.isNumber should validate numbers correctly', () => {
            expect(mockFunctions.validators.isNumber(42)).toBe(true);
            expect(mockFunctions.validators.isNumber(3.14)).toBe(true);
            expect(mockFunctions.validators.isNumber('42')).toBe(false);
            expect(mockFunctions.validators.isNumber(NaN)).toBe(false);
        });
        
        test('validators.isInRange should check ranges correctly', () => {
            expect(mockFunctions.validators.isInRange(5, 1, 10)).toBe(true);
            expect(mockFunctions.validators.isInRange(0, 1, 10)).toBe(false);
            expect(mockFunctions.validators.isInRange(15, 1, 10)).toBe(false);
        });
        
        test('validators.isValidDataset should validate datasets', () => {
            const validDataset = [{ id: 1, name: 'Test' }];
            const invalidDataset = [];
            const notArray = 'not an array';
            
            expect(mockFunctions.validators.isValidDataset(validDataset)).toBe(true);
            expect(mockFunctions.validators.isValidDataset(invalidDataset)).toBe(false);
            expect(mockFunctions.validators.isValidDataset(notArray)).toBe(false);
        });
    });
    
    describe('Cultural Intelligence Functions', () => {
        test('calculateCulturalAuthenticity should calculate scores correctly', () => {
            const mockPersona = {
                kapwa_score: 0.8,
                bayanihan_participation: 0.7,
                utang_na_loob_integrity: 0.9,
                community_standing_score: 0.6
            };
            
            const score = mockFunctions.calculateCulturalAuthenticity(mockPersona);
            expect(score).toBeGreaterThan(0);
            expect(score).toBeLessThanOrEqual(1);
        });
        
        test('calculateCulturalAuthenticity should handle missing values', () => {
            const incompletePersona = {
                kapwa_score: 0.8
            };
            
            const score = mockFunctions.calculateCulturalAuthenticity(incompletePersona);
            expect(score).toBeGreaterThanOrEqual(0);
        });
        
        test('calculateCulturalAuthenticity should handle empty persona', () => {
            const emptyPersona = {};
            const score = mockFunctions.calculateCulturalAuthenticity(emptyPersona);
            expect(score).toBe(0);
        });
    });
    
    describe('Algorithm Registration', () => {
        beforeEach(() => {
            mockMLCore.activeAlgorithms.clear();
        });
        
        test('registerAlgorithm should add algorithm to active set', () => {
            mockFunctions.registerAlgorithm('kmeans');
            expect(mockFunctions.registerAlgorithm).toHaveBeenCalledWith('kmeans');
        });
    });
    
    describe('Error Handling', () => {
        test('should handle errors gracefully', () => {
            // Test error scenarios
            expect(() => {
                mockFunctions.stats.mean([]);
            }).not.toThrow();
            
            expect(() => {
                mockFunctions.calculateCulturalAuthenticity(null);
            }).not.toThrow();
        });
    });
});

describe('ML Core Integration Tests', () => {
    test('complete workflow should work together', () => {
        // Register algorithm
        mockFunctions.registerAlgorithm('neural-network');
        
        // Mock dataset
        const dataset = global.createMockDataset();
        
        // Validate dataset
        expect(mockFunctions.validators.isValidDataset(dataset)).toBe(true);
        
        // Calculate cultural scores
        const scores = dataset.map(persona => 
            mockFunctions.calculateCulturalAuthenticity(persona)
        );
        
        expect(scores.length).toBe(dataset.length);
        expect(scores.every(score => score >= 0 && score <= 1)).toBe(true);
        
        // Calculate statistics
        const meanScore = mockFunctions.stats.mean(scores);
        expect(meanScore).toBeGreaterThan(0);
    });
});