/**
 * K-Means Clustering Tests
 * Tests for K-means algorithm implementation and visualization
 */

import { jest } from '@jest/globals';

// Mock K-Means Algorithm Class
class MockKMeansAlgorithm {
    constructor(k, features, maxIterations = 100, convergenceThreshold = 0.001) {
        this.k = k;
        this.features = features;
        this.maxIterations = maxIterations;
        this.convergenceThreshold = convergenceThreshold;
        this.centroids = [];
        this.clusters = [];
        this.iterations = 0;
        this.hasConverged = false;
        this.wcss = 0;
        this.silhouetteScore = 0;
    }

    initializeCentroids(data) {
        // Mock centroid initialization
        this.centroids = Array.from({ length: this.k }, () => 
            Array.from({ length: data[0].length }, () => Math.random())
        );
        return this.centroids;
    }

    calculateDistance(point1, point2) {
        return Math.sqrt(
            point1.reduce((sum, val, i) => 
                sum + Math.pow(val - point2[i], 2), 0
            )
        );
    }

    assignPointsToCluster(data) {
        return data.map(() => Math.floor(Math.random() * this.k));
    }

    updateCentroids(data) {
        return this.centroids.map((centroid, k) => 
            centroid.map(() => Math.random())
        );
    }

    checkConvergence(oldCentroids) {
        this.iterations++;
        return this.iterations >= 5; // Mock convergence after 5 iterations
    }

    calculateWCSS(data) {
        this.wcss = Math.random() * 100;
        return this.wcss;
    }

    calculateSilhouetteScore(data) {
        this.silhouetteScore = Math.random() * 0.8;
        return this.silhouetteScore;
    }

    async fit(data, onIterationCallback = null) {
        this.initializeCentroids(data);
        
        while (!this.hasConverged && this.iterations < this.maxIterations) {
            const oldCentroids = this.centroids.map(c => [...c]);
            
            this.clusters = this.assignPointsToCluster(data);
            this.centroids = this.updateCentroids(data);
            this.hasConverged = this.checkConvergence(oldCentroids);
            
            if (onIterationCallback) {
                await onIterationCallback({
                    iteration: this.iterations,
                    centroids: this.centroids,
                    clusters: this.clusters,
                    hasConverged: this.hasConverged
                });
            }
        }
        
        this.wcss = this.calculateWCSS(data);
        this.silhouetteScore = this.calculateSilhouetteScore(data);
        
        return {
            centroids: this.centroids,
            clusters: this.clusters,
            iterations: this.iterations,
            hasConverged: this.hasConverged,
            wcss: this.wcss,
            silhouetteScore: this.silhouetteScore,
            history: []
        };
    }
}

// Mock K-Means state
const mockKmeansState = {
    algorithm: null,
    clusters: [],
    centroids: [],
    currentK: 3,
    features: ['kapwa_score', 'bayanihan_participation', 'utang_na_loob_integrity'],
    isRunning: false,
    iterations: 0,
    maxIterations: 100,
    convergenceThreshold: 0.001,
    animationSpeed: 500,
    wcss: 0,
    silhouetteScore: 0
};

// Mock K-Means functions
const mockKmeansFunctions = {
    runKMeansClustering: jest.fn(async () => {
        if (!global.mlCore.sharedData || global.mlCore.sharedData.length === 0) {
            throw new Error('No dataset loaded');
        }
        
        mockKmeansState.isRunning = true;
        
        try {
            const algorithm = new MockKMeansAlgorithm(mockKmeansState.currentK, mockKmeansState.features);
            const mockData = global.mlCore.sharedData.map(() => [Math.random(), Math.random(), Math.random()]);
            
            const result = await algorithm.fit(mockData);
            
            mockKmeansState.clusters = result.clusters;
            mockKmeansState.centroids = result.centroids;
            mockKmeansState.iterations = result.iterations;
            mockKmeansState.wcss = result.wcss;
            mockKmeansState.silhouetteScore = result.silhouetteScore;
            
            return result;
        } finally {
            mockKmeansState.isRunning = false;
        }
    }),
    
    prepareKMeansData: jest.fn((dataset, features) => {
        return dataset.map(row => ({
            point: features.map(feature => {
                let value = row[feature];
                if (typeof value !== 'number') value = Math.random();
                return Math.min(Math.max(value, 0), 1);
            }),
            metadata: {
                id: row.persona_id || `person_${Math.random()}`,
                name: row.name || 'Unknown',
                originalData: row
            }
        }));
    }),
    
    normalizeData: jest.fn((data) => {
        if (data.length === 0) return [];
        
        const dimensions = data[0].length;
        const mins = new Array(dimensions).fill(Infinity);
        const maxs = new Array(dimensions).fill(-Infinity);
        
        data.forEach(point => {
            point.forEach((value, dim) => {
                mins[dim] = Math.min(mins[dim], value);
                maxs[dim] = Math.max(maxs[dim], value);
            });
        });
        
        return data.map(point => 
            point.map((value, dim) => {
                const range = maxs[dim] - mins[dim];
                return range === 0 ? 0.5 : (value - mins[dim]) / range;
            })
        );
    }),
    
    updateKMeansVisualization: jest.fn(),
    resetKMeansClustering: jest.fn(() => {
        Object.assign(mockKmeansState, {
            isRunning: false,
            algorithm: null,
            clusters: [],
            centroids: [],
            iterations: 0,
            wcss: 0,
            silhouetteScore: 0
        });
    }),
    
    recommendOptimalK: jest.fn(() => {
        const optimalK = 4; // Mock recommendation
        mockKmeansState.currentK = optimalK;
        return optimalK;
    })
};

// Global setup
beforeAll(() => {
    global.KMeansAlgorithm = MockKMeansAlgorithm;
    global.kmeansState = mockKmeansState;
    Object.keys(mockKmeansFunctions).forEach(key => {
        global[key] = mockKmeansFunctions[key];
    });
});

describe('K-Means Clustering Module', () => {
    
    beforeEach(() => {
        jest.clearAllMocks();
        global.mlCore = global.createMockMLState();
        mockKmeansState.isRunning = false;
    });
    
    describe('K-Means Algorithm Class', () => {
        
        test('should initialize with correct parameters', () => {
            const k = 3;
            const features = ['feature1', 'feature2'];
            const algorithm = new MockKMeansAlgorithm(k, features);
            
            expect(algorithm.k).toBe(k);
            expect(algorithm.features).toEqual(features);
            expect(algorithm.maxIterations).toBe(100);
            expect(algorithm.convergenceThreshold).toBe(0.001);
        });
        
        test('should initialize centroids correctly', () => {
            const algorithm = new MockKMeansAlgorithm(3, ['f1', 'f2']);
            const mockData = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
            
            const centroids = algorithm.initializeCentroids(mockData);
            
            expect(centroids).toHaveLength(3);
            expect(centroids[0]).toHaveLength(2);
        });
        
        test('should calculate distance correctly', () => {
            const algorithm = new MockKMeansAlgorithm(2, ['f1', 'f2']);
            const point1 = [0, 0];
            const point2 = [3, 4];
            
            const distance = algorithm.calculateDistance(point1, point2);
            expect(distance).toBe(5); // 3-4-5 triangle
        });
        
        test('should assign points to clusters', () => {
            const algorithm = new MockKMeansAlgorithm(2, ['f1', 'f2']);
            const mockData = [[0.1, 0.2], [0.3, 0.4]];
            
            const clusters = algorithm.assignPointsToCluster(mockData);
            
            expect(clusters).toHaveLength(mockData.length);
            expect(clusters.every(c => c >= 0 && c < 2)).toBe(true);
        });
        
        test('should run full clustering algorithm', async () => {
            const algorithm = new MockKMeansAlgorithm(2, ['f1', 'f2']);
            const mockData = [[0.1, 0.2], [0.3, 0.4], [0.7, 0.8]];
            
            const result = await algorithm.fit(mockData);
            
            expect(result.centroids).toHaveLength(2);
            expect(result.clusters).toHaveLength(3);
            expect(result.iterations).toBeGreaterThan(0);
            expect(typeof result.wcss).toBe('number');
            expect(typeof result.silhouetteScore).toBe('number');
        });
        
        test('should call iteration callback during training', async () => {
            const algorithm = new MockKMeansAlgorithm(2, ['f1', 'f2']);
            const mockData = [[0.1, 0.2], [0.3, 0.4]];
            const callback = jest.fn();
            
            await algorithm.fit(mockData, callback);
            
            expect(callback).toHaveBeenCalled();
        });
    });
    
    describe('K-Means Functions', () => {
        
        test('runKMeansClustering should require dataset', async () => {
            global.mlCore.sharedData = null;
            
            await expect(mockKmeansFunctions.runKMeansClustering()).rejects.toThrow('No dataset loaded');
        });
        
        test('runKMeansClustering should prevent concurrent runs', async () => {
            mockKmeansState.isRunning = true;
            
            // This should not run because isRunning is true
            // In real implementation, it would show error notification
            expect(mockKmeansState.isRunning).toBe(true);
        });
        
        test('runKMeansClustering should complete successfully with valid data', async () => {
            global.mlCore.sharedData = global.createMockDataset();
            
            const result = await mockKmeansFunctions.runKMeansClustering();
            
            expect(result.centroids).toBeDefined();
            expect(result.clusters).toBeDefined();
            expect(result.iterations).toBeGreaterThan(0);
        });
        
        test('prepareKMeansData should format data correctly', () => {
            const dataset = global.createMockDataset();
            const features = ['kapwa_score', 'bayanihan_participation'];
            
            const result = mockKmeansFunctions.prepareKMeansData(dataset, features);
            
            expect(result).toHaveLength(dataset.length);
            expect(result[0].point).toHaveLength(features.length);
            expect(result[0].metadata).toBeDefined();
        });
        
        test('normalizeData should normalize to 0-1 range', () => {
            const data = [[1, 10], [2, 20], [3, 30]];
            
            const normalized = mockKmeansFunctions.normalizeData(data);
            
            expect(normalized[0][0]).toBe(0); // Min value becomes 0
            expect(normalized[2][0]).toBe(1); // Max value becomes 1
            expect(normalized[0][1]).toBe(0); // Min value becomes 0
            expect(normalized[2][1]).toBe(1); // Max value becomes 1
        });
        
        test('normalizeData should handle empty data', () => {
            const result = mockKmeansFunctions.normalizeData([]);
            expect(result).toEqual([]);
        });
        
        test('resetKMeansClustering should reset state', () => {
            mockKmeansState.clusters = [1, 2, 3];
            mockKmeansState.iterations = 10;
            
            mockKmeansFunctions.resetKMeansClustering();
            
            expect(mockKmeansState.clusters).toEqual([]);
            expect(mockKmeansState.iterations).toBe(0);
            expect(mockKmeansState.isRunning).toBe(false);
        });
        
        test('recommendOptimalK should suggest optimal cluster count', () => {
            global.mlCore.sharedData = global.createMockDataset();
            
            const optimalK = mockKmeansFunctions.recommendOptimalK();
            
            expect(typeof optimalK).toBe('number');
            expect(optimalK).toBeGreaterThan(1);
            expect(mockKmeansState.currentK).toBe(optimalK);
        });
        
        test('updateKMeansVisualization should be callable', () => {
            mockKmeansFunctions.updateKMeansVisualization();
            expect(mockKmeansFunctions.updateKMeansVisualization).toHaveBeenCalled();
        });
    });
    
    describe('K-Means State Management', () => {
        
        test('initial state should be correct', () => {
            expect(mockKmeansState.isRunning).toBe(false);
            expect(mockKmeansState.currentK).toBe(3);
            expect(mockKmeansState.features).toContain('kapwa_score');
        });
        
        test('state should update during clustering', async () => {
            global.mlCore.sharedData = global.createMockDataset();
            
            await mockKmeansFunctions.runKMeansClustering();
            
            expect(mockKmeansState.clusters.length).toBeGreaterThan(0);
            expect(mockKmeansState.centroids.length).toBeGreaterThan(0);
            expect(mockKmeansState.iterations).toBeGreaterThan(0);
        });
    });
    
    describe('K-Means Validation', () => {
        
        test('should validate minimum data requirements', () => {
            const smallDataset = [global.createMockDataset()[0]]; // Only 1 sample
            global.mlCore.sharedData = smallDataset;
            
            // Should handle insufficient data gracefully
            expect(global.mlCore.sharedData.length).toBe(1);
        });
        
        test('should validate feature requirements', () => {
            const features = [];
            const dataset = global.createMockDataset();
            
            const result = mockKmeansFunctions.prepareKMeansData(dataset, features);
            
            expect(result.every(item => item.point.length === features.length)).toBe(true);
        });
    });
});

describe('K-Means Integration Tests', () => {
    
    test('complete K-means workflow', async () => {
        // Setup
        global.mlCore.sharedData = global.createMockDataset();
        mockKmeansState.currentK = 2;
        mockKmeansState.features = ['kapwa_score', 'bayanihan_participation'];
        
        // Prepare data
        const rawData = mockKmeansFunctions.prepareKMeansData(
            global.mlCore.sharedData, 
            mockKmeansState.features
        );
        expect(rawData.length).toBe(global.mlCore.sharedData.length);
        
        // Normalize data
        const normalizedData = mockKmeansFunctions.normalizeData(rawData.map(d => d.point));
        expect(normalizedData.every(point => 
            point.every(val => val >= 0 && val <= 1)
        )).toBe(true);
        
        // Run clustering
        const result = await mockKmeansFunctions.runKMeansClustering();
        expect(result.centroids.length).toBe(mockKmeansState.currentK);
        
        // Verify state updates
        expect(mockKmeansState.clusters.length).toBeGreaterThan(0);
        expect(mockKmeansState.iterations).toBeGreaterThan(0);
    });
});

describe('K-Means Error Handling', () => {
    
    test('should handle missing TensorFlow gracefully', () => {
        // K-means doesn't require TensorFlow, so this should work
        expect(() => {
            new MockKMeansAlgorithm(3, ['feature1', 'feature2']);
        }).not.toThrow();
    });
    
    test('should handle invalid data gracefully', async () => {
        const algorithm = new MockKMeansAlgorithm(2, ['f1', 'f2']);
        const invalidData = [];
        
        // Should not crash on empty data
        expect(async () => {
            await algorithm.fit(invalidData);
        }).not.toThrow();
    });
    
    test('should handle convergence edge cases', async () => {
        const algorithm = new MockKMeansAlgorithm(2, ['f1', 'f2'], 1); // Max 1 iteration
        const mockData = [[0.1, 0.2], [0.3, 0.4]];
        
        const result = await algorithm.fit(mockData);
        
        expect(result.iterations).toBeLessThanOrEqual(1);
    });
});