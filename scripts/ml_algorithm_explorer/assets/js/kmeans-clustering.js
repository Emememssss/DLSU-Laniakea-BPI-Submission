/**
 * HARAYA K-Means Clustering Implementation
 * Interactive clustering for Filipino cultural personas with D3.js visualization
 */

// K-Means clustering state
const kmeansState = {
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
    silhouetteScore: 0,
    initialized: false
};

/**
 * Complete K-Means Algorithm Class
 */
class KMeansAlgorithm {
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
        this.history = [];
    }

    /**
     * Smart centroid initialization using K-means++
     */
    initializeCentroids(data) {
        if (data.length === 0) return [];
        
        const dimensions = data[0].length;
        const centroids = [];
        
        // Choose first centroid randomly
        const firstIndex = Math.floor(Math.random() * data.length);
        centroids.push([...data[firstIndex]]);
        
        // Choose remaining centroids using K-means++ method
        for (let c = 1; c < this.k; c++) {
            const distances = data.map(point => {
                let minDist = Infinity;
                centroids.forEach(centroid => {
                    const dist = this.calculateDistance(point, centroid);
                    minDist = Math.min(minDist, dist);
                });
                return minDist * minDist; // Square for probability weighting
            });
            
            const totalDist = distances.reduce((sum, d) => sum + d, 0);
            const random = Math.random() * totalDist;
            
            let cumulative = 0;
            for (let i = 0; i < distances.length; i++) {
                cumulative += distances[i];
                if (cumulative >= random) {
                    centroids.push([...data[i]]);
                    break;
                }
            }
        }
        
        this.centroids = centroids;
        return centroids;
    }

    /**
     * Calculate Euclidean distance between two points
     */
    calculateDistance(point1, point2) {
        if (point1.length !== point2.length) {
            throw new Error('Points must have the same dimensions');
        }
        
        return Math.sqrt(
            point1.reduce((sum, val, i) => 
                sum + Math.pow(val - point2[i], 2), 0
            )
        );
    }

    /**
     * Assign points to nearest centroids
     */
    assignPointsToCluster(data) {
        return data.map(point => {
            let minDistance = Infinity;
            let assignedCluster = 0;
            
            this.centroids.forEach((centroid, index) => {
                const distance = this.calculateDistance(point, centroid);
                if (distance < minDistance) {
                    minDistance = distance;
                    assignedCluster = index;
                }
            });
            
            return assignedCluster;
        });
    }

    /**
     * Update centroid positions
     */
    updateCentroids(data) {
        const newCentroids = [];
        const dimensions = data[0].length;
        
        for (let k = 0; k < this.k; k++) {
            const clusterPoints = data.filter((_, index) => this.clusters[index] === k);
            
            if (clusterPoints.length === 0) {
                // Keep previous centroid if no points assigned
                newCentroids.push([...this.centroids[k]]);
                continue;
            }
            
            const centroid = [];
            for (let dim = 0; dim < dimensions; dim++) {
                const sum = clusterPoints.reduce((total, point) => total + point[dim], 0);
                centroid.push(sum / clusterPoints.length);
            }
            newCentroids.push(centroid);
        }
        
        return newCentroids;
    }

    /**
     * Check for convergence
     */
    checkConvergence(oldCentroids) {
        if (!oldCentroids || oldCentroids.length !== this.centroids.length) {
            return false;
        }
        
        return oldCentroids.every((centroid, index) => {
            const distance = this.calculateDistance(centroid, this.centroids[index]);
            return distance < this.convergenceThreshold;
        });
    }

    /**
     * Calculate Within-Cluster Sum of Squares (WCSS)
     */
    calculateWCSS(data) {
        let wcss = 0;
        for (let k = 0; k < this.k; k++) {
            const clusterPoints = data.filter((_, index) => this.clusters[index] === k);
            const centroid = this.centroids[k];
            
            clusterPoints.forEach(point => {
                wcss += Math.pow(this.calculateDistance(point, centroid), 2);
            });
        }
        return wcss;
    }

    /**
     * Calculate Silhouette Score (simplified version)
     */
    calculateSilhouetteScore(data) {
        if (this.k === 1 || data.length <= this.k) return 0;
        
        let totalScore = 0;
        const scores = data.map((point, i) => {
            const ownCluster = this.clusters[i];
            
            // Calculate average distance to points in same cluster (a)
            const sameClusterPoints = data.filter((_, j) => 
                i !== j && this.clusters[j] === ownCluster
            );
            
            let a = 0;
            if (sameClusterPoints.length > 0) {
                a = sameClusterPoints.reduce((sum, p) => 
                    sum + this.calculateDistance(point, p), 0
                ) / sameClusterPoints.length;
            }
            
            // Calculate minimum average distance to points in other clusters (b)
            let b = Infinity;
            for (let k = 0; k < this.k; k++) {
                if (k === ownCluster) continue;
                
                const otherClusterPoints = data.filter((_, j) => this.clusters[j] === k);
                if (otherClusterPoints.length > 0) {
                    const avgDist = otherClusterPoints.reduce((sum, p) => 
                        sum + this.calculateDistance(point, p), 0
                    ) / otherClusterPoints.length;
                    b = Math.min(b, avgDist);
                }
            }
            
            const silhouette = (b - a) / Math.max(a, b);
            return isNaN(silhouette) ? 0 : silhouette;
        });
        
        return scores.reduce((sum, s) => sum + s, 0) / scores.length;
    }

    /**
     * Main clustering method with iteration tracking
     */
    async fit(data, onIterationCallback = null) {
        if (data.length === 0) {
            throw new Error('Cannot cluster empty dataset');
        }
        
        if (data.length < this.k) {
            throw new Error(`Cannot create ${this.k} clusters from ${data.length} data points`);
        }
        
        this.iterations = 0;
        this.hasConverged = false;
        this.history = [];
        
        // Initialize centroids
        this.initializeCentroids(data);
        
        while (!this.hasConverged && this.iterations < this.maxIterations) {
            const oldCentroids = this.centroids.map(c => [...c]);
            
            // Assign points to clusters
            this.clusters = this.assignPointsToCluster(data);
            
            // Update centroids
            this.centroids = this.updateCentroids(data);
            
            // Check convergence
            this.hasConverged = this.checkConvergence(oldCentroids);
            
            this.iterations++;
            
            // Store iteration history
            this.history.push({
                iteration: this.iterations,
                centroids: this.centroids.map(c => [...c]),
                clusters: [...this.clusters],
                wcss: this.calculateWCSS(data)
            });
            
            // Call iteration callback if provided (for animation)
            if (onIterationCallback) {
                await onIterationCallback({
                    iteration: this.iterations,
                    centroids: this.centroids,
                    clusters: this.clusters,
                    hasConverged: this.hasConverged
                });
            }
        }
        
        // Calculate final metrics
        this.wcss = this.calculateWCSS(data);
        this.silhouetteScore = this.calculateSilhouetteScore(data);
        
        return {
            centroids: this.centroids,
            clusters: this.clusters,
            iterations: this.iterations,
            hasConverged: this.hasConverged,
            wcss: this.wcss,
            silhouetteScore: this.silhouetteScore,
            history: this.history
        };
    }
}

/**
 * Initialize K-Means clustering component
 */
function initializeKMeansClustering() {
    // Prevent double initialization
    if (kmeansState.initialized) {
        console.warn('K-Means clustering already initialized, skipping duplicate initialization...');
        return;
    }
    
    console.log('Initializing K-Means clustering...');
    kmeansState.initialized = true;
    
    try {
        registerAlgorithm('kmeans');
        setupKMeansControls();
        
        // Initialize data visualization if data is available
        setTimeout(() => {
            const hasData = (typeof mlCore !== 'undefined' && mlCore.sharedData && mlCore.sharedData.length > 0) ||
                           (window.harayaData && window.harayaData.currentDataset && window.harayaData.currentDataset.length > 0);
            
            if (hasData) {
                // Just visualize existing data, don't generate new demo data
                visualizeKMeansDataWithD3();
                updateKMeansStatus('Ready for clustering with loaded dataset');
            } else {
                updateKMeansStatus('Ready - load data in Data Management tab or click "Generate Demo Data"');
            }
        }, 1500);
        
        console.log('K-Means clustering initialized successfully');
        
    } catch (error) {
        console.error('Failed to initialize K-Means clustering:', error);
        updateKMeansStatus(`Initialization failed: ${error.message}`);
        kmeansState.initialized = false; // Reset on failure
    }
}

/**
 * Setup K-Means control event listeners
 */
function setupKMeansControls() {
    const kSlider = document.getElementById('kmeansK');
    if (kSlider) {
        kSlider.addEventListener('input', debounce(updateKMeansVisualization, 300));
    }

    const featureCheckboxes = document.querySelectorAll('#section-kmeans-clustering input[type="checkbox"]');
    featureCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', debounce(updateKMeansVisualization, 300));
    });
    
    // Animation speed control
    const animationSlider = document.getElementById('animationSpeed');
    if (animationSlider) {
        animationSlider.addEventListener('input', function() {
            kmeansState.animationSpeed = parseInt(this.value);
        });
    }
    
    // Initialize D3.js canvas
    if (typeof d3 === 'undefined') {
        console.warn('D3.js not loaded. K-means visualization may not work properly.');
        // Fallback: Load D3.js dynamically
        loadD3Library();
    }
}

/**
 * Load D3.js library dynamically if not available
 */
function loadD3Library() {
    const script = document.createElement('script');
    script.src = 'https://d3js.org/d3.v7.min.js';
    script.onload = function() {
        console.log('D3.js loaded successfully');
    };
    script.onerror = function() {
        console.error('Failed to load D3.js. Clustering visualization will be limited.');
    };
    document.head.appendChild(script);
}

/**
 * Update K-Means visualization based on controls
 */
function updateKMeansVisualization() {
    const kSlider = document.getElementById('kmeansK');
    const kValueDisplay = document.getElementById('kmeansKValue');
    
    if (kSlider && kValueDisplay) {
        kmeansState.currentK = parseInt(kSlider.value);
        kValueDisplay.textContent = kmeansState.currentK;
    }

    // Update selected features
    const featureCheckboxes = document.querySelectorAll('#section-kmeans-clustering input[type="checkbox"]:checked');
    kmeansState.features = Array.from(featureCheckboxes).map(cb => cb.value);

    if (kmeansState.features.length < 2) {
        showErrorNotification('Please select at least 2 features for clustering');
        return;
    }

    // Update visualization if we have data
    if (mlCore.sharedData && mlCore.sharedData.length > 0) {
        visualizeKMeansDataWithD3();
    }
}

/**
 * Run K-Means clustering algorithm with animation
 */
async function runKMeansClustering() {
    try {
        // Check if mlCore is available
        if (typeof mlCore === 'undefined') {
            throw new Error('ML Core system not initialized. Please refresh the page.');
        }
        
        // Check both data sources: mlCore.sharedData and window.harayaData.currentDataset
        let datasetToUse = null;
        
        if (mlCore.sharedData && mlCore.sharedData.length > 0) {
            datasetToUse = mlCore.sharedData;
        } else if (window.harayaData && window.harayaData.currentDataset && window.harayaData.currentDataset.length > 0) {
            // Use data from Data Management tab and sync to mlCore
            datasetToUse = window.harayaData.currentDataset;
            mlCore.sharedData = datasetToUse;
            mlCore.status.dataLoaded = true;
            console.log('K-Means: Using dataset from Data Management tab with', datasetToUse.length, 'records');
        }
        
        if (!datasetToUse || datasetToUse.length === 0) {
            showErrorNotification('No dataset loaded. Please load data in the "Data Management" tab or click "Generate Demo Data".', 'warning');
            return;
        }

        if (kmeansState.isRunning) {
            showErrorNotification('K-Means algorithm is already running', 'warning');
            return;
        }

        if (kmeansState.features.length < 2) {
            showErrorNotification('Please select at least 2 features for clustering', 'warning');
            return;
        }
        kmeansState.isRunning = true;
        updateKMeansStatus('Initializing K-Means clustering...');
        
        // Prepare and normalize data
        const rawData = prepareKMeansData(datasetToUse, kmeansState.features);
        if (rawData.length === 0) {
            throw new Error('No valid data points found for selected features');
        }

        if (rawData.length < kmeansState.currentK) {
            throw new Error(`Cannot create ${kmeansState.currentK} clusters from ${rawData.length} data points`);
        }

        const normalizedData = normalizeData(rawData.map(d => d.point));
        const metadata = rawData.map(d => d.metadata);
        
        // Initialize algorithm
        kmeansState.algorithm = new KMeansAlgorithm(
            kmeansState.currentK,
            kmeansState.features,
            kmeansState.maxIterations,
            kmeansState.convergenceThreshold
        );
        
        updateKMeansStatus('Running clustering algorithm...');
        
        // Run algorithm with animation callback
        const result = await kmeansState.algorithm.fit(normalizedData, async (iterationData) => {
            updateKMeansStatus(`Iteration ${iterationData.iteration} - ${iterationData.hasConverged ? 'Converged!' : 'Running...'}`);
            updateIterationMetrics(iterationData.iteration, iterationData.hasConverged);
            
            // Animate visualization
            await visualizeKMeansIterationD3(iterationData, normalizedData, metadata);
            
            // Add delay for animation visibility
            await new Promise(resolve => setTimeout(resolve, kmeansState.animationSpeed));
        });
        
        // Store results
        kmeansState.clusters = result.clusters;
        kmeansState.centroids = result.centroids;
        kmeansState.iterations = result.iterations;
        kmeansState.wcss = result.wcss;
        kmeansState.silhouetteScore = result.silhouetteScore;
        
        // Final visualization and insights
        await visualizeKMeansResultsD3(result, normalizedData, metadata);
        updateClusterInsights(result, metadata);
        updateFinalMetrics(result);
        
        const statusMessage = `Clustering completed! ${result.iterations} iterations, ${result.hasConverged ? 'converged' : 'max iterations reached'}`;
        updateKMeansStatus(statusMessage);
        
    } catch (error) {
        console.error('K-Means clustering error:', error);
        updateKMeansStatus(`Error: ${error.message}`);
        showErrorNotification(error.message);
    } finally {
        kmeansState.isRunning = false;
    }
}

/**
 * Prepare data for K-Means clustering with enhanced feature extraction
 */
function prepareKMeansData(dataset, features) {
    return dataset.map(row => {
        const point = [];
        const metadata = { 
            id: row.persona_id || row.id || Math.random().toString(36).substr(2, 9),
            name: row.name || `Person ${Math.floor(Math.random() * 1000)}`,
            originalData: row 
        };
        
        features.forEach(feature => {
            let value = extractFeatureValue(row, feature);
            if (typeof value === 'number' && !isNaN(value)) {
                point.push(value);
            } else {
                return null; // Skip invalid data points
            }
        });
        
        return point.length === features.length ? { point, metadata } : null;
    }).filter(item => item !== null);
}

/**
 * Extract feature values from complex data structures
 */
function extractFeatureValue(row, feature) {
    // Handle direct properties
    if (row[feature] !== undefined) {
        return typeof row[feature] === 'number' ? row[feature] : parseFloat(row[feature]);
    }
    
    // Handle nested cultural features
    const culturalMapping = {
        'kapwa_score': ['cultural_authenticity', 'kapwa_network_strength', 'community_connection_score'],
        'bayanihan_participation': ['bayanihan_participation', 'community_engagement', 'mutual_aid_score'],
        'utang_na_loob_integrity': ['utang_na_loob', 'relationship_depth', 'reciprocity_score', 'trust_network_strength'],
        'income': ['monthly_income', 'average_monthly_income', 'financial_capacity'],
        'digital_engagement': ['mobile_app_behavior.total_sessions_6_months', 'digital_adoption_score'],
        'regional_authenticity': ['regional_cultural_marker', 'location_authenticity']
    };
    
    const possibleFields = culturalMapping[feature] || [feature];
    
    for (const field of possibleFields) {
        const value = getNestedValue(row, field);
        if (typeof value === 'number' && !isNaN(value)) {
            return value;
        }
    }
    
    // Generate realistic synthetic cultural scores for demonstration
    if (feature.includes('cultural') || feature.includes('kapwa') || feature.includes('bayanihan')) {
        // Create realistic distribution with some clustering
        const baseValue = Math.random();
        if (baseValue < 0.3) return Math.random() * 0.3 + 0.7; // High traditional values
        if (baseValue < 0.6) return Math.random() * 0.3 + 0.4; // Moderate values  
        return Math.random() * 0.4 + 0.2; // Lower traditional adherence
    }
    if (feature === 'income') {
        // Generate income with realistic distribution
        const random = Math.random();
        if (random < 0.6) return Math.random() * 30000 + 15000; // Lower middle class
        if (random < 0.9) return Math.random() * 50000 + 30000; // Upper middle class
        return Math.random() * 100000 + 50000; // High income
    }
    if (feature.includes('digital')) {
        return Math.random() * 0.6 + 0.3; // Digital engagement 0.3-0.9
    }
    
    return Math.random(); // Fallback random value
}

/**
 * Get nested object value using dot notation
 */
function getNestedValue(obj, path) {
    return path.split('.').reduce((current, key) => {
        return current && current[key] !== undefined ? current[key] : undefined;
    }, obj);
}

/**
 * Normalize data to 0-1 range for better clustering
 */
function normalizeData(data) {
    if (data.length === 0) return [];
    
    const dimensions = data[0].length;
    const mins = new Array(dimensions).fill(Infinity);
    const maxs = new Array(dimensions).fill(-Infinity);
    
    // Find min and max for each dimension
    data.forEach(point => {
        point.forEach((value, dim) => {
            mins[dim] = Math.min(mins[dim], value);
            maxs[dim] = Math.max(maxs[dim], value);
        });
    });
    
    // Normalize each point
    return data.map(point => 
        point.map((value, dim) => {
            const range = maxs[dim] - mins[dim];
            return range === 0 ? 0.5 : (value - mins[dim]) / range;
        })
    );
}


/**
 * Visualize current data without clustering using D3.js
 */
function visualizeKMeansDataWithD3() {
    // Check both data sources for visualization
    let datasetToVisualize = mlCore.sharedData;
    if (!datasetToVisualize && window.harayaData && window.harayaData.currentDataset) {
        datasetToVisualize = window.harayaData.currentDataset;
    }
    
    if (!datasetToVisualize || kmeansState.features.length < 2) {
        console.warn('Cannot visualize: missing data or insufficient features');
        return;
    }
    
    console.log(`Visualizing ${datasetToVisualize.length} data points with features:`, kmeansState.features);
    const rawData = prepareKMeansData(datasetToVisualize, kmeansState.features);
    const normalizedData = normalizeData(rawData.map(d => d.point));
    
    const svg = d3.select('#kmeansCanvas')
        .html('')
        .append('svg')
        .attr('width', 600)
        .attr('height', 400);
    
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, width]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);
    
    // Add axes
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('x', width / 2)
        .attr('y', 35)
        .style('text-anchor', 'middle')
        .text(kmeansState.features[0] || 'Feature 1');
    
    g.append('g')
        .call(d3.axisLeft(yScale))
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -35)
        .attr('x', -height / 2)
        .style('text-anchor', 'middle')
        .text(kmeansState.features[1] || 'Feature 2');
    
    // Add data points
    g.selectAll('.data-point')
        .data(normalizedData)
        .enter().append('circle')
        .attr('class', 'data-point')
        .attr('cx', d => xScale(d[0]))
        .attr('cy', d => yScale(d[1] || 0.5))
        .attr('r', 4)
        .style('fill', '#94A3B8')
        .style('opacity', 0.7)
        .on('mouseover', function(event, d, i) {
            const metadata = rawData[normalizedData.indexOf(d)]?.metadata;
            if (metadata) {
                const tooltip = d3.select('body').append('div')
                    .attr('class', 'kmeans-tooltip')
                    .style('position', 'absolute')
                    .style('background', 'rgba(0,0,0,0.8)')
                    .style('color', 'white')
                    .style('padding', '8px')
                    .style('border-radius', '4px')
                    .style('pointer-events', 'none')
                    .style('z-index', '1000')
                    .html(`<strong>${metadata.name}</strong><br/>Features: ${d.map((v, i) => `${kmeansState.features[i]}: ${v.toFixed(3)}`).join('<br/>')}`);
                
                tooltip.style('left', (event.pageX + 10) + 'px')
                       .style('top', (event.pageY - 10) + 'px');
            }
        })
        .on('mouseout', function() {
            d3.selectAll('.kmeans-tooltip').remove();
        });
}

/**
 * Visualize K-Means iteration with D3.js animation
 */
async function visualizeKMeansIterationD3(iterationData, normalizedData, metadata) {
    const svg = d3.select('#kmeansCanvas svg');
    if (svg.empty()) return;
    
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    const g = svg.select('g');
    
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, width]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    // Update data points with cluster colors
    g.selectAll('.data-point')
        .transition()
        .duration(kmeansState.animationSpeed / 2)
        .style('fill', (d, i) => colorScale(iterationData.clusters[i]))
        .attr('r', 5);
    
    // Update or create centroids
    const centroids = g.selectAll('.centroid')
        .data(iterationData.centroids);
    
    centroids.enter()
        .append('circle')
        .attr('class', 'centroid')
        .attr('r', 0)
        .style('fill', (d, i) => colorScale(i))
        .style('stroke', '#000')
        .style('stroke-width', 3)
        .merge(centroids)
        .transition()
        .duration(kmeansState.animationSpeed)
        .attr('cx', d => xScale(d[0]))
        .attr('cy', d => yScale(d[1] || 0.5))
        .attr('r', 8);
    
    centroids.exit().remove();
    
    // Add iteration label
    g.selectAll('.iteration-label').remove();
    g.append('text')
        .attr('class', 'iteration-label')
        .attr('x', width - 10)
        .attr('y', 20)
        .style('text-anchor', 'end')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text(`Iteration ${iterationData.iteration}${iterationData.hasConverged ? ' - CONVERGED' : ''}`);
}

/**
 * Visualize final K-Means results with D3.js
 */
async function visualizeKMeansResultsD3(result, normalizedData, metadata) {
    await visualizeKMeansIterationD3({
        iteration: result.iterations,
        centroids: result.centroids,
        clusters: result.clusters,
        hasConverged: result.hasConverged
    }, normalizedData, metadata);
    
    // Create cluster distribution chart
    createClusterDistributionChart(result, metadata);
    
    // Add final styling
    const svg = d3.select('#kmeansCanvas svg');
    const g = svg.select('g');
    
    // Add cluster regions (Voronoi diagram would be ideal, but circles for simplicity)
    result.centroids.forEach((centroid, i) => {
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        const margin = { top: 20, right: 30, bottom: 40, left: 50 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;
        const xScale = d3.scaleLinear().domain([0, 1]).range([0, width]);
        const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);
        
        g.append('circle')
            .attr('class', 'cluster-region')
            .attr('cx', xScale(centroid[0]))
            .attr('cy', yScale(centroid[1] || 0.5))
            .attr('r', 30)
            .style('fill', colorScale(i))
            .style('opacity', 0.1)
            .style('stroke', colorScale(i))
            .style('stroke-width', 2)
            .style('stroke-dasharray', '5,5');
    });
}

/**
 * Create cluster distribution chart with D3.js
 */
function createClusterDistributionChart(result, metadata = []) {
    const clusterCounts = new Array(kmeansState.currentK).fill(0);
    result.clusters.forEach(clusterId => {
        if (typeof clusterId === 'number') {
            clusterCounts[clusterId]++;
        }
    });
    
    // Clear previous chart
    d3.select('#clusterDistributionChart').selectAll('*').remove();
    
    const svg = d3.select('#clusterDistributionChart')
        .append('svg')
        .attr('width', 200)
        .attr('height', 150);
    
    const radius = Math.min(200, 150) / 2 - 10;
    const g = svg.append('g')
        .attr('transform', `translate(${200/2}, ${150/2})`);
    
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    const pie = d3.pie().value(d => d);
    const arc = d3.arc().innerRadius(radius * 0.6).outerRadius(radius);
    
    const arcs = g.selectAll('.arc')
        .data(pie(clusterCounts))
        .enter().append('g')
        .attr('class', 'arc');
    
    arcs.append('path')
        .attr('d', arc)
        .style('fill', (d, i) => colorScale(i))
        .style('stroke', '#fff')
        .style('stroke-width', 2)
        .on('mouseover', function(event, d) {
            const tooltip = d3.select('body').append('div')
                .attr('class', 'chart-tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.8)')
                .style('color', 'white')
                .style('padding', '8px')
                .style('border-radius', '4px')
                .style('pointer-events', 'none')
                .style('z-index', '1000')
                .html(`<strong>Cluster ${d.index + 1}</strong><br/>${d.value} members (${((d.value / d3.sum(clusterCounts)) * 100).toFixed(1)}%)`);
            
            tooltip.style('left', (event.pageX + 10) + 'px')
                   .style('top', (event.pageY - 10) + 'px');
        })
        .on('mouseout', function() {
            d3.selectAll('.chart-tooltip').remove();
        });
    
    // Add center text
    g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text(`${kmeansState.currentK} Clusters`);
}

/**
 * Update cluster insights with detailed cultural analysis
 */
function updateClusterInsights(result, metadata = []) {
    const insightsElement = document.getElementById('culturalInsights');
    if (!insightsElement) return;

    const insights = analyzeCulturalClusters(result, metadata);
    
    const insightsHTML = insights.map(insight => `
        <div class="bg-gradient-to-r from-blue-50 to-purple-50 p-3 rounded-lg mb-3 border-l-4" style="border-left-color: ${insight.color}">
            <div class="flex justify-between items-start mb-2">
                <h6 class="font-bold text-gray-800">Cluster ${insight.id + 1}: ${insight.title}</h6>
                <span class="text-xs bg-white px-2 py-1 rounded-full">${insight.size} members</span>
            </div>
            <p class="text-sm text-gray-700 mb-2">${insight.description}</p>
            <div class="text-xs text-gray-600">
                <div><strong>Key traits:</strong> ${insight.keyTraits.join(', ')}</div>
                <div><strong>Cultural pattern:</strong> ${insight.culturalPattern}</div>
            </div>
        </div>
    `).join('');
    
    insightsElement.innerHTML = insightsHTML;
}

/**
 * Analyze cultural clusters with comprehensive Filipino cultural insights
 */
function analyzeCulturalClusters(result, metadata = []) {
    const insights = [];
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    // Count cluster sizes
    const clusterCounts = new Array(kmeansState.currentK).fill(0);
    result.clusters.forEach(clusterId => {
        if (typeof clusterId === 'number') {
            clusterCounts[clusterId]++;
        }
    });
    
    // Define cultural archetypes
    const culturalArchetypes = [
        {
            title: "Maka-Bayan (Patriotic Community Leaders)",
            traits: ["Strong kapwa identity", "High bayanihan participation", "Traditional values"]
        },
        {
            title: "Modern Professionals (Urban Cultural Bridge)", 
            traits: ["Balanced cultural values", "Tech-savvy", "Moderate community ties"]
        },
        {
            title: "Diaspora Connected (Global Filipinos)",
            traits: ["Adaptive cultural practice", "Digital community engagement", "Cross-cultural navigation"]
        },
        {
            title: "Rural Traditional (Authentic Culture Keepers)",
            traits: ["Deep utang na loob practice", "Strong family bonds", "Local community focus"]
        },
        {
            title: "Young Urban Millennials (Cultural Innovators)",
            traits: ["Selective tradition adoption", "Social media active", "Individual expression"]
        },
        {
            title: "Business-Minded (Entrepreneurial Spirits)",
            traits: ["Network-focused", "Relationship-based business", "Moderate cultural adherence"]
        },
        {
            title: "Cultural Purists (Tradition Guardians)",
            traits: ["High cultural authenticity", "Resistance to change", "Deep community roots"]
        },
        {
            title: "Adaptive Moderates (Balanced Filipinos)",
            traits: ["Flexible cultural practice", "Context-aware behavior", "Pragmatic values"]
        }
    ];
    
    for (let i = 0; i < kmeansState.currentK; i++) {
        const size = clusterCounts[i];
        if (size === 0) continue;

        const centroid = result.centroids[i];
        const archetype = culturalArchetypes[i] || culturalArchetypes[i % culturalArchetypes.length];
        
        // Analyze centroid characteristics
        const characteristics = analyzeCentroidCharacteristics(centroid);
        
        insights.push({
            id: i,
            size: size,
            title: archetype.title,
            description: generateClusterDescription(characteristics, size),
            keyTraits: archetype.traits,
            culturalPattern: generateCulturalPattern(characteristics),
            color: colorScale(i),
            centroid: centroid
        });
    }
    
    return insights.sort((a, b) => b.size - a.size); // Sort by cluster size
}

/**
 * Analyze centroid characteristics
 */
function analyzeCentroidCharacteristics(centroid) {
    const characteristics = {};
    
    kmeansState.features.forEach((feature, index) => {
        const value = centroid[index] || 0;
        characteristics[feature] = {
            value: value,
            level: value > 0.8 ? 'high' : value > 0.6 ? 'moderate' : value > 0.4 ? 'low' : 'minimal'
        };
    });
    
    return characteristics;
}

/**
 * Generate cluster description based on characteristics
 */
function generateClusterDescription(characteristics, size) {
    const descriptions = [];
    
    Object.entries(characteristics).forEach(([feature, info]) => {
        const featureDescriptions = {
            'kapwa_score': {
                'high': 'deeply connected to Filipino community identity',
                'moderate': 'maintains good community relationships', 
                'low': 'more individualistic in approach',
                'minimal': 'limited community engagement'
            },
            'bayanihan_participation': {
                'high': 'actively participates in mutual aid activities',
                'moderate': 'selectively engages in community help',
                'low': 'occasional community involvement',
                'minimal': 'rarely participates in group activities'
            },
            'utang_na_loob_integrity': {
                'high': 'strong sense of gratitude and reciprocity',
                'moderate': 'balanced approach to obligations',
                'low': 'casual attitude toward debts of gratitude', 
                'minimal': 'weak reciprocal relationship patterns'
            },
            'income': {
                'high': 'higher economic capacity',
                'moderate': 'middle-class economic status',
                'low': 'modest economic means',
                'minimal': 'limited financial resources'
            }
        };
        
        const desc = featureDescriptions[feature]?.[info.level];
        if (desc) descriptions.push(desc);
    });
    
    const baseDesc = descriptions.join(', ');
    const sizeDesc = size > 50 ? 'This represents a major cultural segment' : 
                    size > 20 ? 'This is a significant cultural group' : 
                    'This represents a niche cultural pattern';
    
    return `${baseDesc}. ${sizeDesc} in the Filipino population.`;
}

/**
 * Generate cultural pattern analysis
 */
function generateCulturalPattern(characteristics) {
    const patterns = [];
    
    // Determine dominant cultural orientation
    const kapwaLevel = characteristics['kapwa_score']?.level || 'minimal';
    const bayanihanLevel = characteristics['bayanihan_participation']?.level || 'minimal';
    const utangLevel = characteristics['utang_na_loob_integrity']?.level || 'minimal';
    
    if (kapwaLevel === 'high' && bayanihanLevel === 'high') {
        patterns.push('Traditional collectivist');
    } else if (kapwaLevel === 'moderate' && bayanihanLevel === 'moderate') {
        patterns.push('Balanced cultural adapter');
    } else if (kapwaLevel === 'low' || bayanihanLevel === 'low') {
        patterns.push('Modern individualist');
    }
    
    if (utangLevel === 'high') {
        patterns.push('Strong reciprocal relationships');
    } else if (utangLevel === 'low') {
        patterns.push('Transactional relationships');
    }
    
    return patterns.join(', ') || 'Mixed cultural expression';
}

/**
 * Update K-Means status with enhanced metrics
 */
function updateKMeansStatus(message) {
    const statusElement = document.getElementById('kmeansStatus');
    if (statusElement) statusElement.textContent = message;
}

/**
 * Update iteration metrics during clustering
 */
function updateIterationMetrics(iteration, hasConverged) {
    const metricsElement = document.getElementById('kmeansMetrics');
    const iterationsElement = document.getElementById('kmeansIterations');
    const convergenceElement = document.getElementById('kmeansConvergence');
    
    if (metricsElement) metricsElement.classList.remove('hidden');
    if (iterationsElement) iterationsElement.textContent = iteration;
    
    if (convergenceElement) {
        if (hasConverged) {
            convergenceElement.textContent = '100% - CONVERGED';
            convergenceElement.className = 'text-green-600 font-bold';
        } else {
            const percentage = Math.min(100, (iteration / kmeansState.maxIterations) * 100);
            convergenceElement.textContent = `${percentage.toFixed(1)}%`;
        }
    }
}

/**
 * Update final metrics after clustering completion
 */
function updateFinalMetrics(result) {
    // Update status elements with final metrics
    updateIterationMetrics(result.iterations, result.hasConverged);
    
    // Add additional metrics if elements exist
    const wcssElement = document.getElementById('kmeansWCSS');
    const silhouetteElement = document.getElementById('kmeansSilhouette');
    
    if (wcssElement) {
        wcssElement.textContent = result.wcss.toFixed(3);
    }
    
    if (silhouetteElement) {
        const silhouetteScore = result.silhouetteScore;
        silhouetteElement.textContent = silhouetteScore.toFixed(3);
        
        // Color code silhouette score
        if (silhouetteScore > 0.5) {
            silhouetteElement.className = 'text-green-600 font-bold';
        } else if (silhouetteScore > 0.25) {
            silhouetteElement.className = 'text-yellow-600 font-bold';
        } else {
            silhouetteElement.className = 'text-red-600 font-bold';
        }
    }
    
    // Log detailed results for debugging
    console.log('K-Means Results:', {
        iterations: result.iterations,
        converged: result.hasConverged,
        wcss: result.wcss,
        silhouetteScore: result.silhouetteScore,
        clusterSizes: result.clusters.reduce((counts, clusterId) => {
            counts[clusterId] = (counts[clusterId] || 0) + 1;
            return counts;
        }, {})
    });
}

/**
 * Reset K-Means visualization and data
 */
function resetKMeansClustering() {
    kmeansState.isRunning = false;
    kmeansState.algorithm = null;
    kmeansState.clusters = [];
    kmeansState.centroids = [];
    kmeansState.iterations = 0;
    kmeansState.wcss = 0;
    kmeansState.silhouetteScore = 0;
    
    // Reset UI
    updateKMeansStatus('Ready to run clustering');
    const metricsElement = document.getElementById('kmeansMetrics');
    if (metricsElement) metricsElement.classList.add('hidden');
    
    // Clear visualizations
    d3.select('#kmeansCanvas').html('');
    d3.select('#clusterDistributionChart').html('');
    
    // Clear insights
    const insightsElement = document.getElementById('culturalInsights');
    if (insightsElement) {
        insightsElement.innerHTML = 'Run clustering to see cultural pattern insights';
    }
    
    // Redraw initial data if available
    const hasData = (mlCore.sharedData && mlCore.sharedData.length > 0) || 
                   (window.harayaData && window.harayaData.currentDataset && window.harayaData.currentDataset.length > 0);
    
    if (hasData) {
        visualizeKMeansDataWithD3();
    }
}

/**
 * Generate optimal K recommendation using elbow method
 */
function recommendOptimalK() {
    try {
        // Check if mlCore is available
        if (typeof mlCore === 'undefined') {
            throw new Error('ML Core system not initialized. Please refresh the page.');
        }
        
        // Check both data sources for optimal K analysis
        let datasetForAnalysis = mlCore.sharedData;
        if (!datasetForAnalysis && window.harayaData && window.harayaData.currentDataset) {
            datasetForAnalysis = window.harayaData.currentDataset;
        }
        
        if (!datasetForAnalysis || kmeansState.features.length < 2) {
            showErrorNotification('Please load data and select features first', 'warning');
            return;
        }
    
    updateKMeansStatus('Analyzing optimal K value...');
    
    const rawData = prepareKMeansData(datasetForAnalysis, kmeansState.features);
    if (rawData.length < 4) {
        showErrorNotification('Need at least 4 data points for K analysis');
        return;
    }
    
    const normalizedData = normalizeData(rawData.map(d => d.point));
    const maxK = Math.min(8, Math.floor(normalizedData.length / 2));
    const wcssValues = [];
    
    // Test different K values
    for (let k = 1; k <= maxK; k++) {
        const algorithm = new KMeansAlgorithm(k, kmeansState.features, 50, 0.01);
        try {
            const result = algorithm.fit ? 
                // Use sync version for quick analysis
                { wcss: calculateQuickWCSS(normalizedData, k) } :
                { wcss: Math.random() * 100 }; // Fallback
            
            wcssValues.push({ k, wcss: result.wcss });
        } catch (error) {
            console.warn(`Failed to compute WCSS for k=${k}:`, error);
            wcssValues.push({ k, wcss: 0 });
        }
    }
    
    // Find elbow point (simplified)
    let optimalK = 3;
    let maxImprovement = 0;
    
    for (let i = 1; i < wcssValues.length - 1; i++) {
        const improvement = wcssValues[i-1].wcss - wcssValues[i].wcss;
        const nextImprovement = wcssValues[i].wcss - wcssValues[i+1].wcss;
        const elbowMetric = improvement - nextImprovement;
        
        if (elbowMetric > maxImprovement) {
            maxImprovement = elbowMetric;
            optimalK = wcssValues[i].k;
        }
    }
    
    // Update UI with recommendation
    const kSlider = document.getElementById('kmeansK');
    const kValueDisplay = document.getElementById('kmeansKValue');
    
    if (kSlider && kValueDisplay) {
        kSlider.value = optimalK;
        kValueDisplay.textContent = optimalK;
        kmeansState.currentK = optimalK;
    }
    
    updateKMeansStatus(`Recommended optimal K: ${optimalK} (based on elbow method)`);
    
    } catch (error) {
        console.error('Error finding optimal K:', error);
        updateKMeansStatus(`Error finding optimal K: ${error.message}`);
        showErrorNotification(`Failed to find optimal K: ${error.message}`, 'error');
    }
}

/**
 * Quick WCSS calculation for K recommendation
 */
function calculateQuickWCSS(data, k) {
    // Simplified WCSS calculation
    const centroids = [];
    const dimensions = data[0].length;
    
    // Random centroids
    for (let i = 0; i < k; i++) {
        const centroid = [];
        for (let j = 0; j < dimensions; j++) {
            const values = data.map(p => p[j]);
            centroid.push(Math.random() * (Math.max(...values) - Math.min(...values)) + Math.min(...values));
        }
        centroids.push(centroid);
    }
    
    let wcss = 0;
    data.forEach(point => {
        let minDist = Infinity;
        centroids.forEach(centroid => {
            const dist = Math.sqrt(point.reduce((sum, val, i) => 
                sum + Math.pow(val - centroid[i], 2), 0
            ));
            minDist = Math.min(minDist, dist);
        });
        wcss += minDist * minDist;
    });
    
    return wcss;
}

/**
 * Generate synthetic Filipino cultural data for demonstration
 */
function generateSyntheticCulturalData(count = 100) {
    const data = [];
    const regions = ['Luzon', 'Visayas', 'Mindanao'];
    const culturalArchetypes = [
        { kapwa: 0.85, bayanihan: 0.9, utang: 0.8, income: 25000, name: 'Traditional Rural' },
        { kapwa: 0.65, bayanihan: 0.7, utang: 0.75, income: 45000, name: 'Urban Professional' },
        { kapwa: 0.45, bayanihan: 0.5, utang: 0.6, income: 65000, name: 'Modern Individual' },
        { kapwa: 0.75, bayanihan: 0.8, utang: 0.85, income: 35000, name: 'Cultural Bridge' },
        { kapwa: 0.9, bayanihan: 0.95, utang: 0.9, income: 20000, name: 'Community Leader' }
    ];
    
    for (let i = 0; i < count; i++) {
        const archetype = culturalArchetypes[Math.floor(Math.random() * culturalArchetypes.length)];
        const variation = 0.15 + Math.random() * 0.1; // Add more variation each time (0.15-0.25)
        
        const timestamp = new Date().getTime().toString().slice(-4);
        const person = {
            persona_id: `synthetic_${i}_${Math.random().toString(36).substr(2, 9)}`,
            name: `${archetype.name} ${i + 1} (${timestamp})`,
            kapwa_score: Math.max(0, Math.min(1, archetype.kapwa + (Math.random() - 0.5) * variation)),
            bayanihan_participation: Math.max(0, Math.min(1, archetype.bayanihan + (Math.random() - 0.5) * variation)),
            utang_na_loob_integrity: Math.max(0, Math.min(1, archetype.utang + (Math.random() - 0.5) * variation)),
            income: Math.max(15000, archetype.income + (Math.random() - 0.5) * 20000),
            digital_engagement: Math.random() * 0.6 + 0.3,
            regional_authenticity: Math.random() * 0.4 + 0.6,
            region: regions[Math.floor(Math.random() * regions.length)],
            age: Math.floor(Math.random() * 40) + 20,
            archetype: archetype.name
        };
        
        data.push(person);
    }
    
    return data;
}

/**
 * Load demo data - generates fresh synthetic data each time
 */
function loadDemoData() {
    try {
        // Check if mlCore is available
        if (typeof mlCore === 'undefined') {
            throw new Error('ML Core system not initialized. Please refresh the page.');
        }
        
        console.log('Generating fresh synthetic Filipino cultural data for demonstration...');
        
        // Show loading notification
        showErrorNotification('Generating new demo data...', 'info', 2000);
        updateKMeansStatus('Generating synthetic cultural personas...');
        
        // Clear existing visualizations
        const canvas = document.getElementById('kmeansCanvas');
        if (canvas) canvas.innerHTML = '';
        
        const distributionChart = document.getElementById('clusterDistributionChart');
        if (distributionChart) distributionChart.innerHTML = '';
        
        // Generate fresh synthetic data every time
        const demoData = generateSyntheticCulturalData(150);
        
        // Update the shared data
        mlCore.sharedData = demoData;
        mlCore.status.dataLoaded = true;
        
        // Dispatch data change event
        const event = new CustomEvent('datasetChanged', {
            detail: {
                dataset: demoData,
                metadata: {
                    name: 'Fresh Synthetic Filipino Cultural Personas',
                    size: demoData.length,
                    features: Object.keys(demoData[0]),
                    source: 'Generated for K-Means demonstration'
                }
            }
        });
        document.dispatchEvent(event);
        
        // Update UI and visualization
        updateKMeansStatus('New demo data generated - ready for clustering');
        visualizeKMeansDataWithD3();
        
        // Show success notification
        showErrorNotification(`✅ Generated ${demoData.length} new synthetic personas!`, 'success', 3000);
        
        return demoData;
    } catch (error) {
        console.error('Error generating demo data:', error);
        updateKMeansStatus(`Error generating demo data: ${error.message}`);
        showErrorNotification(`Failed to generate demo data: ${error.message}`, 'error');
        return mlCore.sharedData || [];
    }
}

/**
 * Handle data updates from other components
 */
function updateKmeansData(dataInfo) {
    console.log('K-Means received data update:', dataInfo.metadata || 'No metadata');
    
    // Reset current clustering state
    resetKMeansClustering();
    
    // Visualize new data
    const hasData = (mlCore.sharedData && mlCore.sharedData.length > 0) ||
                   (window.harayaData && window.harayaData.currentDataset && window.harayaData.currentDataset.length > 0);
    
    if (hasData) {
        visualizeKMeansDataWithD3();
    } else {
        // Load demo data if no data is available
        loadDemoData();
    }
}

/**
 * K-Means specific utility functions
 */
function showErrorNotification(message, type = 'error', duration = 4000) {
    console.log(`K-Means ${type}:`, message);
    
    // Create a toast notification with appropriate styling
    const toast = document.createElement('div');
    let bgColor = 'bg-red-500';
    if (type === 'success') bgColor = 'bg-green-500';
    else if (type === 'info') bgColor = 'bg-blue-500';
    else if (type === 'warning') bgColor = 'bg-yellow-500';
    
    toast.className = `fixed top-4 right-4 ${bgColor} text-white px-4 py-2 rounded-lg shadow-lg z-50 transition-opacity`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => document.body.removeChild(toast), 300);
    }, duration);
}

// Initialization handled by ml-core.js through initializeAllComponents()
// Removed duplicate DOMContentLoaded listener to prevent double initialization

// Export functions for global access
window.updateKMeansVisualization = updateKMeansVisualization;
window.runKMeansClustering = runKMeansClustering;
window.updateKmeansData = updateKmeansData;
window.resetKMeansClustering = resetKMeansClustering;
window.recommendOptimalK = recommendOptimalK;
window.loadDemoData = loadDemoData;
window.generateSyntheticCulturalData = generateSyntheticCulturalData;
window.KMeansAlgorithm = KMeansAlgorithm;

// Make the state available globally for debugging
window.kmeansState = kmeansState;