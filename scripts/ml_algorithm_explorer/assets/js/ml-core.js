/**
 * HARAYA ML Core Functions
 * Core machine learning utilities and shared functionality
 */

// Global ML state management
const mlCore = {
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

/**
 * Initialize the ML Explorer system
 */
function initializeMLExplorer() {
    console.log('Initializing HARAYA ML Explorer Core...');
    
    try {
        // Show loading indicator
        showLoadingOverlay('Initializing ML Explorer...');
        
        // Initialize Chart.js defaults
        initializeChartDefaults();
        
        // Setup data change listeners
        setupDataChangeListeners();
        
        // Initialize TensorFlow.js
        initializeTensorFlow();
        
        // Initialize all ML components
        initializeAllComponents();
        
        // Load default dataset
        setTimeout(() => {
            loadPresetDataset('filipino-personas');
            hideLoadingOverlay();
        }, 1000);
        
        mlCore.initialized = true;
        console.log('ML Explorer Core initialized successfully');
        
        // Show success notification
        showNotification('HARAYA ML Explorer loaded successfully!', 'success');
        
    } catch (error) {
        console.error('Failed to initialize ML Explorer:', error);
        showNotification('Failed to initialize ML Explorer. Please refresh the page.', 'error');
        hideLoadingOverlay();
    }
}

/**
 * Initialize Chart.js default configurations
 */
function initializeChartDefaults() {
    if (typeof Chart !== 'undefined') {
        Chart.defaults.font.family = "'Nunito Sans', sans-serif";
        Chart.defaults.color = '#374151';
        Chart.defaults.plugins.legend.display = true;
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
    }
}

/**
 * Initialize all ML algorithm components
 */
function initializeAllComponents() {
    // Initialize components if their functions exist
    if (typeof initializeDataManagement === 'function') {
        initializeDataManagement();
    }
    
    if (typeof initializeKMeansClustering === 'function') {
        initializeKMeansClustering();
    }
    
    if (typeof initializeNeuralNetwork === 'function') {
        initializeNeuralNetwork();
    }
    
    if (typeof initializeCulturalIntelligence === 'function') {
        initializeCulturalIntelligence();
    }
    
    if (typeof initializeScamDetection === 'function') {
        initializeScamDetection();
    }
}

/**
 * Show loading overlay with message
 */
function showLoadingOverlay(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.querySelector('p').textContent = message;
        overlay.classList.remove('hidden');
    }
}

/**
 * Hide loading overlay
 */
function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info', duration = 5000) {
    // Create notification element if it doesn't exist
    let notificationContainer = document.getElementById('notificationContainer');
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notificationContainer';
        notificationContainer.className = 'fixed top-20 right-4 z-50 space-y-2';
        document.body.appendChild(notificationContainer);
    }
    
    const notification = document.createElement('div');
    notification.className = `bg-white border rounded-lg shadow-lg p-4 max-w-sm transform transition-all duration-300 translate-x-full opacity-0`;
    
    // Set color based on type
    const colors = {
        success: 'border-green-500 text-green-800',
        error: 'border-red-500 text-red-800', 
        warning: 'border-yellow-500 text-yellow-800',
        info: 'border-blue-500 text-blue-800'
    };
    
    notification.className += ` ${colors[type] || colors.info}`;
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    notification.innerHTML = `
        <div class="flex items-start space-x-3">
            <div class="text-xl">${icons[type] || icons.info}</div>
            <div class="flex-1">
                <div class="text-sm font-medium">${message}</div>
            </div>
            <button onclick="this.parentElement.parentElement.remove()" class="text-gray-400 hover:text-gray-600 ml-2">
                ✕
            </button>
        </div>
    `;
    
    notificationContainer.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full', 'opacity-0');
    }, 10);
    
    // Auto remove after duration
    setTimeout(() => {
        notification.classList.add('translate-x-full', 'opacity-0');
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

/**
 * Initialize TensorFlow.js with error handling
 */
async function initializeTensorFlow() {
    try {
        if (typeof tf !== 'undefined') {
            await tf.ready();
            console.log('TensorFlow.js initialized successfully');
            console.log('Backend:', tf.getBackend());
            
            // Set backend preferences
            try {
                await tf.setBackend('webgl');
            } catch (e) {
                console.log('WebGL not available, falling back to CPU');
                await tf.setBackend('cpu');
            }
        } else {
            console.warn('TensorFlow.js not loaded. Neural network features will be disabled.');
            showNotification('TensorFlow.js not loaded. Neural network features disabled.', 'warning');
        }
    } catch (error) {
        console.error('TensorFlow.js initialization failed:', error);
        showNotification('Neural network initialization failed. Some features may be limited.', 'warning');
    }
}

/**
 * Global error handler for ML algorithms
 */
function handleMLError(component, operation, error) {
    console.error(`${component} - ${operation} failed:`, error);
    
    const errorMessages = {
        'K-Means': 'Clustering algorithm failed. Please check your data and try again.',
        'Neural Network': 'Neural network training failed. Please verify your data and parameters.',
        'Data Management': 'Data loading failed. Please check your file format.',
        'Cultural Intelligence': 'Cultural analysis failed. Please verify your inputs.',
        'Scam Detection': 'Risk analysis failed. Please check your persona data.'
    };
    
    showNotification(errorMessages[component] || 'An error occurred. Please try again.', 'error');
    
    // Reset component state if needed
    resetComponentState(component);
}

/**
 * Reset component state after error
 */
function resetComponentState(component) {
    switch (component) {
        case 'K-Means':
            // Reset K-Means state
            if (typeof resetKMeansState === 'function') {
                resetKMeansState();
            }
            break;
        case 'Neural Network':
            // Reset neural network state
            if (typeof resetNeuralNetworkState === 'function') {
                resetNeuralNetworkState();
            }
            break;
        // Add other components as needed
    }
}

/**
 * Validate browser compatibility
 */
function validateBrowserCompatibility() {
    const issues = [];
    
    // Check for required APIs
    if (!window.fetch) issues.push('Fetch API not supported');
    if (!window.FileReader) issues.push('File API not supported');
    if (!window.requestAnimationFrame) issues.push('Animation API not supported');
    
    // Check for Chart.js
    if (typeof Chart === 'undefined') issues.push('Chart.js not loaded');
    
    // Check for D3.js
    if (typeof d3 === 'undefined') issues.push('D3.js not loaded');
    
    if (issues.length > 0) {
        console.warn('Browser compatibility issues:', issues);
        showNotification('Some features may not work properly in this browser.', 'warning');
        return false;
    }
    
    return true;
}

/**
 * Browser-compatible timing utility
 */
const getNow = () => {
    // Prefer tf.util.now() if TensorFlow.js is loaded and ready
    if (typeof tf !== 'undefined' && tf.util && typeof tf.util.now === 'function') {
        return tf.util.now();
    }
    // Fallback to window.performance.now() if available
    if (typeof window.performance !== 'undefined' && typeof window.performance.now === 'function') {
        return window.performance.now();
    }
    // Final fallback to Date.now()
    return Date.now();
};

/**
 * Performance monitoring utility
 */
const Performance = {
    timers: new Map(),
    
    start(label) {
        this.timers.set(label, getNow());
    },
    
    end(label) {
        const startTime = this.timers.get(label);
        if (startTime) {
            const duration = getNow() - startTime;
            console.log(`⏱️ ${label}: ${duration.toFixed(2)}ms`);
            this.timers.delete(label);
            return duration;
        }
    }
};

/**
 * Memory management utilities
 */
const MemoryManager = {
    cleanup() {
        // Clean up TensorFlow.js tensors
        if (typeof tf !== 'undefined') {
            const numTensors = tf.memory().numTensors;
            if (numTensors > 100) {
                console.log(`⚠️ High tensor count: ${numTensors}, running cleanup`);
                tf.disposeVariables();
            }
        }
        
        // Clean up charts
        Object.values(mlCore.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        mlCore.charts = {};
    }
};

/**
 * Setup listeners for dataset changes
 */
function setupDataChangeListeners() {
    document.addEventListener('datasetChanged', function(event) {
        console.log('Dataset changed:', event.detail);
        mlCore.sharedData = event.detail.dataset;
        mlCore.status.dataLoaded = true;
        
        // Update active algorithms count
        updateActiveAlgorithmsCount();
        
        // Trigger data update for other components
        document.dispatchEvent(new CustomEvent('dataUpdated', {
            detail: { dataset: event.detail.dataset }
        }));
    });
    
    // Listen for algorithm state changes
    document.addEventListener('algorithmStateChanged', function(event) {
        const { algorithm, state } = event.detail;
        console.log(`${algorithm} state changed:`, state);
        
        if (state === 'active') {
            mlCore.activeAlgorithms.add(algorithm);
        } else if (state === 'inactive') {
            mlCore.activeAlgorithms.delete(algorithm);
        }
        
        updateActiveAlgorithmsCount();
    });
}

/**
 * Update active algorithms counter in header
 */
function updateActiveAlgorithmsCount() {
    const countElement = document.getElementById('activeAlgorithmsCount');
    if (countElement) {
        countElement.textContent = mlCore.activeAlgorithms.size;
    }
}

/**
 * Register an algorithm as active
 */
function registerAlgorithm(name) {
    mlCore.activeAlgorithms.add(name);
    document.dispatchEvent(new CustomEvent('algorithmStateChanged', {
        detail: { algorithm: name, state: 'active' }
    }));
}

/**
 * Unregister an algorithm
 */
function unregisterAlgorithm(name) {
    mlCore.activeAlgorithms.delete(name);
    document.dispatchEvent(new CustomEvent('algorithmStateChanged', {
        detail: { algorithm: name, state: 'inactive' }
    }));
}

/**
 * Debounce utility function
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Throttle utility function
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

/**
 * Utility to format numbers for display
 */
function formatNumber(num, decimals = 2) {
    if (typeof num !== 'number') return '--';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toFixed(decimals);
}

/**
 * Utility to generate Filipino cultural colors
 */
function getCulturalColors() {
    return {
        primary: '#0057A6',     // Sea Blue
        secondary: '#FCD116',   // Flag Yellow
        success: '#2E8B57',     // Tropical Green
        accent: '#FF8C42',      // Warm Orange
        text: '#8B4513',        // Rich Brown
        background: '#FFF9E6'   // Warm White
    };
}

/**
 * Initialize when DOM is ready
 */
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeMLExplorer);
} else {
    initializeMLExplorer();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        mlCore,
        showNotification,
        handleMLError,
        Performance,
        MemoryManager,
        debounce,
        throttle,
        formatNumber,
        getCulturalColors
    };
}

/**
 * Initialize TensorFlow.js if available
 */
async function initializeTensorFlow() {
    if (typeof tf !== 'undefined') {
        try {
            console.log('TensorFlow.js version:', tf.version.tfjs);
            await tf.ready();
            console.log('TensorFlow.js initialized successfully');
            mlCore.status.modelsReady = true;
        } catch (error) {
            console.error('Failed to initialize TensorFlow.js:', error);
        }
    } else {
        console.warn('TensorFlow.js not available');
    }
}

/**
 * Notify all active algorithms of data changes
 */
function notifyAlgorithmsOfDataChange(dataInfo) {
    const algorithms = ['kmeans', 'neuralNetwork', 'culturalIntelligence', 'scamDetection'];
    
    algorithms.forEach(algorithm => {
        const updateFunction = window[`update${algorithm.charAt(0).toUpperCase() + algorithm.slice(1)}Data`];
        if (typeof updateFunction === 'function') {
            updateFunction(dataInfo);
        }
    });
}

/**
 * Register an algorithm as active
 */
function registerAlgorithm(algorithmName) {
    mlCore.activeAlgorithms.add(algorithmName);
    updateActiveAlgorithmsCount();
}

/**
 * Unregister an algorithm
 */
function unregisterAlgorithm(algorithmName) {
    mlCore.activeAlgorithms.delete(algorithmName);
    updateActiveAlgorithmsCount();
}

/**
 * Update the active algorithms counter
 */
function updateActiveAlgorithmsCount() {
    const countElement = document.getElementById('activeAlgorithmsCount');
    if (countElement) {
        countElement.textContent = mlCore.activeAlgorithms.size;
    }
}

/**
 * Utility function to create a Chart.js chart
 */
function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element ${canvasId} not found`);
        return null;
    }

    // Destroy existing chart if it exists
    if (mlCore.charts[canvasId]) {
        mlCore.charts[canvasId].destroy();
    }

    try {
        mlCore.charts[canvasId] = new Chart(canvas, config);
        return mlCore.charts[canvasId];
    } catch (error) {
        console.error(`Failed to create chart ${canvasId}:`, error);
        return null;
    }
}

/**
 * Utility function to update an existing chart
 */
function updateChart(canvasId, newData) {
    const chart = mlCore.charts[canvasId];
    if (!chart) {
        console.error(`Chart ${canvasId} not found`);
        return;
    }

    try {
        if (newData.labels) {
            chart.data.labels = newData.labels;
        }
        if (newData.datasets) {
            chart.data.datasets = newData.datasets;
        }
        chart.update();
    } catch (error) {
        console.error(`Failed to update chart ${canvasId}:`, error);
    }
}

/**
 * Utility function for statistical calculations
 */
const stats = {
    mean: (arr) => arr.reduce((sum, val) => sum + val, 0) / arr.length,
    
    median: (arr) => {
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    },
    
    standardDeviation: (arr) => {
        const mean = stats.mean(arr);
        const squaredDiffs = arr.map(val => Math.pow(val - mean, 2));
        return Math.sqrt(stats.mean(squaredDiffs));
    },
    
    correlation: (x, y) => {
        if (x.length !== y.length) return 0;
        
        const meanX = stats.mean(x);
        const meanY = stats.mean(y);
        
        let numerator = 0;
        let sumSquaredX = 0;
        let sumSquaredY = 0;
        
        for (let i = 0; i < x.length; i++) {
            const diffX = x[i] - meanX;
            const diffY = y[i] - meanY;
            
            numerator += diffX * diffY;
            sumSquaredX += diffX * diffX;
            sumSquaredY += diffY * diffY;
        }
        
        const denominator = Math.sqrt(sumSquaredX * sumSquaredY);
        return denominator === 0 ? 0 : numerator / denominator;
    },
    
    normalize: (arr) => {
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const range = max - min;
        return range === 0 ? arr.map(() => 0) : arr.map(val => (val - min) / range);
    }
};

/**
 * Utility function for distance calculations (used in clustering)
 */
function euclideanDistance(point1, point2) {
    if (point1.length !== point2.length) {
        throw new Error('Points must have the same dimensions');
    }
    
    return Math.sqrt(
        point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0)
    );
}

/**
 * Generate random color for visualizations
 */
function generateRandomColor(alpha = 1) {
    const hue = Math.floor(Math.random() * 360);
    const saturation = 60 + Math.floor(Math.random() * 40); // 60-100%
    const lightness = 45 + Math.floor(Math.random() * 20);  // 45-65%
    
    return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
}

/**
 * Generate a color palette for multiple series
 */
function generateColorPalette(count) {
    const colors = [
        '#0057A6', // Sea Blue
        '#FCD116', // Flag Yellow
        '#2E8B57', // Tropical Green
        '#FF8C42', // Warm Orange
        '#8B4513', // Rich Brown
        '#DC143C', // Crimson
        '#4169E1', // Royal Blue
        '#32CD32', // Lime Green
        '#FF1493', // Deep Pink
        '#8A2BE2'  // Blue Violet
    ];
    
    if (count <= colors.length) {
        return colors.slice(0, count);
    }
    
    // Generate additional colors if needed
    const palette = [...colors];
    for (let i = colors.length; i < count; i++) {
        palette.push(generateRandomColor());
    }
    
    return palette;
}

/**
 * Format numbers for display
 */
function formatNumber(value, decimals = 2) {
    if (typeof value !== 'number') return value;
    
    if (Math.abs(value) < 0.001 && value !== 0) {
        return value.toExponential(2);
    }
    
    return value.toFixed(decimals);
}

/**
 * Format percentage values
 */
function formatPercentage(value, decimals = 1) {
    return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Debounce function for performance optimization
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Deep clone utility for data manipulation
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj.getTime());
    if (obj instanceof Array) return obj.map(item => deepClone(item));
    
    const cloned = {};
    Object.keys(obj).forEach(key => {
        cloned[key] = deepClone(obj[key]);
    });
    return cloned;
}

/**
 * Check if dataset has required cultural features
 */
function hasCulturalFeatures(dataset) {
    if (!dataset || dataset.length === 0) return false;
    
    const culturalFeatures = ['kapwa_score', 'bayanihan_participation', 'utang_na_loob_integrity'];
    const firstRow = dataset[0];
    
    return culturalFeatures.some(feature => firstRow.hasOwnProperty(feature));
}

/**
 * Extract cultural features from dataset
 */
function extractCulturalFeatures(dataset) {
    const culturalColumns = [
        'kapwa_score', 
        'bayanihan_participation', 
        'utang_na_loob_integrity', 
        'cultural_authenticity',
        'community_standing_score'
    ];
    
    return dataset.map(row => {
        const features = {};
        culturalColumns.forEach(col => {
            if (row.hasOwnProperty(col) && typeof row[col] === 'number') {
                features[col] = row[col];
            }
        });
        return features;
    }).filter(features => Object.keys(features).length > 0);
}

/**
 * Calculate cultural authenticity score
 */
function calculateCulturalAuthenticity(persona) {
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

/**
 * Performance monitoring utilities
 */
const performanceMonitor = {
    start: (label) => {
        console.time(label);
    },
    
    end: (label) => {
        console.timeEnd(label);
    },
    
    measure: (func, label) => {
        performanceMonitor.start(label);
        const result = func();
        performanceMonitor.end(label);
        return result;
    },
    
    measureAsync: async (func, label) => {
        performanceMonitor.start(label);
        const result = await func();
        performanceMonitor.end(label);
        return result;
    }
};

/**
 * Error handling utilities
 */
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    
    // Show user-friendly error message
    const message = error.message || 'An unexpected error occurred';
    showErrorNotification(`${context ? context + ': ' : ''}${message}`);
}

/**
 * Validation utilities
 */
const validators = {
    isNumber: (value) => typeof value === 'number' && !isNaN(value),
    
    isInRange: (value, min, max) => validators.isNumber(value) && value >= min && value <= max,
    
    hasRequiredFields: (obj, fields) => fields.every(field => obj.hasOwnProperty(field)),
    
    isValidDataset: (dataset) => {
        return Array.isArray(dataset) && 
               dataset.length > 0 && 
               typeof dataset[0] === 'object';
    }
};

// Export core utilities for global access
window.mlCore = mlCore;
window.stats = stats;
window.createChart = createChart;
window.updateChart = updateChart;
window.euclideanDistance = euclideanDistance;
window.generateColorPalette = generateColorPalette;
window.formatNumber = formatNumber;
window.formatPercentage = formatPercentage;
window.debounce = debounce;
window.deepClone = deepClone;
window.hasCulturalFeatures = hasCulturalFeatures;
window.extractCulturalFeatures = extractCulturalFeatures;
window.calculateCulturalAuthenticity = calculateCulturalAuthenticity;
window.handleError = handleError;
window.validators = validators;
window.registerAlgorithm = registerAlgorithm;
window.unregisterAlgorithm = unregisterAlgorithm;

// Enhanced notification system for better UX
function showSuccessNotification(message, duration = 4000) {
    return showToast(message, 'success', duration);
}

function showErrorNotification(message, duration = 6000) {
    return showToast(message, 'error', duration);
}

function showWarningNotification(message, duration = 5000) {
    return showToast(message, 'warning', duration);
}

function showInfoNotification(message, duration = 4000) {
    return showToast(message, 'info', duration);
}

/**
 * Enhanced toast notification system
 */
function showToast(message, type = 'info', duration = 4000) {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        console.error('Toast container not found in HTML');
        return;
    }

    const toastId = 'toast_' + Math.random().toString(36).substr(2, 9);
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };

    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast ${type} p-4 max-w-sm`;
    toast.innerHTML = `
        <div class="flex items-start">
            <div class="flex-shrink-0 mr-3 text-lg">
                ${icons[type] || icons.info}
            </div>
            <div class="flex-1">
                <p class="text-sm font-medium text-gray-900">${message}</p>
            </div>
            <button onclick="removeToast('${toastId}')" class="flex-shrink-0 ml-4 text-gray-400 hover:text-gray-600">
                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                </svg>
            </button>
        </div>
        <div class="toast-progress"></div>
    `;

    toastContainer.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    // Auto-remove toast
    setTimeout(() => {
        removeToast(toastId);
    }, duration);

    return toastId;
}

/**
 * Remove a specific toast notification
 */
function removeToast(toastId) {
    const toast = document.getElementById(toastId);
    if (toast) {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }
}

/**
 * Enhanced progress bar utilities
 */
function updateProgressBar(elementId, percentage, message = '') {
    const progressElement = document.getElementById(elementId);
    if (progressElement) {
        progressElement.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
        
        // Update message if provided
        const messageElement = document.getElementById(elementId + 'Message');
        if (messageElement && message) {
            messageElement.textContent = message;
        }
    }
}

function resetProgressBar(elementId) {
    updateProgressBar(elementId, 0);
}

// Export notification functions globally
window.showSuccessNotification = showSuccessNotification;
window.showErrorNotification = showErrorNotification;
window.showWarningNotification = showWarningNotification;
window.showInfoNotification = showInfoNotification;
window.showToast = showToast;
window.removeToast = removeToast;
window.updateProgressBar = updateProgressBar;
window.resetProgressBar = resetProgressBar;