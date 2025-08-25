/**
 * HARAYA Evaluation Metrics Module
 * Comprehensive model evaluation including confusion matrices, ROC curves, and performance metrics
 */

console.log('Evaluation Metrics module loaded');

// Global evaluation state
const evaluationState = {
    lastEvaluationResults: null,
    confusionMatrix: null,
    rocCurve: null,
    currentModel: null,
    benchmarkResults: {}
};

/**
 * Calculate comprehensive evaluation metrics for binary classification
 */
function calculateBinaryClassificationMetrics(yTrue, yPred, yProb = null, threshold = 0.5) {
    if (yTrue.length !== yPred.length) {
        throw new Error('Prediction and ground truth arrays must have the same length');
    }

    // Convert predictions to binary using threshold
    const binaryPred = yProb ? 
        yProb.map(prob => prob >= threshold ? 1 : 0) : 
        yPred.map(pred => pred >= threshold ? 1 : 0);

    const binaryTrue = yTrue.map(val => val >= threshold ? 1 : 0);

    // Calculate confusion matrix
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < binaryTrue.length; i++) {
        if (binaryTrue[i] === 1 && binaryPred[i] === 1) tp++;
        else if (binaryTrue[i] === 0 && binaryPred[i] === 0) tn++;
        else if (binaryTrue[i] === 0 && binaryPred[i] === 1) fp++;
        else if (binaryTrue[i] === 1 && binaryPred[i] === 0) fn++;
    }

    // Calculate metrics
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const specificity = tn / (tn + fp) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
    const f2Score = 5 * (precision * recall) / (4 * precision + recall) || 0;

    // Matthews Correlation Coefficient
    const mcc = (tp * tn - fp * fn) / Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) || 0;

    // Calculate additional metrics if probabilities are available
    let auc = null;
    let rocPoints = null;
    
    if (yProb) {
        const rocData = calculateROCCurve(binaryTrue, yProb);
        auc = rocData.auc;
        rocPoints = rocData.points;
    }

    const confusionMatrix = {
        truePositive: tp,
        trueNegative: tn,
        falsePositive: fp,
        falseNegative: fn,
        total: tp + tn + fp + fn
    };

    const metrics = {
        accuracy,
        precision,
        recall,
        specificity,
        f1Score,
        f2Score,
        mcc,
        auc,
        confusionMatrix,
        rocCurve: rocPoints,
        threshold,
        sampleSize: yTrue.length
    };

    // Store results
    evaluationState.lastEvaluationResults = metrics;
    evaluationState.confusionMatrix = confusionMatrix;
    evaluationState.rocCurve = rocPoints;

    return metrics;
}

/**
 * Calculate ROC curve and AUC
 */
function calculateROCCurve(yTrue, yProb) {
    if (yTrue.length !== yProb.length) {
        throw new Error('Labels and probabilities must have the same length');
    }

    // Create array of (probability, label) pairs and sort by probability
    const pairs = yTrue.map((label, i) => ({ prob: yProb[i], label: label }))
                       .sort((a, b) => b.prob - a.prob);

    const rocPoints = [];
    let tp = 0, fp = 0;
    const totalPositives = yTrue.reduce((sum, val) => sum + val, 0);
    const totalNegatives = yTrue.length - totalPositives;

    // Add starting point (0, 0)
    rocPoints.push({ fpr: 0, tpr: 0, threshold: 1 });

    let prevProb = 1;
    
    for (let i = 0; i < pairs.length; i++) {
        if (pairs[i].prob !== prevProb && i > 0) {
            // Add point when threshold changes
            const fpr = fp / totalNegatives;
            const tpr = tp / totalPositives;
            rocPoints.push({ fpr, tpr, threshold: prevProb });
        }

        if (pairs[i].label === 1) {
            tp++;
        } else {
            fp++;
        }
        prevProb = pairs[i].prob;
    }

    // Add final point (1, 1)
    const finalFpr = fp / totalNegatives;
    const finalTpr = tp / totalPositives;
    rocPoints.push({ fpr: finalFpr, tpr: finalTpr, threshold: 0 });

    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const dx = rocPoints[i].fpr - rocPoints[i-1].fpr;
        const avgHeight = (rocPoints[i].tpr + rocPoints[i-1].tpr) / 2;
        auc += dx * avgHeight;
    }

    return { points: rocPoints, auc };
}

/**
 * Calculate K-Means clustering evaluation metrics
 */
function calculateClusteringMetrics(data, labels, centroids) {
    if (data.length !== labels.length) {
        throw new Error('Data and labels must have the same length');
    }

    const numClusters = centroids.length;
    const numPoints = data.length;

    // Calculate Within-Cluster Sum of Squares (WCSS)
    let wcss = 0;
    for (let i = 0; i < numPoints; i++) {
        const point = data[i];
        const clusterLabel = labels[i];
        const centroid = centroids[clusterLabel];
        
        const distance = euclideanDistance(point, centroid);
        wcss += distance * distance;
    }

    // Calculate Between-Cluster Sum of Squares (BCSS)
    const globalCentroid = calculateGlobalCentroid(data);
    let bcss = 0;
    
    for (let k = 0; k < numClusters; k++) {
        const clusterSize = labels.filter(label => label === k).length;
        const centroid = centroids[k];
        const distance = euclideanDistance(centroid, globalCentroid);
        bcss += clusterSize * distance * distance;
    }

    // Calculate Total Sum of Squares (TSS)
    const tss = wcss + bcss;

    // Calculate Calinski-Harabasz Index (Variance Ratio Criterion)
    const calinskiHarabasz = numClusters === 1 ? 0 : 
        (bcss / (numClusters - 1)) / (wcss / (numPoints - numClusters));

    // Calculate Silhouette Score
    const silhouetteScore = calculateSilhouetteScore(data, labels, centroids);

    // Calculate Davies-Bouldin Index
    const daviesBouldinIndex = calculateDaviesBouldinIndex(data, labels, centroids);

    // Calculate cluster cohesion and separation
    const cohesion = wcss / numPoints; // Average intra-cluster distance
    const separation = bcss / (numClusters * (numClusters - 1) / 2); // Average inter-cluster distance

    return {
        wcss,
        bcss,
        tss,
        calinskiHarabasz,
        silhouetteScore,
        daviesBouldinIndex,
        cohesion,
        separation,
        numClusters,
        numPoints,
        clusterSizes: Array.from({length: numClusters}, (_, k) => 
            labels.filter(label => label === k).length
        )
    };
}

/**
 * Calculate Silhouette Score for clustering
 */
function calculateSilhouetteScore(data, labels, centroids) {
    const numPoints = data.length;
    const numClusters = centroids.length;
    
    if (numClusters <= 1 || numPoints <= numClusters) {
        return 0;
    }

    let totalScore = 0;
    
    for (let i = 0; i < numPoints; i++) {
        const point = data[i];
        const ownCluster = labels[i];
        
        // Calculate average distance to points in same cluster (a)
        const sameClusterPoints = [];
        for (let j = 0; j < numPoints; j++) {
            if (i !== j && labels[j] === ownCluster) {
                sameClusterPoints.push(data[j]);
            }
        }
        
        let a = 0;
        if (sameClusterPoints.length > 0) {
            a = sameClusterPoints.reduce((sum, otherPoint) => 
                sum + euclideanDistance(point, otherPoint), 0
            ) / sameClusterPoints.length;
        }
        
        // Calculate minimum average distance to points in other clusters (b)
        let b = Infinity;
        for (let k = 0; k < numClusters; k++) {
            if (k === ownCluster) continue;
            
            const otherClusterPoints = [];
            for (let j = 0; j < numPoints; j++) {
                if (labels[j] === k) {
                    otherClusterPoints.push(data[j]);
                }
            }
            
            if (otherClusterPoints.length > 0) {
                const avgDist = otherClusterPoints.reduce((sum, otherPoint) => 
                    sum + euclideanDistance(point, otherPoint), 0
                ) / otherClusterPoints.length;
                b = Math.min(b, avgDist);
            }
        }
        
        const silhouette = b === Infinity ? 0 : (b - a) / Math.max(a, b);
        totalScore += isNaN(silhouette) ? 0 : silhouette;
    }
    
    return totalScore / numPoints;
}

/**
 * Calculate Davies-Bouldin Index
 */
function calculateDaviesBouldinIndex(data, labels, centroids) {
    const numClusters = centroids.length;
    
    if (numClusters <= 1) return 0;
    
    // Calculate cluster spreads
    const clusterSpreads = [];
    for (let k = 0; k < numClusters; k++) {
        const clusterPoints = data.filter((_, i) => labels[i] === k);
        if (clusterPoints.length === 0) {
            clusterSpreads.push(0);
            continue;
        }
        
        const centroid = centroids[k];
        const spread = clusterPoints.reduce((sum, point) => 
            sum + euclideanDistance(point, centroid), 0
        ) / clusterPoints.length;
        clusterSpreads.push(spread);
    }
    
    // Calculate Davies-Bouldin Index
    let dbIndex = 0;
    for (let i = 0; i < numClusters; i++) {
        let maxRatio = 0;
        for (let j = 0; j < numClusters; j++) {
            if (i !== j) {
                const centroidDistance = euclideanDistance(centroids[i], centroids[j]);
                if (centroidDistance > 0) {
                    const ratio = (clusterSpreads[i] + clusterSpreads[j]) / centroidDistance;
                    maxRatio = Math.max(maxRatio, ratio);
                }
            }
        }
        dbIndex += maxRatio;
    }
    
    return dbIndex / numClusters;
}

/**
 * Calculate global centroid of dataset
 */
function calculateGlobalCentroid(data) {
    if (data.length === 0) return [];
    
    const dimensions = data[0].length;
    const centroid = new Array(dimensions).fill(0);
    
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < dimensions; j++) {
            centroid[j] += data[i][j];
        }
    }
    
    return centroid.map(val => val / data.length);
}

/**
 * Create confusion matrix visualization
 */
function createConfusionMatrixChart(confusionMatrix, elementId = 'confusionMatrixChart') {
    const canvas = document.getElementById(elementId);
    if (!canvas) {
        console.warn(`Canvas element ${elementId} not found`);
        return null;
    }

    const { truePositive: tp, trueNegative: tn, falsePositive: fp, falseNegative: fn } = confusionMatrix;

    // Create data for Chart.js matrix
    const data = {
        labels: ['Predicted Negative', 'Predicted Positive'],
        datasets: [{
            label: 'Actual Negative',
            data: [tn, fp],
            backgroundColor: ['#10B981', '#EF4444'],
            borderColor: '#ffffff',
            borderWidth: 2
        }, {
            label: 'Actual Positive', 
            data: [fn, tp],
            backgroundColor: ['#EF4444', '#10B981'],
            borderColor: '#ffffff',
            borderWidth: 2
        }]
    };

    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Confusion Matrix',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        afterBody: function(context) {
                            const total = tp + tn + fp + fn;
                            const value = context[0].parsed.y;
                            const percentage = ((value / total) * 100).toFixed(1);
                            return [`Count: ${value}`, `Percentage: ${percentage}%`];
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Predicted Class'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Count'
                    },
                    beginAtZero: true
                }
            }
        }
    };

    return createChart(elementId, config);
}

/**
 * Create ROC curve visualization
 */
function createROCCurveChart(rocPoints, auc, elementId = 'rocCurveChart') {
    const canvas = document.getElementById(elementId);
    if (!canvas) {
        console.warn(`Canvas element ${elementId} not found`);
        return null;
    }

    const data = {
        datasets: [{
            label: `ROC Curve (AUC = ${auc.toFixed(3)})`,
            data: rocPoints.map(point => ({ x: point.fpr, y: point.tpr })),
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1
        }, {
            label: 'Random Classifier',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#9CA3AF',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
        }]
    };

    const config = {
        type: 'scatter',
        data: data,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'ROC Curve',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    },
                    min: 0,
                    max: 1
                }
            },
            elements: {
                point: {
                    radius: 2
                }
            }
        }
    };

    return createChart(elementId, config);
}

/**
 * Create precision-recall curve
 */
function createPrecisionRecallCurve(yTrue, yProb, elementId = 'precisionRecallChart') {
    const pairs = yTrue.map((label, i) => ({ prob: yProb[i], label: label }))
                       .sort((a, b) => b.prob - a.prob);

    const prPoints = [];
    let tp = 0, fp = 0;
    const totalPositives = yTrue.reduce((sum, val) => sum + val, 0);

    for (let i = 0; i < pairs.length; i++) {
        if (pairs[i].label === 1) {
            tp++;
        } else {
            fp++;
        }

        const precision = tp / (tp + fp);
        const recall = tp / totalPositives;
        prPoints.push({ recall, precision });
    }

    const canvas = document.getElementById(elementId);
    if (!canvas) {
        console.warn(`Canvas element ${elementId} not found`);
        return null;
    }

    const data = {
        datasets: [{
            label: 'Precision-Recall Curve',
            data: prPoints.map(point => ({ x: point.recall, y: point.precision })),
            borderColor: '#10B981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1
        }]
    };

    const config = {
        type: 'scatter',
        data: data,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Precision-Recall Curve',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Recall' },
                    min: 0, max: 1
                },
                y: {
                    title: { display: true, text: 'Precision' },
                    min: 0, max: 1
                }
            },
            elements: { point: { radius: 1 } }
        }
    };

    return createChart(elementId, config);
}

/**
 * Generate comprehensive evaluation report
 */
function generateEvaluationReport(metrics) {
    const report = {
        timestamp: new Date().toISOString(),
        modelType: evaluationState.currentModel || 'Unknown',
        metrics: metrics,
        interpretation: interpretMetrics(metrics),
        recommendations: generateRecommendations(metrics)
    };

    return report;
}

/**
 * Interpret metrics and provide insights
 */
function interpretMetrics(metrics) {
    const interpretations = [];

    // Accuracy interpretation
    if (metrics.accuracy >= 0.9) {
        interpretations.push('🎯 Excellent accuracy: Model performs exceptionally well');
    } else if (metrics.accuracy >= 0.8) {
        interpretations.push('✅ Good accuracy: Model shows strong predictive performance');
    } else if (metrics.accuracy >= 0.7) {
        interpretations.push('⚠️ Moderate accuracy: Room for improvement in model performance');
    } else {
        interpretations.push('❌ Low accuracy: Significant model improvements needed');
    }

    // Precision/Recall balance
    if (metrics.precision > 0.9 && metrics.recall > 0.9) {
        interpretations.push('⚖️ Excellent precision-recall balance: Low false positives and negatives');
    } else if (metrics.precision > metrics.recall + 0.1) {
        interpretations.push('🎯 High precision model: Conservative predictions with few false positives');
    } else if (metrics.recall > metrics.precision + 0.1) {
        interpretations.push('🔍 High recall model: Captures most positives but may have false positives');
    }

    // F1 Score interpretation
    if (metrics.f1Score >= 0.8) {
        interpretations.push('🎊 Strong F1-score: Well-balanced model performance');
    } else if (metrics.f1Score >= 0.6) {
        interpretations.push('📊 Moderate F1-score: Balanced but improvable performance');
    }

    // AUC interpretation
    if (metrics.auc && metrics.auc >= 0.9) {
        interpretations.push('🏆 Excellent AUC: Outstanding discriminative ability');
    } else if (metrics.auc && metrics.auc >= 0.8) {
        interpretations.push('👍 Good AUC: Strong ability to distinguish between classes');
    } else if (metrics.auc && metrics.auc >= 0.7) {
        interpretations.push('📈 Fair AUC: Acceptable discriminative performance');
    }

    return interpretations;
}

/**
 * Generate improvement recommendations
 */
function generateRecommendations(metrics) {
    const recommendations = [];

    if (metrics.accuracy < 0.8) {
        recommendations.push('Consider feature engineering or collecting more training data');
        recommendations.push('Try different model architectures or hyperparameter tuning');
    }

    if (metrics.precision < 0.7) {
        recommendations.push('Reduce false positives by adjusting decision threshold');
        recommendations.push('Add more discriminative features to improve precision');
    }

    if (metrics.recall < 0.7) {
        recommendations.push('Increase sensitivity by lowering decision threshold');
        recommendations.push('Address class imbalance with resampling techniques');
    }

    if (metrics.auc && metrics.auc < 0.7) {
        recommendations.push('Model may be underfitting - consider more complex architectures');
        recommendations.push('Check for data quality issues or feature relevance');
    }

    return recommendations;
}

/**
 * Export evaluation results
 */
function exportEvaluationResults(format = 'json') {
    if (!evaluationState.lastEvaluationResults) {
        showErrorNotification('No evaluation results to export');
        return;
    }

    const report = generateEvaluationReport(evaluationState.lastEvaluationResults);

    if (format === 'json') {
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `haraya-evaluation-report-${Date.now()}.json`);
    } else if (format === 'csv') {
        const csvContent = convertMetricsToCSV(report.metrics);
        const blob = new Blob([csvContent], { type: 'text/csv' });
        downloadBlob(blob, `haraya-evaluation-metrics-${Date.now()}.csv`);
    }

    showSuccessNotification(`Evaluation results exported as ${format.toUpperCase()}`);
}

/**
 * Convert metrics to CSV format
 */
function convertMetricsToCSV(metrics) {
    const rows = [
        ['Metric', 'Value'],
        ['Accuracy', metrics.accuracy.toFixed(4)],
        ['Precision', metrics.precision.toFixed(4)],
        ['Recall', metrics.recall.toFixed(4)],
        ['F1-Score', metrics.f1Score.toFixed(4)],
        ['Specificity', metrics.specificity.toFixed(4)],
        ['AUC', metrics.auc ? metrics.auc.toFixed(4) : 'N/A'],
        ['MCC', metrics.mcc.toFixed(4)],
        ['True Positives', metrics.confusionMatrix.truePositive],
        ['True Negatives', metrics.confusionMatrix.trueNegative],
        ['False Positives', metrics.confusionMatrix.falsePositive],
        ['False Negatives', metrics.confusionMatrix.falseNegative],
        ['Sample Size', metrics.sampleSize],
        ['Threshold', metrics.threshold]
    ];

    return rows.map(row => row.join(',')).join('\n');
}

/**
 * Utility to download blob as file
 */
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Export functions for global access
window.calculateBinaryClassificationMetrics = calculateBinaryClassificationMetrics;
window.calculateClusteringMetrics = calculateClusteringMetrics;
window.calculateROCCurve = calculateROCCurve;
window.calculateSilhouetteScore = calculateSilhouetteScore;
window.createConfusionMatrixChart = createConfusionMatrixChart;
window.createROCCurveChart = createROCCurveChart;
window.createPrecisionRecallCurve = createPrecisionRecallCurve;
window.generateEvaluationReport = generateEvaluationReport;
window.exportEvaluationResults = exportEvaluationResults;
window.evaluationState = evaluationState;

// Register as algorithm component
document.addEventListener('DOMContentLoaded', function() {
    if (typeof registerAlgorithm === 'function') {
        registerAlgorithm('evaluation-metrics');
        console.log('Evaluation Metrics module registered successfully');
    }
});