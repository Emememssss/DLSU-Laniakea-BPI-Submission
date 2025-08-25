/**
 * HARAYA ML Algorithm Explorer - Data Management Module
 * Complete implementation of dataset loading, preprocessing, and management
 */

console.log('Data Management module loaded');

// Global data state
window.harayaData = {
    currentDataset: null,
    filteredData: null,
    statistics: {},
    culturalFeatures: ['kapwa_score', 'bayanihan_participation', 'utang_na_loob_integrity'],
    rawCSVData: null,
    currentPage: 0,
    pageSize: 50
};

// Enhanced CSV parsing utility
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        try {
            // Handle complex CSV with JSON arrays and quoted strings
            const values = parseCSVLine(line, headers.length);
            const row = {};
            
            headers.forEach((header, index) => {
                let value = values[index] || '';
                
                // Remove outer quotes if present
                if (value.startsWith('"') && value.endsWith('"')) {
                    value = value.slice(1, -1);
                }
                
                // Handle JSON arrays (like temporal_activity_pattern)
                if (value.startsWith('[') && value.endsWith(']')) {
                    try {
                        row[header] = JSON.parse(value);
                    } catch (e) {
                        row[header] = value; // Keep as string if parsing fails
                    }
                }
                // Convert numeric values
                else if (!isNaN(value) && value !== '' && value !== 'True' && value !== 'False') {
                    row[header] = parseFloat(value);
                }
                // Convert boolean values
                else if (value === 'True') {
                    row[header] = true;
                } else if (value === 'False') {
                    row[header] = false;
                }
                // Keep as string
                else {
                    row[header] = value;
                }
            });
            
            data.push(row);
        } catch (error) {
            console.warn(`Error parsing CSV line ${i + 1}:`, error.message);
            // Skip problematic lines but continue processing
            continue;
        }
    }
    
    return data;
}

// Parse a single CSV line with proper handling of quoted strings and arrays
function parseCSVLine(line, expectedColumns) {
    const result = [];
    let current = '';
    let inQuotes = false;
    let inArray = false;
    let arrayDepth = 0;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"' && !inArray) {
            if (inQuotes && nextChar === '"') {
                // Escaped quote
                current += '"';
                i++; // Skip next quote
            } else {
                inQuotes = !inQuotes;
            }
        } else if (char === '[') {
            inArray = true;
            arrayDepth++;
            current += char;
        } else if (char === ']') {
            arrayDepth--;
            if (arrayDepth === 0) {
                inArray = false;
            }
            current += char;
        } else if (char === ',' && !inQuotes && !inArray) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    // Add the last field
    result.push(current.trim());
    
    // Ensure we have the expected number of columns
    while (result.length < expectedColumns) {
        result.push('');
    }
    
    return result;
}

// Calculate cultural scores from the dataset
function calculateCulturalScores(persona) {
    // Extract cultural metrics from existing data
    const kapwa = persona.community_standing_score || Math.random() * 0.5 + 0.5;
    const bayanihan = persona.location_stability_score || Math.random() * 0.5 + 0.5;
    const utang = persona.bill_payment_consistency || Math.random() * 0.5 + 0.5;
    
    return {
        kapwa_score: kapwa,
        bayanihan_participation: bayanihan,
        utang_na_loob_integrity: utang
    };
}

// Load dataset from CSV file
async function loadDatasetFromFile(filePath) {
    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`Failed to load dataset: ${response.statusText}`);
        }
        
        const csvText = await response.text();
        const data = parseCSV(csvText);
        
        // Enhance data with cultural scores
        data.forEach(persona => {
            const culturalScores = calculateCulturalScores(persona);
            Object.assign(persona, culturalScores);
        });
        
        window.harayaData.rawCSVData = csvText;
        window.harayaData.currentDataset = data;
        window.harayaData.currentPage = 0;
        
        updateDataStatistics();
        updateDataPreview();
        updateQuickAnalytics();
        
        return data;
    } catch (error) {
        console.error('Error loading dataset:', error);
        showDataError('Failed to load dataset: ' + error.message);
        return null;
    }
}

// File upload handler
function handleDatasetUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showDataError('File size too large. Maximum 10MB allowed.');
        return;
    }
    
    const allowedTypes = ['text/csv', 'application/json', '.csv', '.json'];
    const fileExtension = file.name.toLowerCase().split('.').pop();
    
    if (!allowedTypes.includes(file.type) && !['csv', 'json'].includes(fileExtension)) {
        showDataError('Only CSV and JSON files are supported.');
        return;
    }
    
    showLoadingState('Uploading and processing file...');
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            let data;
            
            if (fileExtension === 'csv') {
                data = parseCSV(e.target.result);
                window.harayaData.rawCSVData = e.target.result;
            } else if (fileExtension === 'json') {
                data = JSON.parse(e.target.result);
                if (!Array.isArray(data)) {
                    throw new Error('JSON file must contain an array of objects');
                }
            }
            
            // Validate data structure
            if (!validateDataset(data)) {
                return;
            }
            
            // Enhance with cultural scores
            data.forEach(persona => {
                const culturalScores = calculateCulturalScores(persona);
                Object.assign(persona, culturalScores);
            });
            
            window.harayaData.currentDataset = data;
            window.harayaData.currentPage = 0;
            
            updateDataStatistics();
            updateDataPreview();
            updateQuickAnalytics();
            hideLoadingState();
            
            // Dispatch datasetChanged event for ML modules
            const datasetChangedEvent = new CustomEvent('datasetChanged', {
                detail: { 
                    dataset: data,
                    source: 'file_upload',
                    filename: file.name,
                    recordCount: data.length
                }
            });
            document.dispatchEvent(datasetChangedEvent);
            
            showDataSuccess(`Successfully loaded ${data.length} records from ${file.name}`);
            
            // Log sample data for debugging
            console.log('Sample loaded data:', data.slice(0, 2));
            
        } catch (error) {
            hideLoadingState();
            showDataError('Error processing file: ' + error.message);
        }
    };
    
    reader.onerror = function() {
        hideLoadingState();
        showDataError('Error reading file');
    };
    
    reader.readAsText(file);
}

// Load preset dataset
async function loadPresetDataset(datasetName) {
    showLoadingState(`Loading ${datasetName} dataset...`);
    
    try {
        // Load the main dataset
        const fullDataset = await loadDatasetFromFile('datasets/neural_network_training_dataset.csv');
        if (!fullDataset) return;
        
        let filteredData;
        let description = '';
        
        switch (datasetName) {
            case 'filipino-personas':
                filteredData = fullDataset.filter(p => p.trustworthiness_label === 'trustworthy').slice(0, 120);
                description = 'Trustworthy Filipino cultural personas with authentic regional patterns';
                break;
                
            case 'scammer-patterns':
                filteredData = fullDataset.filter(p => 
                    p.trustworthiness_label === 'untrustworthy' || 
                    p.trustworthiness_label === 'adversarial_attack'
                ).slice(0, 60);
                description = 'Suspicious behavioral patterns and adversarial examples';
                break;
                
            case 'cultural-variations':
                // Get samples from different regions
                const luzonData = fullDataset.filter(p => p.region?.includes('luzon')).slice(0, 100);
                const visayasData = fullDataset.filter(p => p.region?.includes('visayas')).slice(0, 100);
                const mindanaoData = fullDataset.filter(p => p.region?.includes('mindanao')).slice(0, 100);
                filteredData = [...luzonData, ...visayasData, ...mindanaoData];
                description = 'Regional cultural variations across Luzon, Visayas, and Mindanao';
                break;

            case 'neural-network-training':
                filteredData = fullDataset; // Use full 2572 samples
                description = 'Large-scale dataset for Cultural Intelligence training (2052 trustworthy, 520 untrustworthy) with regional variations';
                break;
                
            case 'enhanced-cultural-intelligence':
                filteredData = fullDataset; // Use full 2572 samples
                description = 'Comprehensive Cultural Intelligence dataset (2,572 samples) with temporal patterns, advanced behavioral features, regional variations, and integrated scam detection labels';
                break;
                
            default:
                filteredData = fullDataset;
                description = 'Complete enhanced synthetic dataset';
        }
        
        window.harayaData.currentDataset = filteredData;
        window.harayaData.filteredData = filteredData;
        window.harayaData.currentPage = 0;
        
        updateDataStatistics();
        updateDataPreview();
        updateQuickAnalytics();
        hideLoadingState();
        
        // Dispatch datasetChanged event for ML modules
        const datasetChangedEvent = new CustomEvent('datasetChanged', {
            detail: { 
                dataset: filteredData,
                source: 'preset_dataset',
                datasetName: datasetName,
                recordCount: filteredData.length,
                description: description
            }
        });
        document.dispatchEvent(datasetChangedEvent);
        
        showDataSuccess(`Loaded ${filteredData.length} records: ${description}`);
        
    } catch (error) {
        hideLoadingState();
        showDataError('Failed to load preset dataset: ' + error.message);
    }
}

// Update data preview table
function updateDataPreview() {
    const previewContainer = document.getElementById('dataPreview');
    if (!previewContainer || !window.harayaData.currentDataset) return;
    
    const data = window.harayaData.currentDataset;
    const startIndex = window.harayaData.currentPage * window.harayaData.pageSize;
    const endIndex = Math.min(startIndex + window.harayaData.pageSize, data.length);
    const pageData = data.slice(startIndex, endIndex);
    
    if (pageData.length === 0) {
        previewContainer.innerHTML = '<p class="text-gray-500 text-center mt-20">No data available</p>';
        return;
    }
    
    // Get columns to display (limit for better UX)
    const allColumns = Object.keys(pageData[0]);
    const priorityColumns = [
        'region', 'age', 'monthly_income', 'digital_literacy_score', 
        'community_standing_score', 'trustworthiness_label', 'business_type', 'gender'
    ];
    const displayColumns = priorityColumns.filter(col => allColumns.includes(col));
    
    let html = `
        <div class="overflow-x-auto">
            <table class="min-w-full bg-white border border-gray-200 rounded-lg">
                <thead class="bg-gray-50">
                    <tr>
                        ${displayColumns.map(col => `
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">
                                ${col.replace(/_/g, ' ')}
                            </th>
                        `).join('')}
                    </tr>
                </thead>
                <tbody class="divide-y divide-gray-200">
                    ${pageData.map((row, index) => `
                        <tr class="hover:bg-gray-50 ${index % 2 === 0 ? 'bg-white' : 'bg-gray-25'}">
                            ${displayColumns.map(col => {
                                let value = row[col];
                                if (typeof value === 'number') {
                                    value = value.toFixed(2);
                                } else if (typeof value === 'boolean') {
                                    value = value ? 'Yes' : 'No';
                                }
                                return `<td class="px-4 py-2 text-sm text-gray-900 border-b">${value || 'N/A'}</td>`;
                            }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        
        <div class="mt-4 flex items-center justify-between">
            <div class="text-sm text-gray-600">
                Showing ${startIndex + 1}-${endIndex} of ${data.length} records
            </div>
            <div class="flex space-x-2">
                <button onclick="previousPage()" 
                        ${window.harayaData.currentPage === 0 ? 'disabled' : ''} 
                        class="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50">
                    Previous
                </button>
                <span class="px-3 py-1 text-sm bg-blue-100 text-blue-800 rounded">
                    Page ${window.harayaData.currentPage + 1} of ${Math.ceil(data.length / window.harayaData.pageSize)}
                </span>
                <button onclick="nextPage()" 
                        ${endIndex >= data.length ? 'disabled' : ''} 
                        class="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50">
                    Next
                </button>
            </div>
        </div>
    `;
    
    previewContainer.innerHTML = html;
}

// Pagination functions
function previousPage() {
    if (window.harayaData.currentPage > 0) {
        window.harayaData.currentPage--;
        updateDataPreview();
    }
}

function nextPage() {
    const data = window.harayaData.currentDataset;
    const maxPages = Math.ceil(data.length / window.harayaData.pageSize);
    if (window.harayaData.currentPage < maxPages - 1) {
        window.harayaData.currentPage++;
        updateDataPreview();
    }
}

// Update statistics
function updateDataStatistics() {
    const data = window.harayaData.currentDataset;
    if (!data) return;
    
    const stats = {
        rows: data.length,
        columns: Object.keys(data[0] || {}).length,
        size: JSON.stringify(data).length / 1024 // Size in KB
    };
    
    window.harayaData.statistics = stats;
    
    // Update UI elements
    const elements = {
        'datasetRows': stats.rows,
        'datasetColumns': stats.columns,
        'datasetSize': Math.round(stats.size)
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });
}

// Data validation
function validateDataset(data) {
    if (!Array.isArray(data) || data.length === 0) {
        showDataError('Dataset must be a non-empty array of objects');
        return false;
    }
    
    const requiredFields = ['age'];
    const firstRow = data[0];
    
    for (const field of requiredFields) {
        if (!(field in firstRow)) {
            showDataError(`Missing required field: ${field}`);
            return false;
        }
    }
    
    // Check data quality
    const validRows = data.filter(row => row.age && !isNaN(row.age));
    if (validRows.length < data.length * 0.8) {
        showDataError('Dataset appears to have significant data quality issues');
        return false;
    }
    
    return true;
}

// Export functions
function exportDataset(format) {
    const data = window.harayaData.currentDataset;
    if (!data || data.length === 0) {
        showDataError('No data to export');
        return;
    }
    
    let content, filename, mimeType;
    
    switch (format) {
        case 'json':
            content = JSON.stringify(data, null, 2);
            filename = `haraya_dataset_${Date.now()}.json`;
            mimeType = 'application/json';
            break;
            
        case 'csv':
            if (window.harayaData.rawCSVData) {
                content = window.harayaData.rawCSVData;
            } else {
                const headers = Object.keys(data[0]);
                const csvRows = [headers.join(',')];
                data.forEach(row => {
                    const values = headers.map(header => {
                        let value = row[header];
                        if (typeof value === 'string' && value.includes(',')) {
                            value = `"${value}"`;
                        }
                        return value;
                    });
                    csvRows.push(values.join(','));
                });
                content = csvRows.join('\n');
            }
            filename = `haraya_dataset_${Date.now()}.csv`;
            mimeType = 'text/csv';
            break;
            
        case 'report':
            content = generateDataReport(data);
            filename = `haraya_data_report_${Date.now()}.txt`;
            mimeType = 'text/plain';
            break;
            
        default:
            showDataError('Unsupported export format');
            return;
    }
    
    downloadFile(content, filename, mimeType);
    showDataSuccess(`Dataset exported as ${format.toUpperCase()}`);
}

// Generate data report
function generateDataReport(data) {
    const stats = window.harayaData.statistics;
    const culturalStats = analyzeCulturalFeatures(data);
    
    return `
HARAYA Cultural Intelligence Dataset Report
Generated: ${new Date().toISOString()}

DATASET OVERVIEW
================
Total Records: ${stats.rows}
Total Columns: ${stats.columns}
Dataset Size: ${Math.round(stats.size)} KB

CULTURAL FEATURES ANALYSIS
==========================
${culturalStats.map(stat => `${stat.feature}: ${stat.summary}`).join('\n')}

TRUSTWORTHINESS DISTRIBUTION
============================
${getTrustworthinessDistribution(data)}

REGIONAL BREAKDOWN
==================
${getRegionalBreakdown(data)}

SAMPLE RECORDS (First 5)
========================
${data.slice(0, 5).map((record, i) => `
Record ${i + 1}:
- Region: ${record.region || 'N/A'}
- Age: ${record.age || 'N/A'}
- Income: ${record.monthly_income || 'N/A'}
- Trust Level: ${record.trustworthiness_label || 'N/A'}
`).join('\n')}
    `.trim();
}

// Analyze cultural features
function analyzeCulturalFeatures(data) {
    return window.harayaData.culturalFeatures.map(feature => {
        const values = data.map(d => d[feature]).filter(v => v !== undefined);
        const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        return {
            feature: feature.replace(/_/g, ' '),
            summary: `Avg: ${avg.toFixed(3)}, Range: ${min.toFixed(3)}-${max.toFixed(3)}`
        };
    });
}

// Get trustworthiness distribution
function getTrustworthinessDistribution(data) {
    const distribution = {};
    data.forEach(record => {
        const label = record.trustworthiness_label || 'unknown';
        distribution[label] = (distribution[label] || 0) + 1;
    });
    
    return Object.entries(distribution)
        .map(([label, count]) => `${label}: ${count} (${(count/data.length*100).toFixed(1)}%)`)
        .join('\n');
}

// Get regional breakdown
function getRegionalBreakdown(data) {
    const regions = {};
    data.forEach(record => {
        const region = record.region || 'unknown';
        regions[region] = (regions[region] || 0) + 1;
    });
    
    return Object.entries(regions)
        .map(([region, count]) => `${region}: ${count} records`)
        .join('\n');
}

// Filter data by region
function filterDataByRegion(region) {
    const fullData = window.harayaData.currentDataset;
    if (!fullData) return;
    
    if (region === 'all') {
        window.harayaData.filteredData = fullData;
    } else {
        window.harayaData.filteredData = fullData.filter(record => 
            record.region?.includes(region.toLowerCase())
        );
    }
    
    window.harayaData.currentPage = 0;
    updateDataPreview();
    updateDataStatistics();
}

// Utility functions for UI feedback
function showDataError(message) {
    showNotification(message, 'error');
}

function showDataSuccess(message) {
    showNotification(message, 'success');
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
        type === 'error' ? 'bg-red-500 text-white' : 
        type === 'success' ? 'bg-green-500 text-white' : 
        'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

function showLoadingState(message) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.querySelector('p').textContent = message;
        overlay.classList.remove('hidden');
    }
}

function hideLoadingState() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Get data statistics for other components
function getDataStatistics() {
    return window.harayaData.statistics;
}

// Update quick analytics (called from HTML)
function updateQuickAnalytics() {
    const data = window.harayaData.currentDataset;
    if (!data || data.length === 0) {
        const elements = ['avgAge', 'avgIncome', 'trustworthyPercent', 'avgDigitalLiteracy'];
        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '--';
        });
        return;
    }
    
    // Calculate analytics
    const avgAge = Math.round(data.reduce((sum, d) => sum + (d.age || 0), 0) / data.length);
    const avgIncome = Math.round(data.reduce((sum, d) => sum + (d.monthly_income || 0), 0) / data.length);
    const trustworthyCount = data.filter(d => d.trustworthiness_label === 'trustworthy').length;
    const trustworthyPercent = Math.round((trustworthyCount / data.length) * 100);
    const avgDigitalLit = (data.reduce((sum, d) => sum + (d.digital_literacy_score || 0), 0) / data.length).toFixed(1);
    
    // Update UI
    const updates = {
        'avgAge': avgAge,
        'avgIncome': '₱' + avgIncome.toLocaleString(),
        'trustworthyPercent': trustworthyPercent + '%',
        'avgDigitalLiteracy': avgDigitalLit
    };
    
    Object.entries(updates).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });
}

// Initialize data management
document.addEventListener('DOMContentLoaded', function() {
    // Load default dataset if none is loaded
    if (!window.harayaData.currentDataset) {
        loadPresetDataset('filipino-personas');
    }
});

// Export functions for global access
window.handleDatasetUpload = handleDatasetUpload;
window.loadPresetDataset = loadPresetDataset;
window.updateDataPreview = updateDataPreview;
window.exportDataset = exportDataset;
window.getDataStatistics = getDataStatistics;
window.filterDataByRegion = filterDataByRegion;
window.validateDataset = validateDataset;
window.previousPage = previousPage;
window.nextPage = nextPage;
window.updateQuickAnalytics = updateQuickAnalytics;

console.log('Data Management module fully initialized');