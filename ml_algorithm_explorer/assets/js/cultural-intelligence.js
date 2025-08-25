/**
 * HARAYA Cultural Intelligence Algorithm Explorer
 * Interactive cultural weight adjustment and analysis
 */

// Cultural intelligence state
const culturalIntelligenceState = {
    weights: {
        kapwa: 30,
        bayanihan: 25,
        utang: 25,
        traditional: 20
    },
    regionalContext: 'luzon_metro',
    currentAnalysis: null,
    regionalProfiles: {
        'luzon_metro': { kapwa: 0.65, bayanihan: 0.60, utang: 0.75, traditional: 0.4 },
        'luzon_rural': { kapwa: 0.80, bayanihan: 0.85, utang: 0.85, traditional: 0.8 },
        'visayas_urban': { kapwa: 0.75, bayanihan: 0.80, utang: 0.80, traditional: 0.6 },
        'visayas_rural': { kapwa: 0.85, bayanihan: 0.90, utang: 0.88, traditional: 0.9 },
        'mindanao_urban': { kapwa: 0.70, bayanihan: 0.75, utang: 0.78, traditional: 0.5 },
        'mindanao_rural': { kapwa: 0.88, bayanihan: 0.92, utang: 0.90, traditional: 0.95 }
    }
};

/**
 * Initialize Cultural Intelligence component
 */
function initializeCulturalIntelligence() {
    console.log('Initializing Cultural Intelligence...');
    registerAlgorithm('culturalIntelligence');
    setupCulturalIntelligenceControls();
    createInitialCulturalChart();
    updateCulturalWeights();
}

/**
 * Setup Cultural Intelligence control event listeners
 */
function setupCulturalIntelligenceControls() {
    // Weight sliders
    const sliders = ['kapwa', 'bayanihan', 'utang', 'traditional'];
    sliders.forEach(slider => {
        const element = document.getElementById(`${slider}Slider`);
        if (element) {
            element.addEventListener('input', debounce(updateCulturalWeights, 300));
        }
    });

    // Regional context selector
    const regionalSelect = document.getElementById('regionalContext');
    if (regionalSelect) {
        regionalSelect.addEventListener('change', updateCulturalWeights);
    }
}

/**
 * Calculate adaptive cultural weights based on persona and regional context
 */
function calculateAdaptiveWeights(persona, regionalContext) {
    let weights = { kapwa: 30, bayanihan: 25, utang: 25, traditional: 20 };
    
    // Regional adjustments (real calculations)
    if (regionalContext === 'luzon_rural') {
        weights.bayanihan += 5; // Stronger community emphasis
        weights.utang += 3;
        weights.traditional += 2;
    } else if (regionalContext === 'mindanao_rural') {
        weights.bayanihan += 8; // Highest community emphasis
        weights.utang += 5;
        weights.traditional += 7;
    } else if (regionalContext.includes('metro')) {
        weights.kapwa += 5; // Urban individualism but with shared identity
        weights.traditional -= 5;
    } else if (regionalContext.includes('visayas')) {
        weights.kapwa += 3; // Family-centric balance
        weights.bayanihan += 2;
    }
    
    // Age-based adjustments
    if (persona && persona.age) {
        if (persona.age > 50) {
            weights.utang += 5; // Older = more traditional values
            weights.traditional += 3;
        } else if (persona.age < 25) {
            weights.kapwa += 3; // Younger = more identity-focused
            weights.traditional -= 2;
        }
    }
    
    // Education level adjustments
    if (persona && persona.education_level) {
        if (persona.education_level >= 4) { // College+
            weights.kapwa += 2;
            weights.traditional -= 1;
        }
    }
    
    // Normalize to 100%
    const total = weights.kapwa + weights.bayanihan + weights.utang + weights.traditional;
    if (total !== 100) {
        const factor = 100 / total;
        Object.keys(weights).forEach(key => {
            weights[key] = Math.round(weights[key] * factor);
        });
    }
    
    return weights;
}

/**
 * Update cultural weights and analysis with adaptive calculation
 */
function updateCulturalWeights() {
    // Update weights from sliders
    const sliders = ['kapwa', 'bayanihan', 'utang', 'traditional'];
    let totalWeight = 0;
    let useAdaptiveWeights = false;

    // Check if we should use adaptive weights (when data is loaded)
    if (mlCore.sharedData && mlCore.sharedData.length > 0) {
        const regionalSelect = document.getElementById('regionalContext');
        const regionalContext = regionalSelect ? regionalSelect.value : 'luzon_metro';
        
        // Use first persona as reference for adaptive weights
        const referencePersona = mlCore.sharedData[0];
        const adaptiveWeights = calculateAdaptiveWeights(referencePersona, regionalContext);
        
        // Apply adaptive weights if user hasn't manually adjusted sliders significantly
        const currentWeights = culturalIntelligenceState.weights;
        const defaultWeights = { kapwa: 30, bayanihan: 25, utang: 25, traditional: 20 };
        
        // Check if weights are still close to default (user hasn't customized much)
        const isNearDefault = Object.keys(defaultWeights).every(key => 
            Math.abs(currentWeights[key] - defaultWeights[key]) <= 5
        );
        
        if (isNearDefault) {
            culturalIntelligenceState.weights = adaptiveWeights;
            useAdaptiveWeights = true;
        }
    }

    if (!useAdaptiveWeights) {
        // Manual weight adjustment from sliders
        sliders.forEach(slider => {
            const element = document.getElementById(`${slider}Slider`);
            
            if (element) {
                const value = parseInt(element.value);
                culturalIntelligenceState.weights[slider] = value;
                totalWeight += value;
            }
        });

        // Normalize weights to 100%
        if (totalWeight !== 100 && totalWeight > 0) {
            const factor = 100 / totalWeight;
            sliders.forEach(slider => {
                culturalIntelligenceState.weights[slider] = Math.round(culturalIntelligenceState.weights[slider] * factor);
            });
        }
    }

    // Update displays and sliders to reflect current weights
    sliders.forEach(slider => {
        const display = document.getElementById(`${slider}Weight`);
        const sliderElement = document.getElementById(`${slider}Slider`);
        
        if (display) {
            display.textContent = `${culturalIntelligenceState.weights[slider]}%`;
        }
        if (sliderElement) {
            sliderElement.value = culturalIntelligenceState.weights[slider];
        }
    });

    // Update regional context
    const regionalSelect = document.getElementById('regionalContext');
    if (regionalSelect) {
        culturalIntelligenceState.regionalContext = regionalSelect.value;
    }

    // Perform cultural analysis
    performCulturalAnalysis();
    updateCulturalVisualization();
}

/**
 * Perform cultural analysis based on current settings
 */
function performCulturalAnalysis() {
    if (!mlCore.sharedData || mlCore.sharedData.length === 0) {
        return;
    }

    const analysis = {
        totalPersonas: mlCore.sharedData.length,
        authenticity: calculateAuthenticityScores(),
        regional: calculateRegionalAlignment(),
        community: calculateCommunityIntegration(),
        insights: generateCulturalInsights()
    };

    culturalIntelligenceState.currentAnalysis = analysis;
    updateCulturalScoreDisplays(analysis);
}

/**
 * Calculate authenticity scores for current dataset
 */
function calculateAuthenticityScores() {
    if (!mlCore.sharedData) return { average: 0, distribution: [] };

    const scores = mlCore.sharedData.map(persona => {
        return calculatePersonaAuthenticity(persona);
    });

    return {
        average: stats.mean(scores),
        median: stats.median(scores),
        standardDeviation: stats.standardDeviation(scores),
        distribution: scores
    };
}

/**
 * Calculate authenticity for individual persona with enhanced cultural metrics
 */
function calculatePersonaAuthenticity(persona) {
    const weights = culturalIntelligenceState.weights;
    let score = 0;
    let totalWeight = 0;

    // Kapwa component - enhanced calculation
    let kapwaScore = 0;
    if (persona.kapwa_score !== undefined) {
        kapwaScore = persona.kapwa_score;
    } else {
        // Derive kapwa from community connections and family size
        let derivedKapwa = 0;
        if (persona.community_standing_score !== undefined) {
            derivedKapwa += persona.community_standing_score * 0.4;
        }
        if (persona.family_size !== undefined) {
            // Larger families often indicate stronger kapwa connections
            derivedKapwa += Math.min(persona.family_size / 8, 1.0) * 0.3;
        }
        if (persona.location_stability_score !== undefined) {
            derivedKapwa += persona.location_stability_score * 0.3;
        }
        kapwaScore = Math.min(derivedKapwa, 1.0);
    }
    
    if (kapwaScore > 0) {
        score += kapwaScore * (weights.kapwa / 100);
        totalWeight += weights.kapwa / 100;
    }

    // Bayanihan component - enhanced calculation
    let bayanihanScore = 0;
    if (persona.bayanihan_participation !== undefined) {
        bayanihanScore = persona.bayanihan_participation;
    } else {
        // Derive bayanihan from community engagement indicators
        let derivedBayanihan = 0;
        if (persona.community_standing_score !== undefined) {
            derivedBayanihan += persona.community_standing_score * 0.5;
        }
        if (persona.bill_payment_consistency !== undefined) {
            // Consistent bill payment shows community responsibility
            derivedBayanihan += persona.bill_payment_consistency * 0.3;
        }
        if (persona.region && persona.region.includes('rural')) {
            derivedBayanihan += 0.2; // Rural areas typically have stronger bayanihan
        }
        bayanihanScore = Math.min(derivedBayanihan, 1.0);
    }
    
    if (bayanihanScore > 0) {
        score += bayanihanScore * (weights.bayanihan / 100);
        totalWeight += weights.bayanihan / 100;
    }

    // Utang na Loob component - enhanced calculation
    let utangScore = 0;
    if (persona.utang_na_loob_integrity !== undefined) {
        utangScore = persona.utang_na_loob_integrity;
    } else {
        // Derive utang na loob from financial behavior and relationships
        let derivedUtang = 0;
        if (persona.bill_payment_consistency !== undefined) {
            derivedUtang += persona.bill_payment_consistency * 0.4;
        }
        if (persona.income_variability !== undefined) {
            // Lower income variability indicates more stable, relationship-based income
            derivedUtang += (1 - persona.income_variability) * 0.3;
        }
        if (persona.community_standing_score !== undefined) {
            derivedUtang += persona.community_standing_score * 0.3;
        }
        utangScore = Math.min(derivedUtang, 1.0);
    }
    
    if (utangScore > 0) {
        score += utangScore * (weights.utang / 100);
        totalWeight += weights.utang / 100;
    }

    // Traditional factors - enhanced calculation
    let traditionalScore = 0;
    if (persona.community_standing_score !== undefined) {
        traditionalScore = persona.community_standing_score * 0.3;
    }
    if (persona.location_stability_score !== undefined) {
        traditionalScore += persona.location_stability_score * 0.25;
    }
    if (persona.family_size !== undefined) {
        // Larger families traditionally indicate stronger cultural ties
        traditionalScore += Math.min((persona.family_size - 2) / 6, 0.3);
    }
    if (persona.age !== undefined && persona.age > 35) {
        // Older individuals typically have stronger traditional values
        traditionalScore += Math.min((persona.age - 35) / 30, 0.15);
    }
    
    traditionalScore = Math.min(traditionalScore, 1.0);
    if (traditionalScore > 0) {
        score += traditionalScore * (weights.traditional / 100);
        totalWeight += weights.traditional / 100;
    }

    return totalWeight > 0 ? score / totalWeight : 0;
}

/**
 * Calculate regional alignment scores
 */
function calculateRegionalAlignment() {
    if (!mlCore.sharedData) return { average: 0 };

    const regionalProfile = culturalIntelligenceState.regionalProfiles[culturalIntelligenceState.regionalContext];
    if (!regionalProfile) return { average: 0 };

    const alignmentScores = mlCore.sharedData.map(persona => {
        let alignment = 0;
        let factors = 0;

        if (persona.kapwa_score !== undefined) {
            alignment += 1 - Math.abs(persona.kapwa_score - regionalProfile.kapwa);
            factors++;
        }

        if (persona.bayanihan_participation !== undefined) {
            alignment += 1 - Math.abs(persona.bayanihan_participation - regionalProfile.bayanihan);
            factors++;
        }

        if (persona.utang_na_loob_integrity !== undefined) {
            alignment += 1 - Math.abs(persona.utang_na_loob_integrity - regionalProfile.utang);
            factors++;
        }

        return factors > 0 ? alignment / factors : 0;
    });

    return {
        average: stats.mean(alignmentScores),
        distribution: alignmentScores
    };
}

/**
 * Calculate community integration scores
 */
function calculateCommunityIntegration() {
    if (!mlCore.sharedData) return { average: 0 };

    const integrationScores = mlCore.sharedData.map(persona => {
        let integration = 0;
        let factors = 0;

        // Community standing
        if (persona.community_standing_score !== undefined) {
            integration += persona.community_standing_score * 0.4;
            factors += 0.4;
        }

        // Bayanihan participation
        if (persona.bayanihan_participation !== undefined) {
            integration += persona.bayanihan_participation * 0.3;
            factors += 0.3;
        }

        // Kapwa network strength
        if (persona.kapwa_score !== undefined) {
            integration += persona.kapwa_score * 0.3;
            factors += 0.3;
        }

        return factors > 0 ? integration / factors : 0;
    });

    return {
        average: stats.mean(integrationScores),
        distribution: integrationScores
    };
}

/**
 * Generate comprehensive cultural insights with detailed analysis
 */
function generateCulturalInsights() {
    if (!mlCore.sharedData || !culturalIntelligenceState.currentAnalysis) {
        return [];
    }

    const insights = [];
    const analysis = culturalIntelligenceState.currentAnalysis;
    const weights = culturalIntelligenceState.weights;

    // Weight distribution insights with cultural context
    const maxWeight = Math.max(weights.kapwa, weights.bayanihan, weights.utang, weights.traditional);
    const maxWeightKey = Object.keys(weights).find(key => weights[key] === maxWeight);
    
    switch (maxWeightKey) {
        case 'kapwa':
            insights.push(`Kapwa-centric model (${weights.kapwa}%) emphasizes shared identity and interconnectedness - ideal for community-based lending`);
            break;
        case 'bayanihan':
            insights.push(`Bayanihan-focused model (${weights.bayanihan}%) prioritizes collective action and mutual aid - strong for group lending programs`);
            break;
        case 'utang':
            insights.push(`Utang na Loob weighted model (${weights.utang}%) values reciprocal obligations - excellent for relationship-based credit assessment`);
            break;
        case 'traditional':
            insights.push(`Traditional values model (${weights.traditional}%) emphasizes stability and cultural preservation - conservative but reliable`);
            break;
    }

    // Authenticity insights with detailed breakdown
    if (analysis.authenticity) {
        const authScore = analysis.authenticity.average;
        const stdDev = analysis.authenticity.standardDeviation;
        
        if (authScore > 0.85) {
            insights.push(`Exceptional cultural authenticity (${Math.round(authScore * 100)}%) - personas show deep Filipino cultural integration`);
        } else if (authScore > 0.7) {
            insights.push(`Strong cultural authenticity (${Math.round(authScore * 100)}%) with ${stdDev > 0.15 ? 'high variance - mixed traditional/modern profiles' : 'consistent patterns'}`);
        } else if (authScore > 0.5) {
            insights.push(`Moderate authenticity (${Math.round(authScore * 100)}%) suggests transitional or urbanized Filipino cultural patterns`);
        } else {
            insights.push(`Lower authenticity (${Math.round(authScore * 100)}%) may indicate synthetic profiles or non-traditional demographics`);
        }
    }

    // Regional insights with specific cultural patterns
    const region = culturalIntelligenceState.regionalContext;
    const regionalProfile = culturalIntelligenceState.regionalProfiles[region];
    
    if (region === 'luzon_metro') {
        insights.push('Metro Manila context: Expects individualized patterns with digital integration, lower traditional scores acceptable');
    } else if (region === 'luzon_rural') {
        insights.push('Luzon rural context: Strong bayanihan and utang na loob expected, family-centered economic patterns');
    } else if (region === 'visayas_urban') {
        insights.push('Visayas urban: Balanced family-community dynamics, moderate digital adoption with strong cultural retention');
    } else if (region === 'visayas_rural') {
        insights.push('Visayas rural: Highest community integration expected, traditional extended family economic structures');
    } else if (region === 'mindanao_urban') {
        insights.push('Mindanao urban: Diverse cultural influences, strong community bonds with modern economic participation');
    } else if (region === 'mindanao_rural') {
        insights.push('Mindanao rural: Peak traditional values, tight-knit communities, relationship-based economic systems');
    }

    // Community integration insights
    if (analysis.community && analysis.regional) {
        const communityScore = analysis.community.average;
        const regionalScore = analysis.regional.average;
        
        if (communityScore > 0.8 && regionalScore > 0.8) {
            insights.push('Strong community-regional alignment - personas fit expected cultural patterns perfectly');
        } else if (Math.abs(communityScore - regionalScore) > 0.2) {
            insights.push('Community-regional mismatch detected - may indicate migration or cultural transition');
        }
    }

    // Dataset-specific insights
    if (mlCore.sharedData.length > 100) {
        const legitimateCount = mlCore.sharedData.filter(p => p.is_legitimate === 'True' || p.is_legitimate === true).length;
        const scammerCount = mlCore.sharedData.filter(p => p.is_scammer === 'True' || p.is_scammer === true).length;
        
        if (legitimateCount > 0 && scammerCount > 0) {
            insights.push(`Dataset contains ${legitimateCount} legitimate and ${scammerCount} fraudulent profiles - suitable for comparative cultural analysis`);
        }
    }

    return insights;
}

/**
 * Update cultural score displays
 */
function updateCulturalScoreDisplays(analysis) {
    // Authenticity score
    const authenticityElement = document.getElementById('authenticityScore');
    const authenticityProgress = document.getElementById('authenticityProgress');
    if (authenticityElement && analysis.authenticity) {
        const percentage = Math.round(analysis.authenticity.average * 100);
        authenticityElement.textContent = `${percentage}%`;
        if (authenticityProgress) {
            authenticityProgress.style.width = `${percentage}%`;
        }
    }

    // Regional alignment score
    const regionalElement = document.getElementById('regionalScore');
    const regionalProgress = document.getElementById('regionalProgress');
    if (regionalElement && analysis.regional) {
        const percentage = Math.round(analysis.regional.average * 100);
        regionalElement.textContent = `${percentage}%`;
        if (regionalProgress) {
            regionalProgress.style.width = `${percentage}%`;
        }
    }

    // Community integration score
    const communityElement = document.getElementById('communityScore');
    const communityProgress = document.getElementById('communityProgress');
    if (communityElement && analysis.community) {
        const percentage = Math.round(analysis.community.average * 100);
        communityElement.textContent = `${percentage}%`;
        if (communityProgress) {
            communityProgress.style.width = `${percentage}%`;
        }
    }
}

/**
 * Create initial cultural analysis chart
 */
function createInitialCulturalChart() {
    const config = {
        type: 'radar',
        data: {
            labels: ['Kapwa\n(Shared Identity)', 'Bayanihan\n(Community Spirit)', 'Utang na Loob\n(Gratitude)', 'Traditional\nFactors'],
            datasets: [
                {
                    label: 'Current Weights',
                    data: [30, 25, 25, 20],
                    borderColor: '#0057A6',
                    backgroundColor: 'rgba(0, 87, 166, 0.2)',
                    borderWidth: 2
                },
                {
                    label: 'Regional Baseline',
                    data: [65, 60, 75, 40],
                    borderColor: '#FCD116',
                    backgroundColor: 'rgba(252, 209, 22, 0.2)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            }
        }
    };

    createChart('culturalAnalysisChart', config);
}

/**
 * Update cultural visualization
 */
function updateCulturalVisualization() {
    const weights = culturalIntelligenceState.weights;
    const regionalProfile = culturalIntelligenceState.regionalProfiles[culturalIntelligenceState.regionalContext];
    
    if (!regionalProfile) return;

    updateChart('culturalAnalysisChart', {
        labels: ['Kapwa\n(Shared Identity)', 'Bayanihan\n(Community Spirit)', 'Utang na Loob\n(Gratitude)', 'Traditional\nFactors'],
        datasets: [
            {
                label: 'Current Weights',
                data: [weights.kapwa, weights.bayanihan, weights.utang, weights.traditional],
                borderColor: '#0057A6',
                backgroundColor: 'rgba(0, 87, 166, 0.2)',
                borderWidth: 2
            },
            {
                label: 'Regional Baseline',
                data: [
                    regionalProfile.kapwa * 100,
                    regionalProfile.bayanihan * 100,
                    regionalProfile.utang * 100,
                    regionalProfile.traditional * 100
                ],
                borderColor: '#FCD116',
                backgroundColor: 'rgba(252, 209, 22, 0.2)',
                borderWidth: 2
            }
        ]
    });
}

/**
 * Apply cultural intelligence scoring to a persona
 */
function applyCulturalIntelligenceScoring(persona) {
    const authenticity = calculatePersonaAuthenticity(persona);
    const regionalProfile = culturalIntelligenceState.regionalProfiles[culturalIntelligenceState.regionalContext];
    
    let regionalAlignment = 0;
    if (regionalProfile) {
        let alignment = 0;
        let factors = 0;

        if (persona.kapwa_score !== undefined) {
            alignment += 1 - Math.abs(persona.kapwa_score - regionalProfile.kapwa);
            factors++;
        }

        if (persona.bayanihan_participation !== undefined) {
            alignment += 1 - Math.abs(persona.bayanihan_participation - regionalProfile.bayanihan);
            factors++;
        }

        if (persona.utang_na_loob_integrity !== undefined) {
            alignment += 1 - Math.abs(persona.utang_na_loob_integrity - regionalProfile.utang);
            factors++;
        }

        regionalAlignment = factors > 0 ? alignment / factors : 0;
    }

    return {
        authenticity,
        regionalAlignment,
        overallScore: (authenticity * 0.7) + (regionalAlignment * 0.3),
        weights: culturalIntelligenceState.weights,
        regionalContext: culturalIntelligenceState.regionalContext
    };
}

/**
 * Handle data updates from other components
 */
function updateCulturalIntelligenceData(dataInfo) {
    console.log('Cultural Intelligence received data update:', dataInfo.metadata);
    performCulturalAnalysis();
    updateCulturalVisualization();
}

/**
 * Export cultural analysis results
 */
function exportCulturalAnalysis() {
    if (!culturalIntelligenceState.currentAnalysis) {
        showErrorNotification('No analysis to export. Please load data first.');
        return;
    }

    const exportData = {
        timestamp: new Date().toISOString(),
        weights: culturalIntelligenceState.weights,
        regionalContext: culturalIntelligenceState.regionalContext,
        analysis: culturalIntelligenceState.currentAnalysis,
        insights: culturalIntelligenceState.currentAnalysis.insights
    };

    const jsonData = JSON.stringify(exportData, null, 2);
    const filename = `haraya_cultural_analysis_${Date.now()}.json`;
    
    downloadFile(jsonData, filename, 'application/json');
    showSuccessNotification('Cultural analysis exported successfully');
}

/**
 * Get cultural scoring for external use
 */
function getCulturalScoringSystem() {
    return {
        weights: culturalIntelligenceState.weights,
        regionalContext: culturalIntelligenceState.regionalContext,
        regionalProfiles: culturalIntelligenceState.regionalProfiles,
        scorePersona: applyCulturalIntelligenceScoring,
        currentAnalysis: culturalIntelligenceState.currentAnalysis
    };
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeCulturalIntelligence();
});

// Export functions for global access
window.updateCulturalWeights = updateCulturalWeights;
window.updateCulturalIntelligenceData = updateCulturalIntelligenceData;
window.applyCulturalIntelligenceScoring = applyCulturalIntelligenceScoring;
window.exportCulturalAnalysis = exportCulturalAnalysis;
window.getCulturalScoringSystem = getCulturalScoringSystem;