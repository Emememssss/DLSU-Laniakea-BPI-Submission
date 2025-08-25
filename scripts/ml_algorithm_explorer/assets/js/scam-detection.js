/**
 * HARAYA Scam Detection Algorithm
 * Interactive scam risk analysis and behavioral pattern recognition
 */

// Scam detection state
const scamDetectionState = {
    currentAnalysis: null,
    testPersona: {
        monthly_income: 25000,
        community_standing_score: 0.85,
        digital_activity_level: 6,
        transaction_velocity: 15,
        regional_context: 'metro_manila'
    },
    riskThresholds: {
        low: 0.3,
        medium: 0.7,
        high: 1.0
    },
    behaviorPatterns: {
        legitimate: {
            income_consistency: 0.8,
            community_validation: 0.9,
            digital_behavior_normal: 0.85,
            transaction_pattern_regular: 0.8
        },
        suspicious: {
            income_consistency: 0.3,
            community_validation: 0.2,
            digital_behavior_normal: 0.4,
            transaction_pattern_regular: 0.3
        }
    },
    // Add chart state management
    chartInitialized: false,
    lastChartUpdate: 0,
    updateThrottleMs: 500
};

/**
 * Initialize Scam Detection component
 */
function initializeScamDetection() {
    console.log('Initializing Scam Detection...');
    registerAlgorithm('scamDetection');
    setupScamDetectionControls();
    createInitialBehaviorChart();
    analyzeScamRisk();
}

/**
 * Setup Scam Detection control event listeners
 */
function setupScamDetectionControls() {
    // Input controls
    const controls = [
        'testIncome', 'testCommunityStanding', 'testDigitalActivity', 
        'testTransactionVelocity', 'testRegion'
    ];

    controls.forEach(controlId => {
        const element = document.getElementById(controlId);
        if (element) {
            // Use different event handling for range vs select inputs
            if (element.type === 'range') {
                // Separate display updates from analysis updates
                const displayId = controlId.replace('test', '').toLowerCase() + 'Value';
                const displayElement = document.getElementById(displayId);
                
                // Immediate display update (no analysis)
                element.addEventListener('input', function() {
                    if (displayElement) {
                        displayElement.textContent = this.value;
                    }
                });
                
                // Throttled analysis update only on value change
                element.addEventListener('change', debounce(analyzeScamRisk, 500));
                // Also update on mouseup to catch drag end
                element.addEventListener('mouseup', debounce(analyzeScamRisk, 300));
            } else {
                // For select and text inputs, use change event
                element.addEventListener('change', debounce(analyzeScamRisk, 300));
            }
        }
    });
}

/**
 * Analyze scam risk for current test persona
 */
function analyzeScamRisk() {
    // Update test persona from UI controls
    updateTestPersonaFromControls();

    // Perform risk analysis
    const riskAnalysis = performRiskAnalysis(scamDetectionState.testPersona);
    scamDetectionState.currentAnalysis = riskAnalysis;

    // Update UI displays
    updateRiskAnalysisDisplay(riskAnalysis);
    updateBehaviorPatternChart(riskAnalysis);
    updateRiskIndicators(riskAnalysis);
}

/**
 * Update test persona from UI controls
 */
function updateTestPersonaFromControls() {
    const incomeInput = document.getElementById('testIncome');
    const communityInput = document.getElementById('testCommunityStanding');
    const digitalInput = document.getElementById('testDigitalActivity');
    const transactionInput = document.getElementById('testTransactionVelocity');
    const regionInput = document.getElementById('testRegion');

    if (incomeInput) scamDetectionState.testPersona.monthly_income = parseInt(incomeInput.value);
    if (communityInput) scamDetectionState.testPersona.community_standing_score = parseFloat(communityInput.value);
    if (digitalInput) scamDetectionState.testPersona.digital_activity_level = parseInt(digitalInput.value);
    if (transactionInput) scamDetectionState.testPersona.transaction_velocity = parseInt(transactionInput.value);
    if (regionInput) scamDetectionState.testPersona.regional_context = regionInput.value;
}

/**
 * Calculate comprehensive scam risk with advanced behavioral pattern analysis
 */
function calculateScamRisk(persona) {
    let riskScore = 0;
    
    // Income vs lifestyle consistency
    if (persona.monthly_income > 50000 && persona.community_standing_score < 0.3) {
        riskScore += 0.4; // High income, low community standing = suspicious
    }
    
    // Income vs digital behavior mismatch
    if (persona.monthly_income < 15000 && persona.digital_literacy_score > 8) {
        riskScore += 0.3; // Low income but high digital sophistication
    }
    
    // Cultural authenticity patterns
    if (typeof applyCulturalIntelligenceScoring === 'function') {
        try {
            const culturalScore = applyCulturalIntelligenceScoring(persona);
            if (culturalScore.authenticity < 0.5) {
                riskScore += 0.3; // Low cultural authenticity
            }
        } catch (error) {
            console.warn('Cultural scoring failed:', error);
        }
    }
    
    // Digital behavior inconsistencies
    if (persona.digital_literacy_score > 8 && persona.banking_app_sessions_per_week < 2) {
        riskScore += 0.2; // High digital skills but low banking usage
    }
    
    // Transaction velocity vs income mismatch
    if (persona.transaction_velocity && persona.monthly_income) {
        const expectedVelocity = Math.min(persona.monthly_income / 2000, 30);
        if (Math.abs(persona.transaction_velocity - expectedVelocity) > 20) {
            riskScore += 0.25;
        }
    }
    
    // Perfect scores anomaly (too good to be true)
    let perfectScores = 0;
    if (persona.community_standing_score >= 0.95) perfectScores++;
    if (persona.bill_payment_consistency >= 0.98) perfectScores++;
    if (persona.location_stability_score >= 0.95) perfectScores++;
    
    if (perfectScores >= 2) {
        riskScore += 0.2; // Multiple perfect scores are suspicious
    }
    
    // Regional income inconsistencies
    const regionalIncomeRanges = {
        'luzon_metro': { min: 15000, max: 80000 },
        'luzon_rural': { min: 8000, max: 25000 },
        'visayas_urban': { min: 12000, max: 35000 },
        'visayas_rural': { min: 8000, max: 20000 },
        'mindanao_urban': { min: 10000, max: 30000 },
        'mindanao_rural': { min: 6000, max: 18000 }
    };
    
    const expectedRange = regionalIncomeRanges[persona.regional_context] || regionalIncomeRanges['luzon_metro'];
    if (persona.monthly_income > expectedRange.max * 2.5) {
        riskScore += 0.35; // Extremely high income for region
    } else if (persona.monthly_income < expectedRange.min * 0.5) {
        riskScore += 0.15; // Unusually low income
    }
    
    return Math.min(riskScore, 1.0);
}

/**
 * Perform comprehensive risk analysis with enhanced behavioral patterns
 */
function performRiskAnalysis(persona) {
    const analysis = {
        overall_risk_score: 0,
        risk_level: 'LOW_RISK',
        risk_factors: [],
        cultural_authenticity: 'High',
        transaction_pattern: 'Normal',
        community_validation: 'Strong',
        behavioral_anomalies: [],
        explanation: '',
        detailed_scores: {},
        fraud_indicators: []
    };

    let riskScore = 0;
    const factors = [];
    const detailedScores = {};

    // 1. Enhanced Income Analysis
    const incomeRisk = analyzeIncomePatterns(persona);
    detailedScores.income_risk = incomeRisk.risk;
    riskScore += incomeRisk.risk * 0.2;
    if (incomeRisk.risk > 0.5) {
        factors.push(incomeRisk.explanation);
        analysis.behavioral_anomalies.push('income_inconsistency');
        analysis.fraud_indicators.push({
            type: 'income_anomaly',
            severity: incomeRisk.risk > 0.7 ? 'high' : 'medium',
            description: incomeRisk.explanation
        });
    }

    // 2. Enhanced Community Standing Analysis
    const communityRisk = analyzeCommunityStanding(persona);
    detailedScores.community_risk = communityRisk.risk;
    riskScore += communityRisk.risk * 0.3;
    if (communityRisk.risk > 0.5) {
        factors.push(communityRisk.explanation);
        analysis.behavioral_anomalies.push('weak_community_ties');
        analysis.fraud_indicators.push({
            type: 'community_isolation',
            severity: communityRisk.risk > 0.8 ? 'high' : 'medium',
            description: communityRisk.explanation
        });
    }
    analysis.community_validation = communityRisk.level;

    // 3. Enhanced Digital Behavior Analysis
    const digitalRisk = analyzeDigitalBehavior(persona);
    detailedScores.digital_risk = digitalRisk.risk;
    riskScore += digitalRisk.risk * 0.25;
    if (digitalRisk.risk > 0.5) {
        factors.push(digitalRisk.explanation);
        analysis.behavioral_anomalies.push('unusual_digital_patterns');
        analysis.fraud_indicators.push({
            type: 'digital_anomaly',
            severity: digitalRisk.risk > 0.6 ? 'high' : 'medium',
            description: digitalRisk.explanation
        });
    }

    // 4. Enhanced Transaction Velocity Analysis
    const transactionRisk = analyzeTransactionVelocity(persona);
    detailedScores.transaction_risk = transactionRisk.risk;
    riskScore += transactionRisk.risk * 0.25;
    if (transactionRisk.risk > 0.5) {
        factors.push(transactionRisk.explanation);
        analysis.behavioral_anomalies.push('suspicious_transaction_velocity');
        analysis.fraud_indicators.push({
            type: 'transaction_anomaly',
            severity: transactionRisk.risk > 0.7 ? 'high' : 'medium',
            description: transactionRisk.explanation
        });
    }
    analysis.transaction_pattern = transactionRisk.pattern;

    // 5. Enhanced Cultural Authenticity Analysis
    const culturalRisk = analyzeCulturalAuthenticity(persona);
    detailedScores.cultural_risk = culturalRisk.risk;
    // Cultural authenticity is a multiplicative factor
    if (culturalRisk.risk > 0.3) {
        riskScore *= (1 + culturalRisk.risk * 0.4);
    }
    analysis.cultural_authenticity = culturalRisk.level;
    if (culturalRisk.risk > 0.6) {
        factors.push(culturalRisk.explanation);
        analysis.behavioral_anomalies.push('cultural_inauthenticity');
        analysis.fraud_indicators.push({
            type: 'cultural_mismatch',
            severity: culturalRisk.risk > 0.8 ? 'high' : 'medium',
            description: culturalRisk.explanation
        });
    }

    // 6. Cross-factor Consistency Analysis
    const consistencyRisk = analyzeDataConsistency(persona);
    detailedScores.consistency_risk = consistencyRisk.risk;
    riskScore += consistencyRisk.risk * 0.15;
    if (consistencyRisk.risk > 0.5) {
        factors.push(consistencyRisk.explanation);
        analysis.fraud_indicators.push({
            type: 'data_inconsistency',
            severity: 'high',
            description: consistencyRisk.explanation
        });
    }

    // 7. Synthetic Profile Detection
    const syntheticRisk = analyzeSyntheticProfile(persona);
    detailedScores.synthetic_risk = syntheticRisk.risk;
    if (syntheticRisk.risk > 0.6) {
        riskScore *= 1.3; // High penalty for synthetic profiles
        analysis.fraud_indicators.push({
            type: 'synthetic_profile',
            severity: 'high',
            description: syntheticRisk.explanation
        });
    }

    // Normalize risk score
    analysis.overall_risk_score = Math.min(riskScore, 1.0);
    analysis.risk_factors = factors;
    analysis.detailed_scores = detailedScores;

    // Enhanced risk level determination
    if (analysis.overall_risk_score < scamDetectionState.riskThresholds.low) {
        analysis.risk_level = 'LOW_RISK';
        analysis.explanation = `Profile demonstrates authentic Filipino cultural patterns (${Math.round((1-culturalRisk.risk)*100)}% authenticity) with consistent behavioral indicators. Strong community ties and normal financial patterns suggest legitimate user.`;
    } else if (analysis.overall_risk_score < scamDetectionState.riskThresholds.medium) {
        analysis.risk_level = 'MEDIUM_RISK';
        analysis.explanation = `Several irregular patterns detected requiring verification. ${analysis.fraud_indicators.length} potential red flags identified. Recommend additional documentation and community validation.`;
    } else {
        analysis.risk_level = 'HIGH_RISK';
        analysis.explanation = `Multiple suspicious indicators suggest high probability of fraudulent activity. ${analysis.fraud_indicators.filter(f => f.severity === 'high').length} high-severity anomalies detected. Recommend rejection or extensive verification.`;
    }

    return analysis;
}

/**
 * Analyze income patterns for inconsistencies
 */
function analyzeIncomePatterns(persona) {
    let risk = 0;
    let explanation = '';

    const income = persona.monthly_income;
    const region = persona.regional_context;

    // Regional income expectations
    const regionalIncomeRanges = {
        'metro_manila': { min: 15000, max: 80000, average: 35000 },
        'luzon_rural': { min: 8000, max: 25000, average: 15000 },
        'visayas_urban': { min: 12000, max: 35000, average: 20000 },
        'mindanao_rural': { min: 6000, max: 20000, average: 12000 }
    };

    const expected = regionalIncomeRanges[region] || regionalIncomeRanges['metro_manila'];
    
    if (income < expected.min) {
        risk += 0.3;
        explanation = 'Declared income unusually low for region';
    } else if (income > expected.max * 2) {
        risk += 0.6;
        explanation = 'Declared income suspiciously high for region';
    } else if (income === 0 || income % 1000 === 0) {
        // Round numbers can be suspicious
        risk += 0.2;
        explanation = 'Income appears fabricated (round number)';
    }

    return { risk: Math.min(risk, 1.0), explanation };
}

/**
 * Analyze community standing indicators
 */
function analyzeCommunityStanding(persona) {
    const standing = persona.community_standing_score;
    let risk = 1 - standing; // Inverse relationship
    let level = 'Strong';
    let explanation = '';

    if (standing < 0.3) {
        level = 'Weak';
        explanation = 'Very low community standing indicates possible isolation or false identity';
    } else if (standing < 0.6) {
        level = 'Moderate';
        explanation = 'Below-average community ties may indicate recent arrival or social issues';
    } else if (standing > 0.95) {
        // Suspiciously perfect scores can also be fabricated
        risk += 0.1;
        explanation = 'Perfect community score may be fabricated';
    }

    return { risk: Math.min(risk, 1.0), level, explanation };
}

/**
 * Analyze digital behavior patterns
 */
function analyzeDigitalBehavior(persona) {
    const digitalActivity = persona.digital_activity_level;
    let risk = 0;
    let explanation = '';

    // Age-adjusted expectations (assuming age correlation)
    const expectedActivity = persona.age ? Math.max(2, 10 - (persona.age - 25) * 0.1) : 6;

    if (digitalActivity > 9) {
        risk += 0.4;
        explanation = 'Extremely high digital activity may indicate bot-like behavior';
    } else if (digitalActivity < 2) {
        risk += 0.3;
        explanation = 'Unusually low digital activity for microfinance applicant';
    } else if (Math.abs(digitalActivity - expectedActivity) > 3) {
        risk += 0.2;
        explanation = 'Digital activity doesn\'t match expected demographic profile';
    }

    return { risk: Math.min(risk, 1.0), explanation };
}

/**
 * Analyze transaction velocity patterns
 */
function analyzeTransactionVelocity(persona) {
    const velocity = persona.transaction_velocity;
    let risk = 0;
    let pattern = 'Normal';
    let explanation = '';

    if (velocity > 50) {
        risk += 0.7;
        pattern = 'Suspicious';
        explanation = 'Extremely high transaction velocity indicates possible fraud';
    } else if (velocity > 30) {
        risk += 0.4;
        pattern = 'High';
        explanation = 'High transaction frequency requires verification';
    } else if (velocity < 3) {
        risk += 0.2;
        pattern = 'Low';
        explanation = 'Unusually low transaction activity';
    } else if (velocity >= 10 && velocity <= 25) {
        pattern = 'Normal';
    }

    return { risk: Math.min(risk, 1.0), pattern, explanation };
}

/**
 * Analyze cultural authenticity using enhanced pattern recognition
 */
function analyzeCulturalAuthenticity(persona) {
    let risk = 0;
    let level = 'High';
    let explanation = '';

    // If we have cultural intelligence scoring available
    if (typeof applyCulturalIntelligenceScoring === 'function') {
        try {
            const culturalScore = applyCulturalIntelligenceScoring(persona);
            const authenticity = culturalScore.authenticity || 0.5;
            const regionalAlignment = culturalScore.regionalAlignment || 0.5;
            
            // Combine authenticity and regional alignment
            const combinedScore = (authenticity * 0.7) + (regionalAlignment * 0.3);
            risk = 1 - combinedScore;
            
            if (combinedScore < 0.3) {
                level = 'Very Low';
                explanation = 'Severe cultural pattern mismatch - likely synthetic or stolen identity';
            } else if (combinedScore < 0.5) {
                level = 'Low';
                explanation = 'Cultural patterns inconsistent with authentic Filipino personas';
            } else if (combinedScore < 0.7) {
                level = 'Moderate';
                explanation = 'Some cultural inconsistencies detected - may be legitimate but non-traditional';
            } else if (combinedScore < 0.9) {
                level = 'High';
                explanation = 'Strong cultural authenticity with minor variations';
            } else {
                level = 'Very High';
                explanation = 'Exceptional cultural authenticity - strong Filipino identity';
            }
        } catch (error) {
            console.warn('Cultural authenticity analysis failed:', error);
            // Fallback to basic analysis
            risk = 0.5;
            level = 'Unknown';
            explanation = 'Cultural analysis unavailable';
        }
    } else {
        // Enhanced fallback analysis using available demographic data
        let culturalIndicators = [];
        let totalWeight = 0;
        
        // Community standing as cultural indicator
        if (persona.community_standing_score !== undefined) {
            culturalIndicators.push(persona.community_standing_score * 0.3);
            totalWeight += 0.3;
        }
        
        // Location stability indicates cultural roots
        if (persona.location_stability_score !== undefined) {
            culturalIndicators.push(persona.location_stability_score * 0.2);
            totalWeight += 0.2;
        }
        
        // Family size indicates traditional family structures
        if (persona.family_size !== undefined) {
            const familyScore = Math.min(persona.family_size / 6, 1.0); // Larger families are more traditional
            culturalIndicators.push(familyScore * 0.15);
            totalWeight += 0.15;
        }
        
        // Age-based cultural expectations
        if (persona.age !== undefined) {
            const ageScore = persona.age > 35 ? 0.8 : 0.6; // Older = more traditional
            culturalIndicators.push(ageScore * 0.1);
            totalWeight += 0.1;
        }
        
        // Regional income consistency
        if (persona.monthly_income && persona.regional_context) {
            const regionalExpected = {
                'luzon_metro': 30000, 'luzon_rural': 15000,
                'visayas_urban': 22000, 'visayas_rural': 12000,
                'mindanao_urban': 20000, 'mindanao_rural': 10000
            };
            
            const expected = regionalExpected[persona.regional_context] || 25000;
            const ratio = persona.monthly_income / expected;
            const consistencyScore = ratio > 3 ? 0.3 : (ratio < 0.3 ? 0.4 : 0.8);
            culturalIndicators.push(consistencyScore * 0.25);
            totalWeight += 0.25;
        }
        
        const avgCultural = totalWeight > 0 ? 
            culturalIndicators.reduce((sum, score) => sum + score, 0) / totalWeight : 0.5;
        
        risk = 1 - avgCultural;
        
        if (avgCultural < 0.3) {
            level = 'Low';
            explanation = 'Multiple demographic inconsistencies suggest inauthentic profile';
        } else if (avgCultural < 0.6) {
            level = 'Moderate';
            explanation = 'Some demographic patterns don\'t align with typical Filipino profiles';
        } else {
            level = 'High';
            explanation = 'Demographic patterns consistent with authentic Filipino identity';
        }
    }

    return { risk: Math.min(risk, 1.0), level, explanation };
}

/**
 * Analyze data consistency across multiple fields
 */
function analyzeDataConsistency(persona) {
    let risk = 0;
    let explanation = '';
    const inconsistencies = [];
    
    // Income vs Digital Literacy consistency
    if (persona.monthly_income && persona.digital_literacy_score) {
        const expectedDigitalScore = Math.min(Math.max(persona.monthly_income / 8000, 3), 9);
        const digitalGap = Math.abs(persona.digital_literacy_score - expectedDigitalScore);
        if (digitalGap > 3) {
            risk += 0.3;
            inconsistencies.push('digital literacy doesn\'t match income level');
        }
    }
    
    // Community Standing vs Location Stability
    if (persona.community_standing_score && persona.location_stability_score) {
        if (Math.abs(persona.community_standing_score - persona.location_stability_score) > 0.4) {
            risk += 0.2;
            inconsistencies.push('community standing inconsistent with location stability');
        }
    }
    
    // Transaction patterns vs Income
    if (persona.transaction_velocity && persona.monthly_income) {
        const expectedVelocity = Math.min(persona.monthly_income / 1500, 40);
        if (persona.transaction_velocity > expectedVelocity * 2) {
            risk += 0.4;
            inconsistencies.push('transaction velocity too high for reported income');
        }
    }
    
    // Banking behavior consistency
    if (persona.banking_app_sessions_per_week && persona.digital_literacy_score) {
        if (persona.digital_literacy_score > 7 && persona.banking_app_sessions_per_week < 2) {
            risk += 0.25;
            inconsistencies.push('high digital literacy but low banking app usage');
        } else if (persona.digital_literacy_score < 4 && persona.banking_app_sessions_per_week > 6) {
            risk += 0.3;
            inconsistencies.push('low digital literacy but high banking app usage');
        }
    }
    
    // Mobile money vs Banking usage consistency
    if (persona.mobile_money_transactions_per_month && persona.banking_app_sessions_per_week) {
        const mobileMoneyHigh = persona.mobile_money_transactions_per_month > 30;
        const bankingHigh = persona.banking_app_sessions_per_week > 4;
        
        if (mobileMoneyHigh && !bankingHigh) {
            risk += 0.15; // High mobile money but low banking is suspicious
            inconsistencies.push('high mobile money usage but low banking engagement');
        }
    }
    
    if (inconsistencies.length > 0) {
        explanation = `Data inconsistencies detected: ${inconsistencies.join(', ')}`;
    }
    
    return { risk: Math.min(risk, 1.0), explanation, inconsistencies };
}

/**
 * Analyze synthetic profile indicators
 */
function analyzeSyntheticProfile(persona) {
    let risk = 0;
    let explanation = '';
    const syntheticIndicators = [];
    
    // Perfect or round numbers (common in synthetic data)
    let perfectScores = 0;
    const checkFields = [
        'community_standing_score', 'location_stability_score', 
        'bill_payment_consistency', 'income_variability'
    ];
    
    checkFields.forEach(field => {
        if (persona[field] !== undefined) {
            if (persona[field] === 1.0 || persona[field] === 0.0) {
                perfectScores++;
            }
            // Check for suspiciously round decimal values
            if (persona[field].toString().match(/\.[05]0*$/)) {
                risk += 0.1;
            }
        }
    });
    
    if (perfectScores >= 2) {
        risk += 0.4;
        syntheticIndicators.push(`${perfectScores} perfect scores`);
    }
    
    // Income round numbers
    if (persona.monthly_income && persona.monthly_income % 5000 === 0) {
        risk += 0.2;
        syntheticIndicators.push('suspiciously round income');
    }
    
    // Age patterns (common synthetic ages)
    if (persona.age && [25, 30, 35, 40, 45].includes(persona.age)) {
        risk += 0.1;
        syntheticIndicators.push('common synthetic age');
    }
    
    // Family size patterns
    if (persona.family_size && persona.family_size > 10) {
        risk += 0.3;
        syntheticIndicators.push('unrealistic family size');
    }
    
    // Digital activity vs age mismatch
    if (persona.age && persona.digital_literacy_score) {
        if (persona.age > 60 && persona.digital_literacy_score > 8) {
            risk += 0.2;
            syntheticIndicators.push('unrealistic digital literacy for age');
        }
    }
    
    // Temporal pattern analysis (if available)
    if (persona.temporal_activity_pattern && typeof persona.temporal_activity_pattern === 'string') {
        try {
            const pattern = JSON.parse(persona.temporal_activity_pattern);
            if (pattern.length === 30) {
                // Check for unnatural patterns (too uniform or too random)
                const variance = calculateVariance(pattern);
                if (variance < 0.01 || variance > 0.3) {
                    risk += 0.25;
                    syntheticIndicators.push('unnatural temporal activity pattern');
                }
            }
        } catch (e) {
            // Invalid pattern format is suspicious
            risk += 0.2;
            syntheticIndicators.push('malformed temporal pattern');
        }
    }
    
    if (syntheticIndicators.length > 0) {
        explanation = `Synthetic profile indicators: ${syntheticIndicators.join(', ')}`;
    }
    
    return { risk: Math.min(risk, 1.0), explanation, indicators: syntheticIndicators };
}

/**
 * Calculate variance for temporal pattern analysis
 */
function calculateVariance(array) {
    const mean = array.reduce((sum, val) => sum + val, 0) / array.length;
    const variance = array.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / array.length;
    return variance;
}

/**
 * Update risk analysis display with enhanced information
 */
function updateRiskAnalysisDisplay(analysis) {
    // Risk level badge with enhanced styling
    const riskLevelElement = document.getElementById('riskLevel');
    if (riskLevelElement) {
        riskLevelElement.textContent = analysis.risk_level.replace('_', ' ');
        riskLevelElement.className = `px-4 py-2 rounded-full text-white font-semibold text-sm ${getRiskLevelColor(analysis.risk_level)}`;
    }

    // Risk explanation with enhanced detail
    const explanationElement = document.getElementById('riskExplanation');
    if (explanationElement) {
        explanationElement.innerHTML = `
            <p class="text-gray-700 mb-3">${analysis.explanation}</p>
            ${analysis.fraud_indicators && analysis.fraud_indicators.length > 0 ? `
                <div class="mt-4">
                    <h5 class="font-semibold text-gray-800 mb-2">Key Risk Factors:</h5>
                    <ul class="space-y-1">
                        ${analysis.fraud_indicators.slice(0, 3).map(indicator => `
                            <li class="flex items-start text-sm">
                                <span class="mr-2 ${
                                    indicator.severity === 'high' ? 'text-red-500' : 'text-yellow-500'
                                }">●</span>
                                <span>${indicator.description}</span>
                            </li>
                        `).join('')}
                        ${analysis.fraud_indicators.length > 3 ? `
                            <li class="text-sm text-gray-500 italic">
                                ...and ${analysis.fraud_indicators.length - 3} more indicators
                            </li>
                        ` : ''}
                    </ul>
                </div>
            ` : ''}
        `;
    }
    
    // Update recommendation section
    const recommendationElement = document.getElementById('riskRecommendation');
    if (recommendationElement) {
        let recommendation = '';
        let actionClass = '';
        
        if (analysis.risk_level === 'LOW_RISK') {
            recommendation = '✓ APPROVE - Profile shows authentic patterns with minimal risk indicators.';
            actionClass = 'text-green-600 bg-green-50 border-green-200';
        } else if (analysis.risk_level === 'MEDIUM_RISK') {
            recommendation = '⚠ VERIFY - Request additional documentation and community validation before approval.';
            actionClass = 'text-yellow-700 bg-yellow-50 border-yellow-200';
        } else {
            recommendation = '⛔ REJECT - High probability of fraudulent activity. Do not approve without extensive verification.';
            actionClass = 'text-red-600 bg-red-50 border-red-200';
        }
        
        recommendationElement.innerHTML = `
            <div class="p-3 rounded-lg border-2 ${actionClass}">
                <div class="font-semibold">${recommendation}</div>
            </div>
        `;
    }
}

/**
 * Get color class for risk level with enhanced styling
 */
function getRiskLevelColor(riskLevel) {
    switch (riskLevel) {
        case 'LOW_RISK': return 'bg-green-500 shadow-green-200';
        case 'MEDIUM_RISK': return 'bg-yellow-500 shadow-yellow-200';
        case 'HIGH_RISK': return 'bg-red-500 shadow-red-200';
        default: return 'bg-gray-500 shadow-gray-200';
    }
}

/**
 * Update behavior pattern chart with enhanced detailed analysis (optimized)
 */
function updateBehaviorPatternChart(analysis) {
    // Throttle updates to prevent excessive chart recreation
    const now = Date.now();
    if (now - scamDetectionState.lastChartUpdate < scamDetectionState.updateThrottleMs) {
        return; // Skip update if too frequent
    }
    scamDetectionState.lastChartUpdate = now;

    const labels = [
        'Income\nConsistency', 
        'Community\nValidation', 
        'Digital\nBehavior', 
        'Transaction\nPattern',
        'Cultural\nAuthenticity',
        'Data\nConsistency'
    ];
    
    // Calculate behavior scores based on detailed analysis
    const detailedScores = analysis.detailed_scores || {};
    
    const incomeScore = Math.max(0, 1 - (detailedScores.income_risk || 0));
    const communityScore = Math.max(0, 1 - (detailedScores.community_risk || 0));
    const digitalScore = Math.max(0, 1 - (detailedScores.digital_risk || 0));
    const transactionScore = Math.max(0, 1 - (detailedScores.transaction_risk || 0));
    const culturalScore = Math.max(0, 1 - (detailedScores.cultural_risk || 0));
    const consistencyScore = Math.max(0, 1 - (detailedScores.consistency_risk || 0));

    const profileData = [
        incomeScore * 100, 
        communityScore * 100, 
        digitalScore * 100, 
        transactionScore * 100,
        culturalScore * 100,
        consistencyScore * 100
    ];

    // Determine color scheme based on overall risk
    let profileColor, profileBgColor;
    if (analysis.risk_level === 'LOW_RISK') {
        profileColor = '#10B981';
        profileBgColor = 'rgba(16, 185, 129, 0.3)';
    } else if (analysis.risk_level === 'MEDIUM_RISK') {
        profileColor = '#F59E0B';
        profileBgColor = 'rgba(245, 158, 11, 0.3)';
    } else {
        profileColor = '#EF4444';
        profileBgColor = 'rgba(239, 68, 68, 0.3)';
    }

    // Try to update existing chart first
    const existingChart = mlCore && mlCore.charts && mlCore.charts['behaviorPatternChart'];
    if (existingChart && scamDetectionState.chartInitialized) {
        try {
            // Update existing chart data instead of recreating
            existingChart.data.datasets[0].data = profileData;
            existingChart.data.datasets[0].borderColor = profileColor;
            existingChart.data.datasets[0].backgroundColor = profileBgColor;
            existingChart.data.datasets[0].pointBackgroundColor = profileColor;
            
            // Use smooth animation for updates
            existingChart.update('active');
            return;
        } catch (error) {
            console.warn('Chart update failed, recreating:', error);
            // If update fails, recreate the chart
        }
    }

    // Create new chart only if update failed or chart doesn't exist
    const config = {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Current Profile',
                    data: profileData,
                    borderColor: profileColor,
                    backgroundColor: profileBgColor,
                    borderWidth: 3,
                    pointBackgroundColor: profileColor,
                    pointBorderColor: '#fff',
                    pointRadius: 5
                },
                {
                    label: 'Legitimate Baseline',
                    data: [85, 88, 82, 85, 90, 88],
                    borderColor: '#6B7280',
                    backgroundColor: 'rgba(107, 114, 128, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 3
                },
                {
                    label: 'Scammer Pattern',
                    data: [35, 25, 40, 20, 30, 25],
                    borderColor: '#DC2626',
                    backgroundColor: 'rgba(220, 38, 38, 0.1)',
                    borderWidth: 1,
                    borderDash: [2, 4],
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750, // Reduced animation time
                easing: 'easeOutQuart'
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    min: 0,
                    ticks: {
                        stepSize: 20,
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    angleLines: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 11
                        },
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${Math.round(context.raw)}%`;
                        },
                        afterLabel: function(context) {
                            // Add risk interpretation
                            const value = context.raw;
                            if (value > 80) return '✓ Low Risk';
                            if (value > 60) return '⚠ Medium Risk';
                            if (value > 40) return '⚠ High Risk';
                            return '⛔ Critical Risk';
                        }
                    }
                }
            },
            interaction: {
                intersect: false
            }
        }
    };

    createChart('behaviorPatternChart', config);
    scamDetectionState.chartInitialized = true;
}

/**
 * Update risk indicators with enhanced details
 */
function updateRiskIndicators(analysis) {
    // Overall risk score progress bar
    const riskScoreElement = document.getElementById('overallRiskScore');
    const riskProgressElement = document.getElementById('riskProgress');
    
    if (riskScoreElement && analysis.overall_risk_score !== undefined) {
        const percentage = Math.round(analysis.overall_risk_score * 100);
        riskScoreElement.textContent = `${percentage}%`;
        
        if (riskProgressElement) {
            riskProgressElement.style.width = `${percentage}%`;
            // Update progress bar color based on risk level
            riskProgressElement.className = `h-full rounded transition-all duration-300 ${
                analysis.risk_level === 'LOW_RISK' ? 'bg-green-500' :
                analysis.risk_level === 'MEDIUM_RISK' ? 'bg-yellow-500' : 'bg-red-500'
            }`;
        }
    }
    
    // Cultural authenticity with enhanced levels
    const authenticityElement = document.getElementById('scamAuthenticity');
    if (authenticityElement) {
        authenticityElement.textContent = analysis.cultural_authenticity;
        authenticityElement.className = `font-bold ${
            analysis.cultural_authenticity === 'Very High' || analysis.cultural_authenticity === 'High' ? 'text-green-600' :
            analysis.cultural_authenticity === 'Moderate' ? 'text-yellow-600' : 'text-red-600'
        }`;
    }

    // Transaction pattern with more detail
    const transactionElement = document.getElementById('scamTransactionPattern');
    if (transactionElement) {
        transactionElement.textContent = analysis.transaction_pattern;
        transactionElement.className = `font-bold ${
            analysis.transaction_pattern === 'Normal' ? 'text-green-600' :
            analysis.transaction_pattern === 'High' ? 'text-yellow-600' : 'text-red-600'
        }`;
    }

    // Community validation
    const communityElement = document.getElementById('scamCommunityValidation');
    if (communityElement) {
        communityElement.textContent = analysis.community_validation;
        communityElement.className = `font-bold ${
            analysis.community_validation === 'Strong' ? 'text-green-600' :
            analysis.community_validation === 'Moderate' ? 'text-yellow-600' : 'text-red-600'
        }`;
    }
    
    // Fraud indicators count
    const fraudIndicatorsElement = document.getElementById('fraudIndicatorsCount');
    if (fraudIndicatorsElement && analysis.fraud_indicators) {
        const highSeverityCount = analysis.fraud_indicators.filter(f => f.severity === 'high').length;
        const totalCount = analysis.fraud_indicators.length;
        
        fraudIndicatorsElement.textContent = `${totalCount} (${highSeverityCount} high severity)`;
        fraudIndicatorsElement.className = `font-bold ${
            totalCount === 0 ? 'text-green-600' :
            highSeverityCount === 0 ? 'text-yellow-600' : 'text-red-600'
        }`;
    }
    
    // Detailed risk breakdown
    const riskBreakdownElement = document.getElementById('riskBreakdown');
    if (riskBreakdownElement && analysis.detailed_scores) {
        const scores = analysis.detailed_scores;
        const breakdown = [
            `Income: ${Math.round((1-scores.income_risk)*100)}%`,
            `Community: ${Math.round((1-scores.community_risk)*100)}%`,
            `Digital: ${Math.round((1-scores.digital_risk)*100)}%`,
            `Cultural: ${Math.round((1-scores.cultural_risk)*100)}%`,
            `Consistency: ${Math.round((1-scores.consistency_risk)*100)}%`
        ];
        
        riskBreakdownElement.innerHTML = breakdown.map(item => 
            `<span class="inline-block bg-gray-100 px-2 py-1 rounded text-xs mr-2 mb-1">${item}</span>`
        ).join('');
    }
    
    // Behavioral anomalies list
    const anomaliesElement = document.getElementById('behavioralAnomalies');
    if (anomaliesElement && analysis.behavioral_anomalies) {
        if (analysis.behavioral_anomalies.length === 0) {
            anomaliesElement.innerHTML = '<span class="text-green-600">✓ No behavioral anomalies detected</span>';
        } else {
            const anomaliesList = analysis.behavioral_anomalies.map(anomaly => 
                `<li class="text-red-600">⚠ ${anomaly.replace(/_/g, ' ')}</li>`
            ).join('');
            anomaliesElement.innerHTML = `<ul class="list-none">${anomaliesList}</ul>`;
        }
    }
}

/**
 * Generate adversarial (scammer) example with realistic fraud patterns
 */
function generateAdversarialExample() {
    const scammerTypes = [
        {
            name: 'Identity Thief',
            pattern: {
                monthly_income: Math.round(15000 + Math.random() * 20000), // Normal income
                community_standing_score: Math.random() * 0.3, // Very low community ties
                digital_activity_level: Math.round(8 + Math.random() * 2), // High digital activity
                transaction_velocity: Math.round(35 + Math.random() * 40), // High transaction velocity
                regional_context: ['luzon_metro', 'visayas_urban'][Math.floor(Math.random() * 2)]
            }
        },
        {
            name: 'Cultural Mimic',
            pattern: {
                monthly_income: Math.round(25000 + Math.random() * 15000), // Reasonable income
                community_standing_score: 0.95 + Math.random() * 0.05, // Perfect score (suspicious)
                digital_activity_level: Math.round(5 + Math.random() * 3), // Normal digital activity
                transaction_velocity: Math.round(45 + Math.random() * 35), // Very high velocity
                regional_context: 'mindanao_rural' // Claims rural but has sophisticated patterns
            }
        },
        {
            name: 'Synthetic Profile',
            pattern: {
                monthly_income: Math.round(50000 + Math.random() * 150000), // Unrealistically high
                community_standing_score: 0.1 + Math.random() * 0.2, // Low community standing
                digital_activity_level: Math.round(1 + Math.random() * 2), // Unusually low for income level
                transaction_velocity: Math.round(60 + Math.random() * 40), // Extremely high
                regional_context: 'luzon_rural' // Claims rural but income doesn't match
            }
        },
        {
            name: 'Professional Scammer',
            pattern: {
                monthly_income: Math.round(80000 + Math.random() * 120000), // Very high income
                community_standing_score: 0.85 + Math.random() * 0.1, // Good but not perfect
                digital_activity_level: Math.round(9 + Math.random()), // Very high digital skills
                transaction_velocity: Math.round(25 + Math.random() * 30), // High but not extreme
                regional_context: 'luzon_metro' // Claims metro to justify high income
            }
        },
        {
            name: 'Money Mule',
            pattern: {
                monthly_income: Math.round(8000 + Math.random() * 12000), // Low income
                community_standing_score: 0.6 + Math.random() * 0.2, // Average community standing
                digital_activity_level: Math.round(3 + Math.random() * 3), // Low-medium digital skills
                transaction_velocity: Math.round(80 + Math.random() * 50), // Extremely high velocity
                regional_context: ['luzon_rural', 'visayas_rural', 'mindanao_rural'][Math.floor(Math.random() * 3)]
            }
        }
    ];

    const selectedType = scammerTypes[Math.floor(Math.random() * scammerTypes.length)];
    applyPatternToControls(selectedType.pattern);
    analyzeScamRisk();
    
    showSuccessNotification(`Generated ${selectedType.name} scammer profile pattern`);
}

/**
 * Generate legitimate example with authentic Filipino patterns
 */
function generateLegitimateExample() {
    const legitimateTypes = [
        {
            name: 'Rural Entrepreneur',
            pattern: {
                monthly_income: Math.round(8000 + Math.random() * 12000),
                community_standing_score: 0.75 + Math.random() * 0.2,
                digital_activity_level: Math.round(3 + Math.random() * 3),
                transaction_velocity: Math.round(8 + Math.random() * 12),
                regional_context: ['luzon_rural', 'visayas_rural', 'mindanao_rural'][Math.floor(Math.random() * 3)]
            }
        },
        {
            name: 'Urban Gig Worker',
            pattern: {
                monthly_income: Math.round(18000 + Math.random() * 22000),
                community_standing_score: 0.55 + Math.random() * 0.25,
                digital_activity_level: Math.round(6 + Math.random() * 3),
                transaction_velocity: Math.round(15 + Math.random() * 20),
                regional_context: ['luzon_metro', 'visayas_urban', 'mindanao_urban'][Math.floor(Math.random() * 3)]
            }
        },
        {
            name: 'Small Business Owner',
            pattern: {
                monthly_income: Math.round(25000 + Math.random() * 20000),
                community_standing_score: 0.8 + Math.random() * 0.15,
                digital_activity_level: Math.round(5 + Math.random() * 4),
                transaction_velocity: Math.round(12 + Math.random() * 18),
                regional_context: ['luzon_metro', 'visayas_urban'][Math.floor(Math.random() * 2)]
            }
        },
        {
            name: 'Seasonal Worker',
            pattern: {
                monthly_income: Math.round(12000 + Math.random() * 8000),
                community_standing_score: 0.7 + Math.random() * 0.2,
                digital_activity_level: Math.round(4 + Math.random() * 3),
                transaction_velocity: Math.round(10 + Math.random() * 15),
                regional_context: ['luzon_rural', 'visayas_rural', 'mindanao_rural'][Math.floor(Math.random() * 3)]
            }
        },
        {
            name: 'Tech-Savvy Youth',
            pattern: {
                monthly_income: Math.round(20000 + Math.random() * 15000),
                community_standing_score: 0.65 + Math.random() * 0.2,
                digital_activity_level: Math.round(8 + Math.random() * 2),
                transaction_velocity: Math.round(20 + Math.random() * 15),
                regional_context: ['luzon_metro', 'visayas_urban'][Math.floor(Math.random() * 2)]
            }
        },
        {
            name: 'Traditional Family Head',
            pattern: {
                monthly_income: Math.round(15000 + Math.random() * 10000),
                community_standing_score: 0.85 + Math.random() * 0.1,
                digital_activity_level: Math.round(2 + Math.random() * 3),
                transaction_velocity: Math.round(6 + Math.random() * 12),
                regional_context: ['luzon_rural', 'visayas_rural', 'mindanao_rural'][Math.floor(Math.random() * 3)]
            }
        }
    ];

    const selectedType = legitimateTypes[Math.floor(Math.random() * legitimateTypes.length)];
    applyPatternToControls(selectedType.pattern);
    analyzeScamRisk();
    
    showSuccessNotification(`Generated ${selectedType.name} legitimate profile pattern`);
}

/**
 * Apply pattern to UI controls
 */
function applyPatternToControls(pattern) {
    const incomeInput = document.getElementById('testIncome');
    const communityInput = document.getElementById('testCommunityStanding');
    const digitalInput = document.getElementById('testDigitalActivity');
    const transactionInput = document.getElementById('testTransactionVelocity');
    const regionInput = document.getElementById('testRegion');

    if (incomeInput) incomeInput.value = Math.round(pattern.monthly_income);
    if (communityInput) {
        communityInput.value = pattern.community_standing_score.toFixed(2);
        const displayElement = document.getElementById('communityStandingValue');
        if (displayElement) displayElement.textContent = pattern.community_standing_score.toFixed(2);
    }
    if (digitalInput) {
        digitalInput.value = Math.round(pattern.digital_activity_level);
        const displayElement = document.getElementById('digitalActivityValue');
        if (displayElement) displayElement.textContent = Math.round(pattern.digital_activity_level);
    }
    if (transactionInput) transactionInput.value = Math.round(pattern.transaction_velocity);
    if (regionInput) regionInput.value = pattern.regional_context;
}

/**
 * Create initial behavior pattern chart
 */
function createInitialBehaviorChart() {
    const config = {
        type: 'radar',
        data: {
            labels: ['Income\nConsistency', 'Community\nValidation', 'Digital\nBehavior', 'Transaction\nPattern'],
            datasets: [
                {
                    label: 'Current Profile',
                    data: [80, 85, 75, 80],
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    borderWidth: 2
                },
                {
                    label: 'Legitimate Baseline',
                    data: [80, 90, 85, 80],
                    borderColor: '#6B7280',
                    backgroundColor: 'rgba(107, 114, 128, 0.1)',
                    borderWidth: 1,
                    borderDash: [5, 5]
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

    createChart('behaviorPatternChart', config);
}

/**
 * Handle data updates from other components
 */
function updateScamDetectionData(dataInfo) {
    console.log('Scam Detection received data update:', dataInfo.metadata);
    // Re-analyze current persona with new cultural data
    analyzeScamRisk();
}

/**
 * Get scam detection results for external use
 */
function getScamDetectionResults() {
    return {
        currentAnalysis: scamDetectionState.currentAnalysis,
        testPersona: scamDetectionState.testPersona,
        riskThresholds: scamDetectionState.riskThresholds
    };
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeScamDetection();
});

// Export functions for global access
window.analyzeScamRisk = analyzeScamRisk;
window.generateAdversarialExample = generateAdversarialExample;
window.generateLegitimateExample = generateLegitimateExample;
window.updateScamDetectionData = updateScamDetectionData;
window.getScamDetectionResults = getScamDetectionResults;
window.performRiskAnalysis = performRiskAnalysis;
window.calculateScamRisk = calculateScamRisk;
window.analyzeDataConsistency = analyzeDataConsistency;
window.analyzeSyntheticProfile = analyzeSyntheticProfile;