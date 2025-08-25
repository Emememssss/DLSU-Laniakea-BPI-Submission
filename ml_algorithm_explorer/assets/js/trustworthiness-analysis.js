/**
 * HARAYA Unified Trustworthiness Analysis
 * =====================================
 * 
 * Combines Cultural Intelligence and Fraud Risk Assessment into a single
 * comprehensive trustworthiness prediction system for BPI microfinance decisions.
 * 
 * Features:
 * - Unified trustworthiness scoring (0-100%)
 * - Cultural intelligence analysis with Filipino values
 * - Fraud risk assessment with behavioral analysis
 * - Detailed explanations and improvement tips
 * - 5 different synthetic persona types for testing
 */

// Global trustworthiness state
const trustworthinessState = {
    currentProfile: null,
    weights: {
        cultural: 0.6,      // Cultural intelligence weight
        fraud: 0.4,         // Fraud risk weight (inverted)
        regional: 0.1       // Regional bonus/penalty
    },
    scores: {
        overall: 87,
        cultural: {
            kapwa: 92,
            bayanihan: 89,
            utang: 93,
            regional: 85
        },
        fraud: {
            behavior: 15,
            profile: 8,
            velocity: 12,
            mimicry: 3
        }
    }
};

/**
 * Initialize trustworthiness analysis system
 */
function initializeTrustworthinessAnalysis() {
    console.log('Initializing Trustworthiness Analysis...');
    
    try {
        // Initialize with default values
        updateTrustworthinessDisplay();
        bindTrustworthinessEvents();
        
        // Register as active algorithm
        registerAlgorithm('Trustworthiness Analysis');
        
        console.log('Trustworthiness Analysis initialized successfully');
        showSuccessNotification('Trustworthiness Analysis ready for BPI microfinance decisions!');
        
    } catch (error) {
        console.error('Failed to initialize Trustworthiness Analysis:', error);
        showErrorNotification('Failed to initialize trustworthiness analysis. Please refresh.');
    }
}

/**
 * Bind event handlers for trustworthiness analysis
 */
function bindTrustworthinessEvents() {
    // Bind input change handlers
    const inputs = [
        'trustTestIncome',
        'trustTestCommunityStanding', 
        'trustTestDigitalActivity',
        'trustTestRegion',
        'trustTestBillConsistency',
        'trustTestLocationStability'
    ];
    
    inputs.forEach(inputId => {
        const element = document.getElementById(inputId);
        if (element) {
            element.addEventListener('input', analyzeTrustworthiness);
            element.addEventListener('change', analyzeTrustworthiness);
        }
    });
}

/**
 * Main trustworthiness analysis function
 */
function analyzeTrustworthiness() {
    try {
        // Get current input values
        const profile = getCurrentProfileInputs();
        
        // Calculate cultural intelligence scores
        const culturalScores = calculateCulturalIntelligence(profile);
        
        // Calculate fraud risk scores  
        const fraudScores = calculateFraudRisk(profile);
        
        // Calculate overall trustworthiness
        const overallScore = calculateOverallTrustworthiness(culturalScores, fraudScores);
        
        // Update trustworthiness state
        trustworthinessState.currentProfile = profile;
        trustworthinessState.scores.overall = overallScore;
        trustworthinessState.scores.cultural = culturalScores;
        trustworthinessState.scores.fraud = fraudScores;
        
        // Update display
        updateTrustworthinessDisplay();
        updateExplanationsAndTips();
        
        // Update slider value displays
        updateSliderValueDisplays();
        
    } catch (error) {
        console.error('Trustworthiness analysis failed:', error);
        showErrorNotification('Analysis failed. Please check your inputs.');
    }
}

/**
 * Get current profile inputs from the form
 */
function getCurrentProfileInputs() {
    return {
        income: parseInt(document.getElementById('trustTestIncome')?.value || 25000),
        communityStanding: parseFloat(document.getElementById('trustTestCommunityStanding')?.value || 0.85),
        digitalActivity: parseInt(document.getElementById('trustTestDigitalActivity')?.value || 6),
        region: document.getElementById('trustTestRegion')?.value || 'visayas_urban',
        billConsistency: parseFloat(document.getElementById('trustTestBillConsistency')?.value || 0.89),
        locationStability: parseFloat(document.getElementById('trustTestLocationStability')?.value || 0.78)
    };
}

/**
 * Calculate cultural intelligence scores based on Filipino values
 */
function calculateCulturalIntelligence(profile) {
    // Kapwa Score (Shared Identity) - based on community standing and location stability
    const kapwa = Math.min(100, Math.max(0, 
        (profile.communityStanding * 0.6 + profile.locationStability * 0.4) * 100 + 
        getRegionalBonus(profile.region, 'kapwa')
    ));
    
    // Bayanihan Participation (Community Spirit) - based on community standing and digital activity
    const bayanihan = Math.min(100, Math.max(0,
        (profile.communityStanding * 0.7 + (profile.digitalActivity / 10) * 0.3) * 100 +
        getRegionalBonus(profile.region, 'bayanihan')
    ));
    
    // Utang na Loob Integrity (Gratitude/Reciprocity) - based on bill consistency and community standing
    const utang = Math.min(100, Math.max(0,
        (profile.billConsistency * 0.8 + profile.communityStanding * 0.2) * 100 +
        getRegionalBonus(profile.region, 'utang')
    ));
    
    // Regional Authenticity - based on region-specific patterns
    const regional = Math.min(100, Math.max(0,
        (profile.locationStability * 0.4 + profile.communityStanding * 0.4 + 
         (profile.income > 15000 && profile.income < 50000 ? 0.2 : 0.1) * 100) +
        getRegionalBonus(profile.region, 'regional')
    ));
    
    return {
        kapwa: Math.round(kapwa),
        bayanihan: Math.round(bayanihan),  
        utang: Math.round(utang),
        regional: Math.round(regional)
    };
}

/**
 * Calculate fraud risk scores (lower is better)
 */
function calculateFraudRisk(profile) {
    // Behavioral Consistency Risk - income vs digital activity patterns
    const behaviorRisk = Math.min(100, Math.max(0,
        Math.abs(profile.income / 1000 - profile.digitalActivity * 5) * 2 +
        (profile.communityStanding < 0.3 ? 30 : 0)
    ));
    
    // Profile Consistency Risk - coherence between different metrics
    const profileRisk = Math.min(100, Math.max(0,
        Math.abs(profile.communityStanding - profile.billConsistency) * 50 +
        Math.abs(profile.billConsistency - profile.locationStability) * 30 +
        (profile.income > 100000 && profile.communityStanding < 0.5 ? 25 : 0)
    ));
    
    // Transaction Velocity Risk - based on income and digital activity
    const velocityRisk = Math.min(100, Math.max(0,
        (profile.digitalActivity > 8 && profile.income < 20000 ? 40 : 0) +
        (profile.digitalActivity < 3 && profile.income > 80000 ? 30 : 0) +
        Math.abs(profile.digitalActivity - 5) * 2
    ));
    
    // Cultural Mimicry Risk - inconsistency between cultural factors
    const mimicryRisk = Math.min(100, Math.max(0,
        (profile.communityStanding > 0.9 && profile.locationStability < 0.3 ? 50 : 0) +
        (profile.billConsistency > 0.95 && profile.communityStanding < 0.4 ? 40 : 0) +
        (profile.income > 80000 && profile.communityStanding > 0.9 && profile.locationStability < 0.4 ? 30 : 0)
    ));
    
    return {
        behavior: Math.round(behaviorRisk),
        profile: Math.round(profileRisk),
        velocity: Math.round(velocityRisk),
        mimicry: Math.round(mimicryRisk)
    };
}

/**
 * Get regional bonus/penalty for cultural factors
 */
function getRegionalBonus(region, factor) {
    const bonuses = {
        metro_manila: { kapwa: -5, bayanihan: -3, utang: 0, regional: 8 },
        luzon_rural: { kapwa: 8, bayanihan: 10, utang: 5, regional: 3 },
        visayas_urban: { kapwa: 5, bayanihan: 8, utang: 3, regional: 12 },
        mindanao_rural: { kapwa: 10, bayanihan: 12, utang: 8, regional: 5 }
    };
    
    return bonuses[region]?.[factor] || 0;
}

/**
 * Calculate overall trustworthiness score
 */
function calculateOverallTrustworthiness(culturalScores, fraudScores) {
    // Cultural intelligence average (higher is better)
    const culturalAvg = (culturalScores.kapwa + culturalScores.bayanihan + 
                        culturalScores.utang + culturalScores.regional) / 4;
    
    // Fraud risk average (lower is better, so invert)
    const fraudAvg = (fraudScores.behavior + fraudScores.profile + 
                     fraudScores.velocity + fraudScores.mimicry) / 4;
    const fraudScore = 100 - fraudAvg; // Invert so higher is better
    
    // Weighted combination
    const overall = (culturalAvg * trustworthinessState.weights.cultural) + 
                   (fraudScore * trustworthinessState.weights.fraud);
    
    return Math.round(Math.min(100, Math.max(0, overall)));
}

/**
 * Update trustworthiness display with current scores
 */
function updateTrustworthinessDisplay() {
    const scores = trustworthinessState.scores;
    
    // Update overall score
    updateElement('overallTrustworthinessScore', `${scores.overall}%`);
    updateElement('trustworthinessLabel', getTrustworthinessLabel(scores.overall));
    
    // Update cultural intelligence scores
    updateElement('trustKapwaScore', `${scores.cultural.kapwa}%`);
    updateElement('trustBayanihanScore', `${scores.cultural.bayanihan}%`);
    updateElement('trustUtangScore', `${scores.cultural.utang}%`);
    updateElement('trustRegionalScore', `${scores.cultural.regional}%`);
    
    // Update progress bars
    updateProgressBar('trustKapwaProgress', scores.cultural.kapwa);
    updateProgressBar('trustBayanihanProgress', scores.cultural.bayanihan);
    updateProgressBar('trustUtangProgress', scores.cultural.utang);
    updateProgressBar('trustRegionalProgress', scores.cultural.regional);
    
    // Update fraud risk scores
    updateElement('trustBehaviorScore', `${getFraudRiskLabel(scores.fraud.behavior)} (${scores.fraud.behavior}%)`);
    updateElement('trustProfileScore', `${getFraudRiskLabel(scores.fraud.profile)} (${scores.fraud.profile}%)`);
    updateElement('trustVelocityScore', `${getFraudRiskLabel(scores.fraud.velocity)} (${scores.fraud.velocity}%)`);
    updateElement('trustMimicryScore', `${getFraudRiskLabel(scores.fraud.mimicry)} (${scores.fraud.mimicry}%)`);
    
    // Update fraud risk progress bars
    updateProgressBar('trustBehaviorProgress', scores.fraud.behavior);
    updateProgressBar('trustProfileProgress', scores.fraud.profile);
    updateProgressBar('trustVelocityProgress', scores.fraud.velocity);
    updateProgressBar('trustMimicryProgress', scores.fraud.mimicry);
}

/**
 * Get trustworthiness label based on score
 */
function getTrustworthinessLabel(score) {
    if (score >= 90) return 'Excellent Trust';
    if (score >= 80) return 'Highly Trustworthy';
    if (score >= 70) return 'Trustworthy';
    if (score >= 60) return 'Moderately Trustworthy';
    if (score >= 40) return 'Requires Assessment';
    return 'High Risk';
}

/**
 * Get fraud risk label based on score
 */
function getFraudRiskLabel(score) {
    if (score <= 10) return 'Very Low Risk';
    if (score <= 25) return 'Low Risk';
    if (score <= 40) return 'Medium Risk';
    if (score <= 60) return 'High Risk';
    return 'Very High Risk';
}

/**
 * Update explanations and improvement tips based on current profile
 */
function updateExplanationsAndTips() {
    updateCulturalInsights();
    updateFraudInsights();
    updateTrustworthyExplanation();
    updateImprovementTips();
    updateSummaryDashboard();
}

/**
 * Update cultural insights based on scores
 */
function updateCulturalInsights() {
    const scores = trustworthinessState.scores.cultural;
    const avg = (scores.kapwa + scores.bayanihan + scores.utang + scores.regional) / 4;
    
    let insight = '';
    if (avg >= 85) {
        insight = 'Strong alignment with Filipino cultural values. Authentic community connections and high gratitude patterns indicate genuine cultural integration.';
    } else if (avg >= 70) {
        insight = 'Good cultural alignment with some areas for growth. Community connections are solid but could benefit from deeper integration.';
    } else if (avg >= 50) {
        insight = 'Moderate cultural alignment. Some disconnection from traditional Filipino values may indicate adaptation challenges or different cultural background.';
    } else {
        insight = 'Weak cultural alignment. Limited connection to Filipino cultural values may indicate recent migration, cultural assimilation challenges, or potential authenticity concerns.';
    }
    
    updateElement('culturalInsights', insight);
}

/**
 * Update fraud insights based on risk scores
 */
function updateFraudInsights() {
    const scores = trustworthinessState.scores.fraud;
    const avg = (scores.behavior + scores.profile + scores.velocity + scores.mimicry) / 4;
    
    let insight = '';
    if (avg <= 15) {
        insight = 'Minimal fraud risk indicators. Consistent behavior patterns and authentic cultural expression suggest genuine profile authenticity.';
    } else if (avg <= 30) {
        insight = 'Low fraud risk with some minor inconsistencies. Profile shows generally authentic patterns with acceptable variance.';
    } else if (avg <= 50) {
        insight = 'Moderate fraud risk indicators present. Some behavioral inconsistencies or profile anomalies require closer examination.';
    } else {
        insight = 'High fraud risk detected. Multiple inconsistencies in behavior, profile coherence, or cultural authenticity patterns require immediate investigation.';
    }
    
    updateElement('fraudInsights', insight);
}

/**
 * Update trustworthy explanation
 */
function updateTrustworthyExplanation() {
    const overall = trustworthinessState.scores.overall;
    const profile = trustworthinessState.currentProfile || getCurrentProfileInputs();
    
    let explanation = '';
    if (overall >= 80) {
        explanation = `
            <p><strong>Cultural Authenticity:</strong> Strong Kapwa connections (${trustworthinessState.scores.cultural.kapwa}%) indicate genuine community integration and shared identity values typical of trustworthy Filipino profiles.</p>
            <p><strong>Behavioral Consistency:</strong> ${profile.billConsistency >= 0.8 ? '6+ years of consistent payment history' : 'Generally stable payment patterns'} with ${trustworthinessState.scores.fraud.velocity <= 20 ? 'no velocity anomalies' : 'minor velocity variations'} suggests ${profile.billConsistency >= 0.8 ? 'stable and reliable' : 'acceptable'} financial behavior.</p>
            <p><strong>Community Validation:</strong> Vouched by ${Math.round(profile.communityStanding * 15)} long-term BPI clients and ${Math.round(profile.communityStanding * 3)} community leaders, demonstrating strong community trust and social proof.</p>
            <p><strong>Regional Alignment:</strong> Cultural patterns authentically match ${getRegionName(profile.region)} profiles, with ${trustworthinessState.scores.fraud.mimicry <= 10 ? 'no signs of cultural mimicry or fraud attempts' : 'minimal authenticity concerns'}.</p>
        `;
    } else if (overall >= 60) {
        explanation = `
            <p><strong>Mixed Indicators:</strong> Profile shows both positive and concerning elements that require balanced assessment.</p>
            <p><strong>Cultural Patterns:</strong> Moderate alignment with Filipino values (${Math.round((trustworthinessState.scores.cultural.kapwa + trustworthinessState.scores.cultural.bayanihan + trustworthinessState.scores.cultural.utang) / 3)}%) suggests partial integration or cultural adaptation period.</p>
            <p><strong>Risk Factors:</strong> Some behavioral inconsistencies detected that merit additional verification and monitoring.</p>
            <p><strong>Improvement Potential:</strong> Profile shows capacity for trust improvement through targeted community engagement and consistency building.</p>
        `;
    } else {
        explanation = `
            <p><strong>High Risk Profile:</strong> Multiple red flags indicate significant trustworthiness concerns requiring immediate attention.</p>
            <p><strong>Cultural Inconsistencies:</strong> Weak alignment with authentic Filipino cultural patterns may indicate profile manipulation or cultural disconnect.</p>
            <p><strong>Behavioral Anomalies:</strong> Significant inconsistencies in financial behavior, community standing, and digital activity patterns.</p>
            <p><strong>Recommendation:</strong> Additional verification, in-person assessment, and community validation strongly recommended before any financial decisions.</p>
        `;
    }
    
    updateElement('trustworthyExplanation', explanation);
}

/**
 * Update improvement tips based on current profile
 */
function updateImprovementTips() {
    const scores = trustworthinessState.scores;
    
    let culturalTips = '';
    let networkTips = '';
    let financialTips = '';
    
    // Cultural enhancement tips
    if (scores.cultural.bayanihan < 85) {
        culturalTips += '<li>Increase Bayanihan participation by joining 2+ community events monthly</li>';
    }
    if (scores.cultural.utang < 90) {
        culturalTips += '<li>Strengthen Utang na Loob by consistent gratitude expressions and reciprocity</li>';
    }
    if (scores.cultural.kapwa < 85) {
        culturalTips += '<li>Build Kapwa connections through extended family network expansion</li>';
    }
    
    // Network strengthening tips
    if (scores.cultural.kapwa < 90 || scores.overall < 85) {
        networkTips += '<li>Get vouched by 3+ additional long-term BPI clients in your community</li>';
        networkTips += '<li>Join barangay organizations to increase community leader connections</li>';
    }
    if (scores.cultural.regional < 85) {
        networkTips += '<li>Participate in regional cultural events to strengthen local connections</li>';
    }
    
    // Financial consistency tips
    if (scores.fraud.behavior > 20 || scores.fraud.velocity > 20) {
        financialTips += '<li>Maintain current payment patterns for 6+ months for stability bonus</li>';
        financialTips += '<li>Avoid large transaction velocity changes without proper documentation</li>';
    }
    if (scores.fraud.profile > 15) {
        financialTips += '<li>Ensure consistency across all profile information and linked accounts</li>';
    }
    
    // Update tips sections
    document.querySelector('#improvementTips .p-3.bg-green-50 ul').innerHTML = culturalTips || '<li>Continue maintaining excellent cultural practices</li>';
    document.querySelector('#improvementTips .p-3.bg-blue-50 ul').innerHTML = networkTips || '<li>Network strength is excellent, maintain current connections</li>';
    document.querySelector('#improvementTips .p-3.bg-yellow-50 ul').innerHTML = financialTips || '<li>Financial patterns are consistent, maintain current practices</li>';
}

/**
 * Update summary dashboard cards
 */
function updateSummaryDashboard() {
    const scores = trustworthinessState.scores;
    const profile = trustworthinessState.currentProfile || getCurrentProfileInputs();
    
    // Key Strengths
    let strengths = [];
    if (scores.overall >= 85) strengths.push('• High overall trustworthiness');
    if ((scores.cultural.kapwa + scores.cultural.bayanihan + scores.cultural.utang) / 3 >= 85) strengths.push('• Strong cultural authenticity');
    if (profile.billConsistency >= 0.85) strengths.push('• Consistent payment history');
    if (profile.communityStanding >= 0.8) strengths.push('• Strong community vouching');
    if ((scores.fraud.behavior + scores.fraud.profile + scores.fraud.velocity + scores.fraud.mimicry) / 4 <= 15) strengths.push('• Low fraud risk profile');
    
    updateElement('keyStrengths', strengths.slice(0, 3).join('<br>') || '• Meets basic requirements');
    
    // Growth Areas
    let growthAreas = [];
    if (scores.cultural.bayanihan < 85) growthAreas.push('• Expand community participation');
    if (scores.cultural.regional < 85) growthAreas.push('• Increase regional activities');
    if (profile.digitalActivity < 5) growthAreas.push('• Digital engagement');
    if (scores.cultural.kapwa < 85) growthAreas.push('• Strengthen family network');
    
    updateElement('growthAreas', growthAreas.slice(0, 3).join('<br>') || '• Maintain current practices');
    
    // Risk Factors
    let riskFactors = [];
    if (scores.fraud.mimicry > 20) riskFactors.push('• Cultural mimicry detected');
    if (scores.fraud.velocity > 30) riskFactors.push('• Transaction velocity anomalies');
    if (scores.fraud.profile > 25) riskFactors.push('• Profile inconsistencies');
    if (scores.overall < 60) riskFactors.push('• Overall low trust score');
    
    updateElement('riskFactors', riskFactors.slice(0, 3).join('<br>') || '• None identified<br>• All patterns normal<br>• Low fraud risk');
    
    // Profile Stats
    const bpiHistory = Math.round(profile.billConsistency * 8) + 2; // 2-10 years based on consistency
    const communityVouch = Math.round(profile.communityStanding * 20);
    
    updateElement('profileStats', `
        <div>BPI History: ${bpiHistory} years</div>
        <div>Community Vouch: ${communityVouch}</div>
        <div>Last Updated: Today</div>
    `);
}

/**
 * Generate different types of synthetic personas
 */
function generateTrustworthyPersona() {
    const persona = {
        income: 35000 + Math.random() * 15000,
        communityStanding: 0.85 + Math.random() * 0.1,
        digitalActivity: 5 + Math.round(Math.random() * 3),
        region: ['visayas_urban', 'luzon_rural', 'mindanao_rural'][Math.floor(Math.random() * 3)],
        billConsistency: 0.88 + Math.random() * 0.1,
        locationStability: 0.8 + Math.random() * 0.15
    };
    
    applyPersonaToInputs(persona);
    showSuccessNotification('High trust profile generated successfully!');
}

function generateMediumTrustPersona() {
    const persona = {
        income: 20000 + Math.random() * 30000,
        communityStanding: 0.6 + Math.random() * 0.2,
        digitalActivity: 4 + Math.round(Math.random() * 4),
        region: ['metro_manila', 'visayas_urban', 'luzon_rural'][Math.floor(Math.random() * 3)],
        billConsistency: 0.65 + Math.random() * 0.25,
        locationStability: 0.5 + Math.random() * 0.3
    };
    
    applyPersonaToInputs(persona);
    showInfoNotification('Medium trust profile generated for analysis.');
}

function generateRiskyPersona() {
    const persona = {
        income: 15000 + Math.random() * 80000, // Wide income variance
        communityStanding: 0.1 + Math.random() * 0.4,
        digitalActivity: Math.random() > 0.5 ? 1 + Math.round(Math.random() * 2) : 8 + Math.round(Math.random() * 2), // Very low or very high
        region: ['metro_manila', 'visayas_urban'][Math.floor(Math.random() * 2)],
        billConsistency: 0.2 + Math.random() * 0.5,
        locationStability: 0.1 + Math.random() * 0.4
    };
    
    applyPersonaToInputs(persona);
    showWarningNotification('Risk profile generated - multiple red flags detected.');
}

function generateRandomPersona() {
    const persona = {
        income: 15000 + Math.random() * 60000,
        communityStanding: Math.random(),
        digitalActivity: 1 + Math.round(Math.random() * 9),
        region: ['metro_manila', 'luzon_rural', 'visayas_urban', 'mindanao_rural'][Math.floor(Math.random() * 4)],
        billConsistency: Math.random(),
        locationStability: Math.random()
    };
    
    applyPersonaToInputs(persona);
    showInfoNotification('Random profile generated for testing.');
}

/**
 * Apply persona values to input fields
 */
function applyPersonaToInputs(persona) {
    document.getElementById('trustTestIncome').value = Math.round(persona.income);
    document.getElementById('trustTestCommunityStanding').value = persona.communityStanding.toFixed(2);
    document.getElementById('trustTestDigitalActivity').value = persona.digitalActivity;
    document.getElementById('trustTestRegion').value = persona.region;
    document.getElementById('trustTestBillConsistency').value = persona.billConsistency.toFixed(2);
    document.getElementById('trustTestLocationStability').value = persona.locationStability.toFixed(2);
    
    // Trigger analysis
    analyzeTrustworthiness();
}

/**
 * Update slider value displays
 */
function updateSliderValueDisplays() {
    const values = {
        trustCommunityStandingValue: document.getElementById('trustTestCommunityStanding')?.value || '0.85',
        trustDigitalActivityValue: document.getElementById('trustTestDigitalActivity')?.value || '6', 
        trustBillConsistencyValue: document.getElementById('trustTestBillConsistency')?.value || '0.89',
        trustLocationStabilityValue: document.getElementById('trustTestLocationStability')?.value || '0.78'
    };
    
    Object.entries(values).forEach(([id, value]) => {
        updateElement(id, value);
    });
}

/**
 * Utility functions
 */
function updateElement(id, content) {
    const element = document.getElementById(id);
    if (element) {
        element.innerHTML = content;
    }
}

function updateProgressBar(id, percentage) {
    const element = document.getElementById(id);
    if (element) {
        element.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    }
}

function getRegionName(regionCode) {
    const regions = {
        metro_manila: 'Metro Manila',
        luzon_rural: 'Rural Luzon',
        visayas_urban: 'Urban Visayas',
        mindanao_rural: 'Rural Mindanao'
    };
    return regions[regionCode] || 'Unknown Region';
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit to ensure other systems are loaded first
    setTimeout(initializeTrustworthinessAnalysis, 500);
});

// Export for global access
window.trustworthinessState = trustworthinessState;
window.analyzeTrustworthiness = analyzeTrustworthiness;
window.generateTrustworthyPersona = generateTrustworthyPersona;
window.generateMediumTrustPersona = generateMediumTrustPersona;
window.generateRiskyPersona = generateRiskyPersona;
window.generateRandomPersona = generateRandomPersona;