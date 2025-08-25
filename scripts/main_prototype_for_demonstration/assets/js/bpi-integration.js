/**
 * HARAYA BPI Integration Module
 * ============================
 * 
 * Simulates the BPI BanKo integration experience showing how HARAYA
 * enhances existing BPI products with cultural intelligence
 * 
 * Features:
 * - Interactive BPI mobile app mockup
 * - Cultural intelligence integration demonstration
 * - Real-time loan enhancement calculations
 * - Partnership value proposition display
 */

class BPIIntegrationDemo {
    constructor() {
        this.integrationData = null;
        this.currentPersona = null;
        this.enhancementResults = null;
        
        this.init();
    }
    
    async init() {
        try {
            await this.loadIntegrationData();
            this.setupEventListeners();
        } catch (error) {
            console.error('Error initializing BPI integration demo:', error);
        }
    }
    
    async loadIntegrationData() {
        try {
            const response = await fetch('data/bpi_integration.json');
            this.integrationData = await response.json();
            console.log('BPI integration data loaded successfully');
        } catch (error) {
            console.error('Error loading BPI integration data:', error);
            this.integrationData = this.getFallbackData();
        }
    }
    
    getFallbackData() {
        return {
            bpi_products_enhanced: [
                {
                    product_name: "NegosyoKo Business Loan",
                    current_offering: {
                        loan_amount: "₱15,000 - ₱500,000",
                        interest_rate: "Starting at 12% per annum",
                        approval_time: "3-5 banking days"
                    },
                    haraya_enhancement: {
                        approval_time: "Same day approval",
                        enhanced_credit_limit: "₱75,000",
                        cultural_assessment_integration: true
                    }
                }
            ]
        };
    }
    
    setupEventListeners() {
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-bpi-action]')) {
                const action = e.target.dataset.bpiAction;
                this.handleBPIAction(action, e.target);
            }
        });
    }
    
    handleBPIAction(action, element) {
        switch (action) {
            case 'apply-loan':
                this.simulateLoanApplication();
                break;
            case 'check-eligibility':
                this.checkEnhancedEligibility();
                break;
            case 'view-cultural-factors':
                this.showCulturalFactorsBreakdown();
                break;
            case 'simulate-approval':
                this.simulateInstantApproval();
                break;
        }
    }
    
    simulateLoanApplication() {
        const container = this.createMobileAppMockup();
        this.showApplicationFlow(container);
    }
    
    createMobileAppMockup() {
        // Create or update the BPI integration container
        const container = document.getElementById('bpiIntegration');
        if (!container) return null;
        
        container.innerHTML = `
            <div class="bg-white rounded-2xl shadow-lg overflow-hidden max-w-sm mx-auto">
                <!-- Mobile App Header -->
                <div class="bg-gradient-to-r from-blue-600 to-blue-800 px-6 py-4 text-white">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <div class="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                                <span class="text-blue-600 font-bold text-sm">BPI</span>
                            </div>
                            <div>
                                <div class="font-semibold">BPI Mobile</div>
                                <div class="text-xs opacity-75">Enhanced by HARAYA</div>
                            </div>
                        </div>
                        <div class="text-flag-yellow text-sm font-medium">🚀 AI</div>
                    </div>
                </div>
                
                <!-- App Content -->
                <div id="appContent" class="p-6">
                    <!-- Content will be dynamically loaded -->
                </div>
                
                <!-- App Navigation -->
                <div class="bg-gray-50 px-6 py-4 border-t">
                    <div class="flex justify-center space-x-4">
                        <button data-bpi-action="check-eligibility" class="text-xs text-blue-600 hover:text-blue-800">
                            Check Eligibility
                        </button>
                        <button data-bpi-action="view-cultural-factors" class="text-xs text-green-600 hover:text-green-800">
                            Cultural Factors
                        </button>
                        <button data-bpi-action="simulate-approval" class="text-xs text-orange-600 hover:text-orange-800">
                            Apply Now
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        return container;
    }
    
    showApplicationFlow(container) {
        const appContent = document.getElementById('appContent');
        if (!appContent) return;
        
        // Step 1: Welcome screen
        this.showStep(appContent, 'welcome');
    }
    
    showStep(container, step) {
        const steps = {
            welcome: () => this.renderWelcomeStep(container),
            cultural_assessment: () => this.renderCulturalAssessmentStep(container),
            score_results: () => this.renderScoreResultsStep(container),
            loan_options: () => this.renderLoanOptionsStep(container),
            approval: () => this.renderApprovalStep(container)
        };
        
        if (steps[step]) {
            steps[step]();
        }
    }
    
    renderWelcomeStep(container) {
        container.innerHTML = `
            <div class="text-center space-y-4">
                <div class="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center mx-auto">
                    <span class="text-white text-2xl">👋</span>
                </div>
                <h3 class="text-lg font-semibold text-gray-800">Kumusta, Maria!</h3>
                <p class="text-sm text-gray-600">
                    Welcome to your enhanced BPI experience powered by HARAYA's cultural intelligence.
                </p>
                
                <div class="bg-gradient-to-r from-green-50 to-blue-50 p-4 rounded-lg border border-green-200">
                    <div class="flex items-center space-x-2 mb-2">
                        <span class="text-green-600">✨</span>
                        <span class="font-medium text-green-800">New Enhancement Available</span>
                    </div>
                    <p class="text-sm text-green-700">
                        Your loan eligibility has been upgraded based on your community trust network!
                    </p>
                </div>
                
                <button 
                    onclick="window.bpiDemo.showStep(this.parentElement, 'cultural_assessment')" 
                    class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors">
                    Explore Enhanced Features
                </button>
                
                <div class="text-xs text-gray-500">
                    Powered by HARAYA Cultural Intelligence
                </div>
            </div>
        `;
    }
    
    renderCulturalAssessmentStep(container) {
        container.innerHTML = `
            <div class="space-y-4">
                <div class="text-center">
                    <h3 class="text-lg font-semibold text-gray-800 mb-2">Cultural Intelligence Assessment</h3>
                    <p class="text-sm text-gray-600">Understanding your community connections</p>
                </div>
                
                <div class="space-y-3">
                    <div class="bg-yellow-50 p-3 rounded-lg border-l-4 border-yellow-400">
                        <div class="flex items-center space-x-2">
                            <span class="text-yellow-600">🤝</span>
                            <div>
                                <div class="font-medium text-yellow-800">Kapwa Network</div>
                                <div class="text-sm text-yellow-700">25 community connections identified</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-green-50 p-3 rounded-lg border-l-4 border-green-400">
                        <div class="flex items-center space-x-2">
                            <span class="text-green-600">🏘️</span>
                            <div>
                                <div class="font-medium text-green-800">Bayanihan Participation</div>
                                <div class="text-sm text-green-700">Active community involvement verified</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-blue-50 p-3 rounded-lg border-l-4 border-blue-400">
                        <div class="flex items-center space-x-2">
                            <span class="text-blue-600">💝</span>
                            <div>
                                <div class="font-medium text-blue-800">Utang na Loob</div>
                                <div class="text-sm text-blue-700">Strong reciprocal relationships</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Assessment Progress -->
                <div class="space-y-2">
                    <div class="flex justify-between text-sm text-gray-600">
                        <span>Assessment Progress</span>
                        <span>90%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full" style="width: 90%"></div>
                    </div>
                </div>
                
                <button 
                    onclick="window.bpiDemo.showStep(this.parentElement, 'score_results')" 
                    class="w-full bg-green-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-green-700 transition-colors">
                    View Your KapwaScore™
                </button>
            </div>
        `;
    }
    
    renderScoreResultsStep(container) {
        container.innerHTML = `
            <div class="text-center space-y-4">
                <h3 class="text-lg font-semibold text-gray-800">Your KapwaScore™</h3>
                
                <!-- Score Display -->
                <div class="relative">
                    <div class="w-32 h-32 mx-auto mb-4 relative">
                        <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                            <path
                                d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none"
                                stroke="#e5e7eb"
                                stroke-width="2"
                                stroke-dasharray="100, 100"
                            />
                            <path
                                d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none"
                                stroke="#10b981"
                                stroke-width="3"
                                stroke-dasharray="78.5, 100"
                                stroke-linecap="round"
                            />
                        </svg>
                        <div class="absolute inset-0 flex items-center justify-center">
                            <div class="text-center">
                                <div class="text-3xl font-bold text-green-600">785</div>
                                <div class="text-xs text-gray-500">/ 1000</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="bg-green-50 p-4 rounded-lg border border-green-200">
                    <div class="font-semibold text-green-800 mb-1">Excellent Community Trust</div>
                    <div class="text-sm text-green-700">
                        Your strong community connections and cultural values indicate very low financial risk.
                    </div>
                </div>
                
                <!-- Score Breakdown -->
                <div class="space-y-2">
                    <div class="flex justify-between text-sm">
                        <span>Cultural Factors (60%)</span>
                        <span class="font-semibold text-green-600">92%</span>
                    </div>
                    <div class="flex justify-between text-sm">
                        <span>Traditional Factors (40%)</span>
                        <span class="font-semibold text-blue-600">68%</span>
                    </div>
                </div>
                
                <button 
                    onclick="window.bpiDemo.showStep(this.parentElement, 'loan_options')" 
                    class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors">
                    View Enhanced Loan Options
                </button>
            </div>
        `;
    }
    
    renderLoanOptionsStep(container) {
        container.innerHTML = `
            <div class="space-y-4">
                <div class="text-center">
                    <h3 class="text-lg font-semibold text-gray-800 mb-2">Enhanced Loan Options</h3>
                    <p class="text-sm text-gray-600">Based on your KapwaScore™ assessment</p>
                </div>
                
                <!-- Loan Options Comparison -->
                <div class="space-y-3">
                    <!-- Before HARAYA -->
                    <div class="bg-red-50 p-4 rounded-lg border border-red-200">
                        <div class="flex items-center space-x-2 mb-2">
                            <span class="text-red-500">❌</span>
                            <span class="font-medium text-red-800">Without HARAYA</span>
                        </div>
                        <div class="space-y-1 text-sm text-red-700">
                            <div>Amount: ₱15,000 - ₱25,000</div>
                            <div>Rate: 18% per annum</div>
                            <div>Approval: 5-7 banking days</div>
                            <div>Status: <strong>Limited Options</strong></div>
                        </div>
                    </div>
                    
                    <!-- With HARAYA -->
                    <div class="bg-green-50 p-4 rounded-lg border border-green-400 border-2">
                        <div class="flex items-center space-x-2 mb-2">
                            <span class="text-green-500">✅</span>
                            <span class="font-medium text-green-800">With HARAYA Enhancement</span>
                        </div>
                        <div class="space-y-1 text-sm text-green-700">
                            <div>Amount: <strong>₱75,000 - ₱150,000</strong></div>
                            <div>Rate: <strong>12% per annum (Preferred)</strong></div>
                            <div>Approval: <strong>Same Day</strong></div>
                            <div>Status: <strong>Pre-Approved</strong></div>
                        </div>
                        <div class="mt-2 p-2 bg-yellow-100 rounded text-xs text-yellow-800">
                            🏆 Community Guarantee: 25 members vouch for you
                        </div>
                    </div>
                </div>
                
                <!-- Special Features -->
                <div class="bg-blue-50 p-3 rounded-lg border border-blue-200">
                    <div class="text-sm font-medium text-blue-800 mb-1">Special HARAYA Features:</div>
                    <ul class="text-xs text-blue-700 space-y-1">
                        <li>• Reduced collateral requirements</li>
                        <li>• Community support network backup</li>
                        <li>• Cultural context consideration</li>
                        <li>• Flexible repayment aligned with business cycles</li>
                    </ul>
                </div>
                
                <button 
                    onclick="window.bpiDemo.showStep(this.parentElement, 'approval')" 
                    class="w-full bg-gradient-to-r from-green-600 to-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:opacity-90 transition-opacity">
                    Apply for Enhanced Loan
                </button>
            </div>
        `;
    }
    
    renderApprovalStep(container) {
        container.innerHTML = `
            <div class="text-center space-y-4">
                <!-- Success Animation -->
                <div class="w-16 h-16 bg-gradient-to-br from-green-400 to-green-600 rounded-full flex items-center justify-center mx-auto animate-pulse">
                    <span class="text-white text-2xl">🎉</span>
                </div>
                
                <div>
                    <h3 class="text-lg font-bold text-green-800">Congratulations, Maria!</h3>
                    <p class="text-sm text-green-600">Your loan has been approved</p>
                </div>
                
                <!-- Approval Details -->
                <div class="bg-gradient-to-r from-green-50 to-blue-50 p-4 rounded-lg border border-green-200">
                    <div class="space-y-2">
                        <div class="flex justify-between text-sm">
                            <span>Approved Amount:</span>
                            <span class="font-bold text-green-700">₱75,000</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span>Interest Rate:</span>
                            <span class="font-bold text-blue-700">12% per annum</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span>Processing Time:</span>
                            <span class="font-bold text-green-700">2.3 minutes</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span>Next Step:</span>
                            <span class="font-bold text-blue-700">Fund Disbursement</span>
                        </div>
                    </div>
                </div>
                
                <!-- Cultural Context Message -->
                <div class="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                    <p class="text-xs text-yellow-800 italic">
                        "Salamat po sa inyong tiwala. Your strong community connections and cultural values 
                        made this enhanced loan possible. Mabuhay ang bayanihan spirit!"
                    </p>
                </div>
                
                <!-- Action Buttons -->
                <div class="space-y-2">
                    <button class="w-full bg-green-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-green-700 transition-colors">
                        Accept Loan Terms
                    </button>
                    <button class="w-full border border-gray-300 text-gray-700 py-2 px-4 rounded-lg text-sm hover:bg-gray-50 transition-colors">
                        Download Approval Letter
                    </button>
                </div>
                
                <div class="text-xs text-gray-500 space-y-1">
                    <p>Loan disbursement: Within 24 hours</p>
                    <p>Cultural Intelligence Score: 785/1000</p>
                </div>
            </div>
        `;
        
        // Show success metrics after a delay
        setTimeout(() => {
            this.showBusinessImpactMetrics();
        }, 2000);
    }
    
    showBusinessImpactMetrics() {
        // Create overlay with business impact metrics
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white rounded-2xl p-8 max-w-md mx-4 text-center">
                <div class="text-4xl mb-4">📊</div>
                <h3 class="text-xl font-bold text-gray-800 mb-4">Business Impact Achieved</h3>
                
                <div class="grid grid-cols-2 gap-4 mb-6">
                    <div class="text-center">
                        <div class="text-2xl font-bold text-green-600">40%</div>
                        <div class="text-sm text-gray-600">Default Risk Reduction</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-blue-600">95%</div>
                        <div class="text-sm text-gray-600">Faster Approval</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-purple-600">27%</div>
                        <div class="text-sm text-gray-600">Higher Approval Rate</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-orange-600">65%</div>
                        <div class="text-sm text-gray-600">Cost Savings</div>
                    </div>
                </div>
                
                <p class="text-sm text-gray-600 mb-4">
                    Maria's successful loan approval demonstrates HARAYA's ability to serve 
                    the 51M unbanked Filipinos with cultural intelligence and dignity.
                </p>
                
                <button onclick="this.parentElement.parentElement.remove()" 
                        class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                    Continue Demo
                </button>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }
    
    checkEnhancedEligibility() {
        // Show eligibility comparison popup
        this.showEligibilityComparison();
    }
    
    showEligibilityComparison() {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-2xl p-6 max-w-lg mx-4">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-bold text-gray-800">Enhanced Eligibility Comparison</h3>
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" 
                            class="text-gray-500 hover:text-gray-700">✕</button>
                </div>
                
                <div class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-red-50 p-4 rounded-lg">
                            <h4 class="font-semibold text-red-800 mb-2">Traditional Assessment</h4>
                            <ul class="text-sm text-red-700 space-y-1">
                                <li>• Limited credit history</li>
                                <li>• Informal income source</li>
                                <li>• Insufficient collateral</li>
                                <li>• <strong>Result: Rejected</strong></li>
                            </ul>
                        </div>
                        
                        <div class="bg-green-50 p-4 rounded-lg">
                            <h4 class="font-semibold text-green-800 mb-2">HARAYA Enhancement</h4>
                            <ul class="text-sm text-green-700 space-y-1">
                                <li>• Strong community trust (92%)</li>
                                <li>• Cultural intelligence score: 785</li>
                                <li>• Community vouching: 25 members</li>
                                <li>• <strong>Result: Pre-approved</strong></li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-blue-800 mb-2">Cultural Intelligence Factors</h4>
                        <div class="grid grid-cols-3 gap-3 text-center">
                            <div>
                                <div class="text-lg font-bold text-blue-600">95%</div>
                                <div class="text-xs text-blue-700">Kapwa Network</div>
                            </div>
                            <div>
                                <div class="text-lg font-bold text-green-600">88%</div>
                                <div class="text-xs text-green-700">Bayanihan</div>
                            </div>
                            <div>
                                <div class="text-lg font-bold text-purple-600">92%</div>
                                <div class="text-xs text-purple-700">Utang na Loob</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    showCulturalFactorsBreakdown() {
        // Implementation for cultural factors breakdown
        console.log('Showing cultural factors breakdown');
    }
    
    simulateInstantApproval() {
        // Simulate the instant approval process
        this.showApplicationFlow(this.createMobileAppMockup());
        
        // Auto-progress through steps
        setTimeout(() => this.showStep(document.getElementById('appContent'), 'cultural_assessment'), 1000);
        setTimeout(() => this.showStep(document.getElementById('appContent'), 'score_results'), 3000);
        setTimeout(() => this.showStep(document.getElementById('appContent'), 'loan_options'), 5000);
        setTimeout(() => this.showStep(document.getElementById('appContent'), 'approval'), 7000);
    }
}

// Initialize BPI integration demo
window.bpiDemo = new BPIIntegrationDemo();

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BPIIntegrationDemo;
}