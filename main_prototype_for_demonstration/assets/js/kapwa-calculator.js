/**
 * HARAYA KapwaScore Calculator
 * =============================
 * 
 * Interactive calculator that simulates the cultural intelligence scoring algorithm
 * using pre-computed data and real-time adjustments for demonstration purposes.
 * 
 * Features:
 * - Real-time score calculation based on cultural factors
 * - Visual feedback with cultural context
 * - Authentic Filipino values integration
 * - Smooth animations and transitions
 */

class KapwaScoreCalculator {
    constructor() {
        this.model = null;
        this.currentPersona = null;
        this.currentScore = 500;
        this.animationDuration = 800;
        this.scoreChart = null;
        
        // Load the scoring model
        this.loadModel();
        
        // Bind event handlers
        this.bindEvents();
        
        // Initialize UI
        this.initializeUI();
    }
    
    async loadModel() {
        try {
            const response = await fetch('data/kapwa_score_model.json');
            this.model = await response.json();
            console.log('KapwaScore model loaded successfully');
        } catch (error) {
            console.error('Error loading KapwaScore model:', error);
            // Fallback to basic model
            this.model = this.getFallbackModel();
        }
    }
    
    getFallbackModel() {
        return {
            base_score: 500,
            maximum_score: 1000,
            cultural_factors: [
                {
                    id: "kapwa_network",
                    label: "Kapwa Network Strength",
                    weight: 0.30,
                    max_points: 300
                },
                {
                    id: "bayanihan_participation",
                    label: "Bayanihan Community Spirit",
                    weight: 0.25,
                    max_points: 250
                },
                {
                    id: "utang_na_loob_integrity",
                    label: "Utang na Loob (Debt of Gratitude)",
                    weight: 0.25,
                    max_points: 250
                },
                {
                    id: "traditional_credit",
                    label: "Traditional Financial Factors",
                    weight: 0.20,
                    max_points: 200
                }
            ]
        };
    }
    
    bindEvents() {
        // Listen for slider changes
        document.addEventListener('input', (e) => {
            if (e.target.matches('[data-cultural-factor]')) {
                this.updateScore();
            }
        });
        
        // Listen for persona selection
        document.addEventListener('change', (e) => {
            if (e.target.id === 'personaSelect') {
                this.loadPersona(e.target.value);
            }
        });
        
        // Listen for calculate button
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-calculate-score]')) {
                this.calculateScore();
            }
        });
    }
    
    async initializeUI() {
        // Wait for model to load
        while (!this.model) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Create the calculator interface
        this.createCalculatorInterface();
        
        // Load default persona (Maria Santos)
        this.loadPersona('maria_santos_sari_sari');
    }
    
    createCalculatorInterface() {
        const calculatorContainer = document.getElementById('kapwaCalculator');
        if (!calculatorContainer) return;
        
        calculatorContainer.innerHTML = `
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <!-- Input Panel -->
                <div class="space-y-6">
                    <!-- Persona Selection -->
                    <div class="bg-red-rose p-6 rounded-2xl border border-bpi-red/20">
                        <h3 class="text-xl font-semibold text-sea-blue mb-4 flex items-center">
                            👤 Choose a Community Member
                        </h3>
                        <select id="personaSelect" class="input-cultural w-full">
                            <option value="">Select a persona...</option>
                            <option value="maria_santos_sari_sari">Maria Santos - Sari-Sari Store Owner</option>
                            <option value="juan_dela_cruz_driver">Juan Dela Cruz - Grab Driver</option>
                            <option value="carmen_reyes_seamstress">Carmen Reyes - Home-based Seamstress</option>
                            <option value="ricardo_santos_construction">Ricardo Santos - Construction Worker</option>
                            <option value="josephine_cruz_online_seller">Josephine Cruz - Online Seller</option>
                        </select>
                        <div id="personaDescription" class="mt-4 p-4 bg-white/50 rounded-xl text-sm text-rich-brown"></div>
                    </div>
                    
                    <!-- Cultural Factors -->
                    <div class="space-y-4" id="culturalFactors">
                        ${this.renderCulturalFactors()}
                    </div>
                    
                    <!-- Calculate Button -->
                    <button data-calculate-score class="btn-cultural btn-primary w-full py-4 text-lg">
                        Calculate KapwaScore™ 🧮
                    </button>
                </div>
                
                <!-- Results Panel -->
                <div class="space-y-6">
                    <!-- Score Display -->
                    <div class="bg-white rounded-2xl shadow-lg p-8 text-center card-cultural">
                        <h3 class="text-2xl font-bold text-sea-blue mb-6">Your KapwaScore™</h3>
                        <div class="relative w-48 h-48 mx-auto mb-6">
                            <canvas id="scoreChart" width="192" height="192"></canvas>
                            <div class="absolute inset-0 flex items-center justify-center">
                                <div class="text-center">
                                    <div id="scoreValue" class="text-5xl font-bold text-sea-blue transition-all duration-800">${this.currentScore}</div>
                                    <div class="text-sm text-rich-brown opacity-75">out of 1000</div>
                                </div>
                            </div>
                        </div>
                        <div id="riskCategory" class="text-xl font-semibold text-tropical-green mb-2">Loading...</div>
                        <div id="loanEligibility" class="text-rich-brown opacity-75">Calculating eligibility...</div>
                    </div>
                    
                    <!-- Score Breakdown -->
                    <div class="bg-white rounded-2xl shadow-lg p-6 card-cultural">
                        <h4 class="text-lg font-semibold text-sea-blue mb-4">Cultural Intelligence Breakdown</h4>
                        <div id="scoreBreakdown" class="space-y-3">
                            ${this.renderScoreBreakdown()}
                        </div>
                    </div>
                    
                    <!-- Cultural Context -->
                    <div class="bg-red-rose p-6 rounded-2xl border border-bpi-red/20">
                        <h4 class="text-lg font-semibold text-tropical-green mb-3 flex items-center">
                            🤝 Cultural Context
                        </h4>
                        <div id="culturalExplanation" class="text-sm text-rich-brown">
                            <p>Your KapwaScore reflects the strength of your community connections and cultural integration. Higher scores indicate stronger trust networks and cultural alignment.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Initialize the score chart
        this.initializeScoreChart();
    }
    
    renderCulturalFactors() {
        if (!this.model || !this.model.cultural_factors) return '';
        
        return this.model.cultural_factors.map((factor, index) => {
            const colors = ['sea-blue', 'tropical-green', 'warm-orange', 'purple-600'];
            const color = colors[index % colors.length];
            
            return `
                <div class="bg-white rounded-2xl p-6 card-cultural border-l-4 border-${color}">
                    <h4 class="text-lg font-semibold text-${color} mb-3">${factor.label}</h4>
                    <p class="text-sm text-rich-brown opacity-75 mb-4">${factor.description || ''}</p>
                    
                    <div class="space-y-2">
                        <div class="flex justify-between text-sm">
                            <span>Strength Level</span>
                            <span id="${factor.id}_value" class="font-semibold">50%</span>
                        </div>
                        <input 
                            type="range" 
                            id="${factor.id}_slider" 
                            data-cultural-factor="${factor.id}"
                            min="0" 
                            max="100" 
                            value="50"
                            class="slider-cultural w-full"
                        >
                        <div class="flex justify-between text-xs text-rich-brown opacity-50">
                            <span>Weak</span>
                            <span>Strong</span>
                        </div>
                    </div>
                    
                    <div class="mt-3 text-xs text-rich-brown opacity-75">
                        Weight in Score: ${Math.round(factor.weight * 100)}%
                    </div>
                </div>
            `;
        }).join('');
    }
    
    renderScoreBreakdown() {
        if (!this.model || !this.model.cultural_factors) return '';
        
        return this.model.cultural_factors.map((factor, index) => {
            const colors = ['sea-blue', 'tropical-green', 'warm-orange', 'purple-600'];
            const color = colors[index % colors.length];
            
            return `
                <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span class="text-sm font-medium">${factor.label}</span>
                    <div class="flex items-center space-x-2">
                        <div class="w-20 bg-gray-200 rounded-full h-2">
                            <div id="${factor.id}_progress" class="bg-${color} h-2 rounded-full transition-all duration-500" style="width: 50%"></div>
                        </div>
                        <span id="${factor.id}_score" class="text-sm font-bold text-${color} w-16 text-right">125/250</span>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    async loadPersona(personaId) {
        if (!personaId) return;
        
        try {
            const response = await fetch('data/personas.json');
            const personas = await response.json();
            this.currentPersona = personas.find(p => p.id === personaId);
            
            if (this.currentPersona) {
                this.updatePersonaUI();
                this.loadPersonaData();
                this.updateScore();
            }
        } catch (error) {
            console.error('Error loading persona:', error);
        }
    }
    
    updatePersonaUI() {
        const description = document.getElementById('personaDescription');
        if (description && this.currentPersona) {
            description.innerHTML = `
                <div class="flex items-start space-x-4">
                    <div class="text-3xl">👤</div>
                    <div>
                        <h5 class="font-semibold text-sea-blue">${this.currentPersona.name}</h5>
                        <p class="text-sm opacity-75 mb-2">${this.currentPersona.occupation} • Age ${this.currentPersona.age} • ${this.currentPersona.location}</p>
                        <p class="text-sm italic">"${this.currentPersona.story}"</p>
                    </div>
                </div>
            `;
        }
    }
    
    loadPersonaData() {
        if (!this.currentPersona || !this.currentPersona.cultural_profile) return;
        
        // Map persona cultural profile to sliders
        const mappings = {
            'kapwa_network': this.currentPersona.cultural_profile.kapwa_network_strength * 100,
            'bayanihan_participation': this.currentPersona.cultural_profile.bayanihan_participation * 100,
            'utang_na_loob_integrity': this.currentPersona.cultural_profile.utang_na_loob_fulfillment * 100,
            'traditional_credit': (this.currentPersona.financial_behavior.savings_discipline + 
                                  this.currentPersona.financial_behavior.bill_payment_consistency) * 50
        };
        
        // Update sliders
        Object.entries(mappings).forEach(([factorId, value]) => {
            const slider = document.getElementById(`${factorId}_slider`);
            const valueDisplay = document.getElementById(`${factorId}_value`);
            
            if (slider) {
                slider.value = Math.round(value);
                this.animateSliderChange(slider, Math.round(value));
            }
            
            if (valueDisplay) {
                valueDisplay.textContent = `${Math.round(value)}%`;
            }
        });
    }
    
    animateSliderChange(slider, targetValue) {
        const currentValue = parseInt(slider.value);
        const increment = targetValue > currentValue ? 1 : -1;
        const steps = Math.abs(targetValue - currentValue);
        const stepDuration = this.animationDuration / steps;
        
        let step = 0;
        const animate = () => {
            if (step < steps) {
                slider.value = currentValue + (increment * step);
                step++;
                setTimeout(animate, stepDuration);
            }
        };
        
        if (steps > 0) {
            animate();
        }
    }
    
    updateScore() {
        if (!this.model) return;
        
        let totalScore = this.model.base_score;
        const factorScores = {};
        
        // Calculate each factor score
        this.model.cultural_factors.forEach(factor => {
            const slider = document.getElementById(`${factor.id}_slider`);
            const valueDisplay = document.getElementById(`${factor.id}_value`);
            const scoreDisplay = document.getElementById(`${factor.id}_score`);
            const progressBar = document.getElementById(`${factor.id}_progress`);
            
            if (slider) {
                const sliderValue = parseInt(slider.value);
                const factorScore = Math.round((sliderValue / 100) * factor.max_points);
                
                factorScores[factor.id] = factorScore;
                totalScore += factorScore;
                
                // Update UI elements
                if (valueDisplay) {
                    valueDisplay.textContent = `${sliderValue}%`;
                }
                
                if (scoreDisplay) {
                    scoreDisplay.textContent = `${factorScore}/${factor.max_points}`;
                }
                
                if (progressBar) {
                    progressBar.style.width = `${sliderValue}%`;
                }
            }
        });
        
        // Cap at maximum score
        totalScore = Math.min(totalScore, this.model.maximum_score);
        this.currentScore = totalScore;
        
        // Update main score display
        this.updateScoreDisplay(totalScore);
        this.updateRiskCategory(totalScore);
        this.updateCulturalExplanation(factorScores);
        this.updateScoreChart(totalScore);
    }
    
    updateScoreDisplay(score) {
        const scoreElement = document.getElementById('scoreValue');
        if (scoreElement) {
            // Animate score change
            const currentScore = parseInt(scoreElement.textContent) || 500;
            this.animateScore(scoreElement, currentScore, score);
        }
    }
    
    animateScore(element, fromScore, toScore) {
        const duration = this.animationDuration;
        const startTime = performance.now();
        const difference = toScore - fromScore;
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Use easing function
            const easedProgress = this.easeOutQuart(progress);
            const currentScore = Math.round(fromScore + (difference * easedProgress));
            
            element.textContent = currentScore;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    easeOutQuart(t) {
        return 1 - Math.pow(1 - t, 4);
    }
    
    updateRiskCategory(score) {
        const categoryElement = document.getElementById('riskCategory');
        const eligibilityElement = document.getElementById('loanEligibility');
        
        if (!categoryElement || !this.model.risk_categories) return;
        
        // Find appropriate risk category
        const category = this.model.risk_categories.find(cat => 
            score >= cat.score_range[0] && score <= cat.score_range[1]
        );
        
        if (category) {
            categoryElement.textContent = category.name;
            categoryElement.style.color = category.color;
            
            if (eligibilityElement) {
                eligibilityElement.textContent = `Eligible for ${category.loan_eligibility} loan`;
            }
        }
    }
    
    updateCulturalExplanation(factorScores) {
        const explanationElement = document.getElementById('culturalExplanation');
        if (!explanationElement) return;
        
        // Generate contextual explanation based on scores
        let explanation = this.generateCulturalExplanation(factorScores);
        
        explanationElement.innerHTML = `<p>${explanation}</p>`;
    }
    
    generateCulturalExplanation(factorScores) {
        if (!this.currentPersona) {
            return "Your KapwaScore reflects the strength of your community connections and cultural integration. Higher scores indicate stronger trust networks and cultural alignment.";
        }
        
        const strongestFactor = Object.entries(factorScores).reduce((a, b) => 
            factorScores[a[0]] > factorScores[b[0]] ? a : b
        )[0];
        
        const explanations = {
            'kapwa_network': `Your strong Kapwa network reflects deep community integration. Like ${this.currentPersona.name}, your shared identity with community members creates a foundation of trust that traditional credit scoring cannot capture.`,
            'bayanihan_participation': `Your active participation in Bayanihan activities demonstrates reliability through community cooperation. This collaborative spirit, evident in ${this.currentPersona.name}'s community involvement, is a powerful indicator of financial responsibility.`,
            'utang_na_loob_integrity': `Your strong Utang na Loob relationships show consistent fulfillment of reciprocal obligations. This cultural value, deeply embedded in Filipino society, creates sustainable repayment incentives that benefit the entire community.`,
            'traditional_credit': `Your traditional financial metrics provide a solid foundation, enhanced by cultural intelligence. Combined with community trust factors, this creates a comprehensive view of your financial reliability.`
        };
        
        return explanations[strongestFactor] || explanations['kapwa_network'];
    }
    
    initializeScoreChart() {
        const canvas = document.getElementById('scoreChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        this.scoreChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [this.currentScore, 1000 - this.currentScore],
                    backgroundColor: [
                        this.getScoreColor(this.currentScore),
                        '#E5E7EB'
                    ],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                animation: {
                    animateRotate: true,
                    duration: this.animationDuration
                }
            }
        });
    }
    
    updateScoreChart(score) {
        if (!this.scoreChart) return;
        
        const color = this.getScoreColor(score);
        this.scoreChart.data.datasets[0].data = [score, 1000 - score];
        this.scoreChart.data.datasets[0].backgroundColor = [color, '#E5E7EB'];
        this.scoreChart.update('active');
    }
    
    getScoreColor(score) {
        if (score >= 850) return '#8B0000';      // Dark red (highest)
        if (score >= 750) return '#C41E3A';      // BPI red
        if (score >= 650) return '#DC143C';      // Light red
        if (score >= 550) return '#800020';      // Burgundy
        return '#FFE4E1';                        // Very light red (lowest)
    }
    
    calculateScore() {
        // Add visual feedback
        const button = document.querySelector('[data-calculate-score]');
        if (button) {
            button.textContent = 'Calculating... 🧮';
            button.disabled = true;
            
            setTimeout(() => {
                button.textContent = 'Calculate KapwaScore™ 🧮';
                button.disabled = false;
                
                // Trigger score update with animation
                this.updateScore();
                
                // Scroll to results
                document.getElementById('scoreValue').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'center' 
                });
            }, 1500);
        }
    }
}

// Initialize the calculator when DOM is loaded or when dynamically called
function initializeKapwaCalculator() {
    if (document.getElementById('kapwaCalculator') && !window.kapwaCalculator) {
        window.kapwaCalculator = new KapwaScoreCalculator();
        console.log('KapwaScore Calculator initialized');
    }
}

document.addEventListener('DOMContentLoaded', initializeKapwaCalculator);

// Also make it available globally
window.initializeKapwaCalculator = initializeKapwaCalculator;

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KapwaScoreCalculator;
}