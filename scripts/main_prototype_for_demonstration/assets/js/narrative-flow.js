/**
 * HARAYA Narrative Flow Controller
 * ===============================
 * 
 * Orchestrates the guided tour through Maria Santos' story
 * Manages section transitions, animations, and user interactions
 * Integrates with Shepherd.js for guided tutorial experience
 * 
 * Features:
 * - Sequential narrative progression
 * - Smooth section transitions
 * - Progress tracking and navigation
 * - Responsive design adaptation
 * - Keyboard and touch controls
 */

class HarayaNarrativeFlow {
    constructor() {
        this.currentSection = 0;
        this.totalSections = 6;
        this.isTransitioning = false;
        this.tourActive = false;
        this.freeExplorationMode = false;
        this.sections = [
            'welcome-section',
            'maria-section', 
            'network-section',
            'calculator-section',
            'bpi-section',
            'summary-section'
        ];
        
        this.sectionTitles = [
            'Introduction',
            'Meet Maria',
            'Community Network',
            'KapwaScore Calculator',
            'BPI Integration',
            'Demo Summary'
        ];
        
        // Initialize
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.setupKeyboardNavigation();
        this.initializeTour();
        this.updateProgress();
    }
    
    bindEvents() {
        // Start demo button
        document.addEventListener('click', (e) => {
            if (e.target.id === 'startDemo') {
                this.startNarrative();
            }
            
            if (e.target.id === 'exploreNetwork') {
                this.transitionToSection(2); // Network section
            }
            
            if (e.target.id === 'calculateScore') {
                this.transitionToSection(3); // Calculator section
            }
            
            if (e.target.id === 'viewBPIIntegration') {
                this.transitionToSection(4); // BPI section
            }
            
            if (e.target.id === 'completeDemo') {
                this.transitionToSection(5); // Summary section
            }
            
            if (e.target.id === 'restartJourney' || e.target.id === 'restartDemo') {
                this.restartDemo();
            }
            
            if (e.target.id === 'skipTour') {
                this.skipTour();
            }
            
            if (e.target.id === 'viewFullPresentation') {
                this.viewFullPresentation();
            }
            
            // Navigation menu controls
            if (e.target.id === 'openNavigation') {
                this.toggleNavigationMenu(true);
            }
            
            if (e.target.id === 'closeNavigation') {
                this.toggleNavigationMenu(false);
            }
            
            if (e.target.id === 'enableFreeMode') {
                this.enableFreeExplorationMode();
            }
        });
        
        // Handle navigation clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-section]')) {
                const sectionIndex = parseInt(e.target.dataset.section);
                this.transitionToSection(sectionIndex);
            }
        });
        
        // Handle scroll events for section detection
        window.addEventListener('scroll', this.throttle(() => {
            this.detectCurrentSection();
        }, 100));
    }
    
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            if (this.isTransitioning) return;
            
            switch (e.key) {
                case 'ArrowRight':
                case ' ':
                    e.preventDefault();
                    this.nextSection();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.previousSection();
                    break;
                case 'Home':
                    e.preventDefault();
                    this.transitionToSection(0);
                    break;
                case 'End':
                    e.preventDefault();
                    this.transitionToSection(this.totalSections - 1);
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.skipTour();
                    break;
            }
        });
    }
    
    initializeTour() {
        // Initialize Shepherd.js tour with better z-index management
        this.tour = new Shepherd.Tour({
            useModalOverlay: true,
            defaultStepOptions: {
                classes: 'shepherd-theme-arrows cultural-tour-step',
                scrollTo: { behavior: 'smooth', block: 'center' },
                cancelIcon: {
                    enabled: true
                },
                modalOverlayOpeningPadding: 4,
                modalOverlayOpeningRadius: 8
            }
        });
        
        this.setupTourSteps();
    }
    
    setupTourSteps() {
        // Step 1: Welcome
        this.tour.addStep({
            title: 'Kumusta! Welcome to HARAYA',
            text: `
                <div class="text-center">
                    <div class="text-4xl mb-4">🌟</div>
                    <p class="mb-4">Experience the world's first culturally-intelligent fintech platform through the story of Maria Santos, a Filipino entrepreneur.</p>
                    <p class="text-sm opacity-75">Use arrow keys or click buttons to navigate.</p>
                </div>
            `,
            attachTo: {
                element: '#welcome-section',
                on: 'center'
            },
            buttons: [{
                text: 'Start Journey',
                action: () => {
                    this.tour.next();
                    this.startNarrative();
                }
            }]
        });
        
        // Step 2: Meet Maria
        this.tour.addStep({
            title: 'Meet Maria Santos',
            text: `
                <p class="mb-3">Maria is a 42-year-old sari-sari store owner who represents millions of Filipino entrepreneurs seeking financial services.</p>
                <p class="text-sm opacity-75">Notice how traditional credit scoring would reject her, while HARAYA recognizes her community value.</p>
            `,
            attachTo: {
                element: '#mariaAvatar',
                on: 'right'
            },
            buttons: [{
                text: 'Previous',
                action: () => this.tour.back()
            }, {
                text: 'Continue',
                action: () => this.tour.next()
            }]
        });
        
        // Step 3: Community Network
        this.tour.addStep({
            title: 'Understanding Trust Networks',
            text: `
                <p class="mb-3">HARAYA sees Maria not as an isolated individual, but as part of a vibrant community trust network.</p>
                <p class="text-sm opacity-75">Explore her relationships with family, suppliers, customers, and community leaders.</p>
            `,
            attachTo: {
                element: '#networkVisualization',
                on: 'left'
            },
            buttons: [{
                text: 'Previous', 
                action: () => this.tour.back()
            }, {
                text: 'Continue',
                action: () => this.tour.next()
            }]
        });
        
        // Step 4: KapwaScore Calculator
        this.tour.addStep({
            title: 'The KapwaScore™ Algorithm',
            text: `
                <p class="mb-3">Experience our revolutionary cultural intelligence algorithm that weighs Filipino values alongside traditional metrics.</p>
                <p class="text-sm opacity-75">Adjust the cultural factors to see how they impact Maria's trust score.</p>
            `,
            attachTo: {
                element: '#kapwaCalculator',
                on: 'top'
            },
            buttons: [{
                text: 'Previous',
                action: () => this.tour.back()
            }, {
                text: 'Try Calculator',
                action: () => this.tour.next()
            }]
        });
        
        // Step 5: BPI Integration
        this.tour.addStep({
            title: 'Partnership Ready',
            text: `
                <p class="mb-3">See how HARAYA enhances BPI's existing products with cultural intelligence for better financial inclusion.</p>
                <p class="text-sm opacity-75">Maria can now access loans that were previously unavailable to her.</p>
            `,
            attachTo: {
                element: '#bpiIntegration',
                on: 'top'
            },
            buttons: [{
                text: 'Previous',
                action: () => this.tour.back()
            }, {
                text: 'Complete Tour',
                action: () => {
                    this.tour.complete();
                    this.completeTour();
                }
            }]
        });
        
        // Handle tour completion
        this.tour.on('complete', () => {
            this.tourActive = false;
            this.transitionToSection(5); // Go to summary
        });
        
        this.tour.on('cancel', () => {
            this.tourActive = false;
        });
    }
    
    toggleNavigationMenu(show) {
        const navigationDropdown = document.getElementById('navigationDropdown');
        if (!navigationDropdown) return;
        
        if (show) {
            navigationDropdown.classList.remove('hidden');
            navigationDropdown.style.opacity = '0';
            // Trigger reflow
            navigationDropdown.offsetHeight;
            navigationDropdown.style.opacity = '1';
            
            // Focus first menu item for accessibility
            const firstMenuItem = navigationDropdown.querySelector('[data-section]');
            if (firstMenuItem) {
                firstMenuItem.focus();
            }
        } else {
            navigationDropdown.style.opacity = '0';
            setTimeout(() => {
                navigationDropdown.classList.add('hidden');
            }, 200);
        }
        
        // Close menu when clicking outside
        if (show) {
            setTimeout(() => {
                document.addEventListener('click', this.closeMenuOnOutsideClick.bind(this), { once: true });
            }, 100);
        }
    }
    
    closeMenuOnOutsideClick(e) {
        const navigationDropdown = document.getElementById('navigationDropdown');
        const openButton = document.getElementById('openNavigation');
        
        if (navigationDropdown && !navigationDropdown.contains(e.target) && e.target !== openButton) {
            this.toggleNavigationMenu(false);
        }
    }
    
    startNarrative() {
        if (this.currentSection === 0) {
            // Start tour immediately for welcome step
            this.startTour();
            
            // Then transition to Maria section after first step
            setTimeout(() => {
                this.transitionToSection(1); // Go to Maria section
            }, 500);
        }
    }
    
    startTour() {
        this.tourActive = true;
        this.tour.start();
    }
    
    skipTour() {
        if (this.tourActive) {
            this.tour.cancel();
            this.tourActive = false;
        }
        
        // Enable free navigation
        document.getElementById('skipTour').style.display = 'none';
        document.getElementById('restartDemo').style.display = 'inline-block';
        
        // Enable free exploration mode
        this.enableFreeExplorationMode();
    }
    
    enableFreeExplorationMode() {
        this.freeExplorationMode = true;
        this.tourActive = false;
        
        // Show navigation menu permanently
        const navigationDropdown = document.getElementById('navigationDropdown');
        const openNavigationBtn = document.getElementById('openNavigation');
        
        if (navigationDropdown && openNavigationBtn) {
            // Keep navigation always visible
            navigationDropdown.classList.remove('hidden');
            navigationDropdown.style.opacity = '1';
            navigationDropdown.style.position = 'fixed';
            navigationDropdown.style.top = '80px';
            navigationDropdown.style.right = '20px';
            navigationDropdown.style.zIndex = '40';
            
            // Hide the toggle button since menu is always visible
            openNavigationBtn.style.display = 'none';
            
            // Add close option for mobile
            this.addNavigationCloseOption();
        }
        
        // Show free exploration message
        this.showFreeExplorationMessage();
    }
    
    addNavigationCloseOption() {
        const navigationDropdown = document.getElementById('navigationDropdown');
        if (!navigationDropdown) return;
        
        // Check if we haven't already added the close option
        if (navigationDropdown.querySelector('#hideFreeMode')) return;
        
        // Add close option for mobile users
        const closeButton = document.createElement('button');
        closeButton.id = 'hideFreeMode';
        closeButton.className = 'w-full mt-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-800 transition-colors md:hidden';
        closeButton.innerHTML = '✕ Hide Menu (Mobile)';
        closeButton.onclick = () => this.disableFreeExplorationMode();
        
        navigationDropdown.querySelector('.p-4').appendChild(closeButton);
    }
    
    disableFreeExplorationMode() {
        this.freeExplorationMode = false;
        
        const navigationDropdown = document.getElementById('navigationDropdown');
        const openNavigationBtn = document.getElementById('openNavigation');
        
        if (navigationDropdown && openNavigationBtn) {
            navigationDropdown.classList.add('hidden');
            openNavigationBtn.style.display = 'inline-block';
        }
    }
    
    showFreeExplorationMessage() {
        const message = document.createElement('div');
        message.className = 'fixed top-4 left-4 z-50 bg-tropical-green text-white p-4 rounded-lg shadow-lg max-w-sm';
        message.innerHTML = `
            <div class="flex items-start space-x-3">
                <span class="text-xl">🗺️</span>
                <div>
                    <div class="font-semibold">Free Exploration Enabled!</div>
                    <div class="text-sm opacity-90">Use the navigation menu to jump between sections freely. Perfect for judges and evaluators.</div>
                </div>
                <button onclick="this.parentElement.parentElement.remove()" class="text-white/70 hover:text-white text-sm">✕</button>
            </div>
        `;
        
        document.body.appendChild(message);
        
        setTimeout(() => {
            if (document.body.contains(message)) {
                message.style.opacity = '0';
                setTimeout(() => {
                    if (document.body.contains(message)) {
                        document.body.removeChild(message);
                    }
                }, 300);
            }
        }, 5000);
    }
    
    nextSection() {
        if (this.currentSection < this.totalSections - 1) {
            this.transitionToSection(this.currentSection + 1);
        }
    }
    
    previousSection() {
        if (this.currentSection > 0) {
            this.transitionToSection(this.currentSection - 1);
        }
    }
    
    transitionToSection(sectionIndex) {
        if (this.isTransitioning || sectionIndex === this.currentSection) return;
        if (sectionIndex < 0 || sectionIndex >= this.totalSections) return;
        
        this.isTransitioning = true;
        
        // Hide current section
        const currentSectionEl = document.getElementById(this.sections[this.currentSection]);
        if (currentSectionEl) {
            currentSectionEl.style.opacity = '0';
            currentSectionEl.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                currentSectionEl.classList.add('hidden');
            }, 300);
        }
        
        // Show new section
        setTimeout(() => {
            const newSectionEl = document.getElementById(this.sections[sectionIndex]);
            if (newSectionEl) {
                newSectionEl.classList.remove('hidden');
                newSectionEl.style.opacity = '0';
                newSectionEl.style.transform = 'translateY(20px)';
                
                // Trigger reflow
                newSectionEl.offsetHeight;
                
                // Animate in
                newSectionEl.style.opacity = '1';
                newSectionEl.style.transform = 'translateY(0)';
                
                // Scroll to top of section
                newSectionEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            this.currentSection = sectionIndex;
            this.updateProgress();
            this.triggerSectionEvents(sectionIndex);
            
            setTimeout(() => {
                this.isTransitioning = false;
            }, 500);
        }, 300);
    }
    
    triggerSectionEvents(sectionIndex) {
        // Trigger specific events when entering sections
        switch (sectionIndex) {
            case 2: // Network section
                // Initialize network visualization if not already initialized
                if (!window.networkVisualization && document.getElementById('networkVisualization')) {
                    window.networkVisualization = new CommunityNetworkVisualization();
                }
                
                if (window.networkVisualization) {
                    // Animate network after a delay
                    setTimeout(() => {
                        if (window.networkVisualization.animateTrustFlow) {
                            window.networkVisualization.animateTrustFlow();
                        }
                    }, 1500);
                }
                break;
                
            case 3: // Calculator section
                // Initialize calculator if not already initialized
                if (!window.kapwaCalculator && document.getElementById('kapwaCalculator')) {
                    window.kapwaCalculator = new KapwaScoreCalculator();
                }
                
                if (window.kapwaCalculator) {
                    // Auto-load Maria's data
                    setTimeout(() => {
                        if (window.kapwaCalculator.loadPersona) {
                            window.kapwaCalculator.loadPersona('maria_santos_sari_sari');
                        }
                    }, 1000);
                }
                break;
                
            case 4: // BPI section
                // Load BPI integration mockup
                this.loadBPIIntegration();
                break;
        }
    }
    
    updateProgress() {
        // Update progress bar
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            const progress = ((this.currentSection + 1) / this.totalSections) * 100;
            progressBar.style.width = `${progress}%`;
        }
        
        // Update step counter
        const stepNumber = document.getElementById('stepNumber');
        if (stepNumber) {
            stepNumber.textContent = this.currentSection + 1;
        }
        
        // Update current step title
        const currentStep = document.getElementById('currentStep');
        if (currentStep) {
            currentStep.textContent = this.sectionTitles[this.currentSection];
        }
    }
    
    detectCurrentSection() {
        // Auto-detect current section based on scroll position
        if (this.isTransitioning || this.tourActive) return;
        
        const sections = this.sections.map(id => document.getElementById(id));
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        
        for (let i = sections.length - 1; i >= 0; i--) {
            const section = sections[i];
            if (section && !section.classList.contains('hidden')) {
                const rect = section.getBoundingClientRect();
                if (rect.top <= windowHeight * 0.3) {
                    if (this.currentSection !== i) {
                        this.currentSection = i;
                        this.updateProgress();
                    }
                    break;
                }
            }
        }
    }
    
    loadBPIIntegration() {
        const container = document.getElementById('bpiIntegration');
        if (!container) return;
        
        // Create enhanced BPI integration mockup with better contrast and completion
        container.innerHTML = `
            <div class="bg-white rounded-2xl shadow-xl border border-gray-200">
                <!-- Header -->
                <div class="bg-gradient-to-r from-sea-blue to-blue-700 text-white p-6 rounded-t-2xl">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-4">
                            <div class="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-lg">
                                <span class="text-blue-600 font-bold text-2xl">BPI</span>
                            </div>
                            <div>
                                <h3 class="text-2xl font-bold">BPI BanKo Mobile</h3>
                                <p class="text-blue-100 text-sm opacity-90">Enhanced with HARAYA Cultural Intelligence</p>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="bg-flag-yellow text-sea-blue px-4 py-2 rounded-full font-bold text-sm">
                                HARAYA ⚡ Powered
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Mobile App Interface -->
                <div class="p-6">
                    <div class="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl p-6 mb-6 border-2 border-gray-200">
                        <h4 class="text-xl font-bold text-gray-800 mb-4 flex items-center">
                            📱 Enhanced Loan Application Experience
                        </h4>
                        
                        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            <!-- Before HARAYA -->
                            <div class="bg-white rounded-xl p-6 border-2 border-red-200">
                                <h5 class="font-bold text-red-700 mb-4 flex items-center">
                                    ❌ Before HARAYA (Traditional Assessment)
                                </h5>
                                <div class="space-y-3 text-sm">
                                    <div class="flex justify-between items-center py-2 border-b border-red-100">
                                        <span class="text-gray-700">Credit History:</span>
                                        <span class="font-bold text-red-600">INSUFFICIENT</span>
                                    </div>
                                    <div class="flex justify-between items-center py-2 border-b border-red-100">
                                        <span class="text-gray-700">Employment Status:</span>
                                        <span class="font-bold text-red-600">HIGH RISK</span>
                                    </div>
                                    <div class="flex justify-between items-center py-2 border-b border-red-100">
                                        <span class="text-gray-700">Loan Amount:</span>
                                        <span class="font-bold text-red-600">₱15,000 MAX</span>
                                    </div>
                                    <div class="flex justify-between items-center py-2">
                                        <span class="text-gray-700">Processing Time:</span>
                                        <span class="font-bold text-red-600">5-7 DAYS</span>
                                    </div>
                                    <div class="mt-4 p-3 bg-red-50 rounded-lg border border-red-200">
                                        <div class="text-red-800 font-bold text-center">APPLICATION LIKELY REJECTED</div>
                                    </div>
                                </div>
                            </div>

                            <!-- After HARAYA -->
                            <div class="bg-white rounded-xl p-6 border-2 border-green-200">
                                <h5 class="font-bold text-green-700 mb-4 flex items-center">
                                    ✅ With HARAYA (Cultural Intelligence Enhanced)
                                </h5>
                                <div class="space-y-3 text-sm">
                                    <div class="flex justify-between items-center py-2 border-b border-green-100">
                                        <span class="text-gray-700">KapwaScore™:</span>
                                        <span class="font-bold text-green-600">785/1000 - EXCELLENT</span>
                                    </div>
                                    <div class="flex justify-between items-center py-2 border-b border-green-100">
                                        <span class="text-gray-700">Community Trust:</span>
                                        <span class="font-bold text-green-600">25 VOUCHERS</span>
                                    </div>
                                    <div class="flex justify-between items-center py-2 border-b border-green-100">
                                        <span class="text-gray-700">Loan Amount:</span>
                                        <span class="font-bold text-green-600">₱75,000 PRE-APPROVED</span>
                                    </div>
                                    <div class="flex justify-between items-center py-2">
                                        <span class="text-gray-700">Processing Time:</span>
                                        <span class="font-bold text-green-600">SAME DAY</span>
                                    </div>
                                    <div class="mt-4 p-3 bg-green-50 rounded-lg border border-green-200">
                                        <div class="text-green-800 font-bold text-center">PRE-APPROVED - READY TO DISBURSE</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Cultural Intelligence Breakdown -->
                    <div class="bg-gradient-to-br from-blue-50 to-yellow-50 rounded-2xl p-6 border-2 border-blue-200">
                        <h4 class="text-xl font-bold text-gray-800 mb-4 flex items-center">
                            🤝 Cultural Intelligence Assessment
                        </h4>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div class="bg-white rounded-xl p-4 border border-blue-200 text-center">
                                <div class="text-3xl mb-2">👨‍👩‍👧‍👦</div>
                                <div class="font-bold text-gray-800">Kapwa Network</div>
                                <div class="text-2xl font-bold text-blue-600 mt-2">95%</div>
                                <div class="text-sm text-gray-600">Strong family support system</div>
                            </div>
                            <div class="bg-white rounded-xl p-4 border border-green-200 text-center">
                                <div class="text-3xl mb-2">🏘️</div>
                                <div class="font-bold text-gray-800">Bayanihan Spirit</div>
                                <div class="text-2xl font-bold text-green-600 mt-2">89%</div>
                                <div class="text-sm text-gray-600">Active community participation</div>
                            </div>
                            <div class="bg-white rounded-xl p-4 border border-yellow-200 text-center">
                                <div class="text-3xl mb-2">🤝</div>
                                <div class="font-bold text-gray-800">Utang na Loob</div>
                                <div class="text-2xl font-bold text-yellow-600 mt-2">96%</div>
                                <div class="text-sm text-gray-600">Perfect obligation fulfillment</div>
                            </div>
                        </div>
                    </div>

                    <!-- Business Impact Metrics -->
                    <div class="mt-6 bg-gradient-to-br from-gray-800 to-gray-900 text-white rounded-2xl p-6">
                        <h4 class="text-xl font-bold mb-4 text-center">
                            🎯 BPI Partnership Business Impact
                        </h4>
                        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                            <div class="bg-white/10 rounded-xl p-4">
                                <div class="text-3xl font-bold text-flag-yellow">40%</div>
                                <div class="text-sm opacity-80">More Eligible Borrowers</div>
                            </div>
                            <div class="bg-white/10 rounded-xl p-4">
                                <div class="text-3xl font-bold text-flag-yellow">95%</div>
                                <div class="text-sm opacity-80">Faster Approvals</div>
                            </div>
                            <div class="bg-white/10 rounded-xl p-4">
                                <div class="text-3xl font-bold text-flag-yellow">25%</div>
                                <div class="text-sm opacity-80">Lower Default Rate</div>
                            </div>
                            <div class="bg-white/10 rounded-xl p-4">
                                <div class="text-3xl font-bold text-flag-yellow">₱2.5B</div>
                                <div class="text-sm opacity-80">Market Opportunity</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    completeTour() {
        // Show completion message
        this.showCompletionMessage();
        
        // Enable restart option
        document.getElementById('skipTour').style.display = 'none';
        document.getElementById('restartDemo').style.display = 'inline-block';
    }
    
    showCompletionMessage() {
        // Create a temporary success message
        const message = document.createElement('div');
        message.className = 'fixed top-4 right-4 z-50 bg-tropical-green text-white p-4 rounded-lg shadow-lg';
        message.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="text-xl">🎉</span>
                <div>
                    <div class="font-semibold">Demo Completed!</div>
                    <div class="text-sm opacity-90">You've experienced HARAYA's cultural intelligence</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(message);
            }, 300);
        }, 4000);
    }
    
    restartDemo() {
        // Reset to beginning
        this.currentSection = 0;
        this.tourActive = false;
        this.isTransitioning = false;
        
        // Cancel any active tour
        if (this.tour && this.tour.isActive()) {
            this.tour.cancel();
        }
        
        // Reset all section styles
        this.sections.forEach((sectionId, index) => {
            const section = document.getElementById(sectionId);
            if (section) {
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
                
                if (index === 0) {
                    section.classList.remove('hidden');
                } else {
                    section.classList.add('hidden');
                }
            }
        });
        
        // Clear any dynamic content
        const bpiContainer = document.getElementById('bpiIntegration');
        if (bpiContainer) {
            bpiContainer.innerHTML = '<div class="text-center p-8 text-gray-500">BPI integration will load when you reach this section.</div>';
        }
        
        // Reset calculators and visualizations
        if (window.kapwaCalculator) {
            delete window.kapwaCalculator;
        }
        if (window.networkVisualization) {
            delete window.networkVisualization;
        }
        
        // Clear network visualization
        const networkContainer = document.getElementById('networkVisualization');
        if (networkContainer) {
            networkContainer.innerHTML = '';
        }
        
        // Clear calculator
        const calculatorContainer = document.getElementById('kapwaCalculator');
        if (calculatorContainer) {
            calculatorContainer.innerHTML = '';
        }
        
        this.updateProgress();
        
        // Scroll to top smoothly
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        // Reset UI buttons
        const skipTourBtn = document.getElementById('skipTour');
        const restartBtn = document.getElementById('restartDemo');
        
        if (skipTourBtn) skipTourBtn.style.display = 'inline-block';
        if (restartBtn) restartBtn.style.display = 'none';
        
        // Show success message
        this.showRestartMessage();
    }
    
    showRestartMessage() {
        // Create a temporary restart success message
        const message = document.createElement('div');
        message.className = 'fixed top-4 right-4 z-50 bg-sea-blue text-white p-4 rounded-lg shadow-lg';
        message.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="text-xl">🔄</span>
                <div>
                    <div class="font-semibold">Demo Restarted!</div>
                    <div class="text-sm opacity-90">Ready to explore HARAYA again</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => {
                if (document.body.contains(message)) {
                    document.body.removeChild(message);
                }
            }, 300);
        }, 2000);
    }
    
    viewFullPresentation() {
        // Open the original presentation flow
        const presentationUrl = '../html_prototype/kapwa_score_demo/presentation_flow.html';
        window.open(presentationUrl, '_blank', 'width=1200,height=800');
    }
    
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Global app initialization
window.HarayaApp = {
    narrativeFlow: null,
    
    init() {
        this.narrativeFlow = new HarayaNarrativeFlow();
        console.log('HARAYA Cultural Intelligence Platform initialized');
    }
};

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.HarayaApp.init();
    });
} else {
    window.HarayaApp.init();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HarayaNarrativeFlow;
}