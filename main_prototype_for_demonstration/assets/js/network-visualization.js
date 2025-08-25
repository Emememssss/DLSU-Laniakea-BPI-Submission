/**
 * HARAYA Community Network Visualization
 * =====================================
 * 
 * Interactive D3.js-powered visualization of Maria's Bayanihan network
 * Shows trust relationships, community connections, and cultural bonds
 * 
 * Features:
 * - Interactive node dragging and exploration
 * - Trust relationship animations
 * - Cultural context tooltips
 * - Responsive design for different screen sizes
 */

class CommunityNetworkVisualization {
    constructor(containerId = 'networkVisualization') {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.networkData = null;
        this.simulation = null;
        this.svg = null;
        this.width = 0;
        this.height = 0;
        
        // Visual settings
        this.nodeRadius = {
            primary_user: 28,              // Maria - largest
            family_core: 20,               // Close family
            family_extended: 16,           // Extended family
            godparent: 18,                // Godparents - important
            neighbor: 14,                 // Neighbors
            community_leader: 22,          // Leaders - important
            bpi_client: 20,               // BPI clients - important
            high_net_worth: 24,           // High net worth - very important
            business_partner: 16,          // Business partners
            mentor: 18                    // Mentors
        };
        
        this.colors = {
            primary_user: '#C41E3A',           // Maria - BPI Red (central)
            family_core: '#8B0000',            // Close family - Dark red
            family_extended: '#DC143C',        // Extended family - Light red
            godparent: '#800020',              // Godparents - Burgundy
            neighbor: '#C41E3A',               // Neighbors - BPI Red
            community_leader: '#8B0000',       // Leaders - Dark red
            bpi_client: '#DC143C',             // BPI clients - Light red
            high_net_worth: '#800020',         // High net worth - Burgundy
            business_partner: '#C41E3A',       // Business partners - BPI Red
            mentor: '#8B0000'                  // Mentors - Dark red
        };
        
        // Initialize
        this.init();
    }
    
    async init() {
        try {
            await this.loadNetworkData();
            this.setupSVG();
            this.createVisualization();
            this.bindEvents();
        } catch (error) {
            console.error('Error initializing network visualization:', error);
            this.showFallback();
        }
    }
    
    async loadNetworkData() {
        try {
            const response = await fetch('data/maria_network.json');
            this.networkData = await response.json();
            console.log('Network data loaded successfully');
        } catch (error) {
            console.error('Error loading network data:', error);
            // Use fallback data
            this.networkData = this.getFallbackNetworkData();
        }
    }
    
    generateComplexNetwork(mariaNode, totalNodes = 65) {
        const nodes = [mariaNode];
        const links = [];
        const nodeTypes = ["family_core", "family_extended", "godparent", "neighbor", "community_leader", "bpi_client", "high_net_worth", "business_partner", "mentor"];
        const linkStrengths = ["strong", "medium", "weak"];
        
        // Generate additional nodes
        for (let i = 1; i < totalNodes; i++) {
            const type = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];
            const trustBase = type === 'bpi_client' ? 85 : type === 'high_net_worth' ? 80 : type === 'family_core' ? 95 : 60;
            const trust_score = (trustBase + Math.random() * 15) / 100;
            const isBPIClient = type === 'bpi_client' || Math.random() > 0.7;
            const isHNWI = type === 'high_net_worth' || Math.random() > 0.85;
            
            const names = {
                family_core: ['Carlos Santos', 'Ana Santos', 'Pedro Santos', 'Rosa Santos'],
                family_extended: ['Tita Carmen', 'Tito Ben', 'Prima Lisa', 'Kuya Jun'],
                godparent: ['Ninong Miguel', 'Ninang Elena', 'Tito Rey', 'Tita Luz'],
                neighbor: ['Mang Jose', 'Aling Nena', 'Ka Bert', 'Ate Rosa'],
                community_leader: ['Kap. Garcia', 'Barangay Chairman Cruz', 'Pastor David', 'Councilor Santos'],
                bpi_client: ['Mr. Antonio Reyes', 'Mrs. Linda Mercado', 'Dr. Jose Dela Cruz', 'Engr. Maria Fernandez'],
                high_net_worth: ['Don Ricardo Villanueva', 'Mrs. Carmen Lopez-Ayala', 'Architect Paolo Santos', 'CEO Michelle Tan'],
                business_partner: ['Supplier Ramon', 'Partner Joyce', 'Vendor Alex', 'Distributor Nina'],
                mentor: ['Mrs. Sofia Gonzales', 'Coach Roberto', 'Mentor Patricia', 'Guide Fernando']
            };
            
            const nameList = names[type] || [`Person ${i}`];
            const name = nameList[i % nameList.length] || `${type.replace('_', ' ')} ${i}`;
            
            nodes.push({
                id: `${type}_${i}`,
                label: name,
                type: type,
                trust_score: trust_score,
                description: this.getNodeDescription(type, name, isBPIClient, isHNWI),
                isBPIClient: isBPIClient,
                isHNWI: isHNWI,
                yearsKnown: Math.floor(Math.random() * 15) + 1
            });
        }
        
        // Connect Maria to majority of the network
        nodes.slice(1).forEach(node => {
            if (Math.random() > 0.25) { // Connect ~75% of nodes directly to Maria
                const strength = node.type.includes('family') ? 'strong' : 
                               node.type === 'bpi_client' || node.type === 'high_net_worth' ? 'strong' :
                               node.type === 'community_leader' ? 'medium' : 'weak';
                
                links.push({
                    source: mariaNode.id,
                    target: node.id,
                    strength: strength,
                    trust: Math.floor(node.trust_score * 100),
                    yearsKnown: node.yearsKnown || Math.floor(Math.random() * 10) + 1
                });
            }
        });
        
        // Add inter-connections between nodes to form clusters
        for (let i = 1; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const node1 = nodes[i];
                const node2 = nodes[j];
                
                // Higher connection probability for same types or related types
                let connectionChance = 0.08; // Base 8%
                
                if (node1.type === node2.type) connectionChance = 0.25; // Same type: 25%
                if ((node1.type.includes('family') && node2.type.includes('family')) ||
                    (node1.type === 'bpi_client' && node2.type === 'high_net_worth') ||
                    (node1.type === 'community_leader' && node2.type === 'neighbor')) {
                    connectionChance = 0.35; // Related types: 35%
                }
                
                if (Math.random() < connectionChance) {
                    const strength = linkStrengths[Math.floor(Math.random() * linkStrengths.length)];
                    links.push({
                        source: node1.id,
                        target: node2.id,
                        strength: strength,
                        trust: Math.floor(Math.random() * 40) + 40, // 40-80 trust for indirect
                        yearsKnown: Math.floor(Math.random() * 8) + 1
                    });
                }
            }
        }
        
        return { nodes, links };
    }
    
    getNodeDescription(type, name, isBPIClient, isHNWI) {
        const descriptions = {
            family_core: "Close family member, strongest trust bond",
            family_extended: "Extended family, reliable relationship",
            godparent: "Spiritual guide and family advisor",
            neighbor: "Long-time neighbor, community connection",
            community_leader: "Local leader, community influencer",
            bpi_client: "Fellow BPI client, financial trust network",
            high_net_worth: "High net-worth individual, business mentor",
            business_partner: "Business relationship, professional trust",
            mentor: "Personal mentor, guidance provider"
        };
        
        let desc = descriptions[type] || "Community connection";
        if (isBPIClient) desc += " • BPI Client";
        if (isHNWI) desc += " • High Net Worth";
        
        return desc;
    }

    getFallbackNetworkData() {
        const mariaNode = {
            id: "maria_santos",
            label: "Maria Santos",
            type: "primary_user", 
            trust_score: 1.0,
            description: "Sari-sari store owner, 42 years old • BPI Client",
            isBPIClient: true,
            isHNWI: false,
            yearsKnown: 0
        };
        
        return this.generateComplexNetwork(mariaNode, 65);
    }
    
    setupSVG() {
        if (!this.container) {
            console.error('Container not found:', this.containerId);
            return;
        }
        
        // Get container dimensions
        const rect = this.container.getBoundingClientRect();
        this.width = rect.width || 800;
        this.height = rect.height || 600;
        
        // Clear container
        this.container.innerHTML = '';
        
        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('viewBox', `0 0 ${this.width} ${this.height}`)
            .style('background', '#f8fafc')
            .style('border-radius', '12px');
        
        // Add definitions for patterns and gradients
        this.createDefs();
    }
    
    createDefs() {
        const defs = this.svg.append('defs');
        
        // Create gradients for different node types
        Object.entries(this.colors).forEach(([type, color]) => {
            const gradient = defs.append('radialGradient')
                .attr('id', `gradient-${type}`)
                .attr('cx', '30%')
                .attr('cy', '30%');
            
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', d3.color(color).brighter(0.8));
            
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', color);
        });
        
        // Create arrow markers for directed edges
        defs.append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 15)
            .attr('refY', -1.5)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#666');
    }
    
    createVisualization() {
        if (!this.networkData || !this.networkData.nodes) {
            console.error('No network data available');
            return;
        }
        
        // Process the data
        const nodes = this.networkData.nodes.map(d => ({ ...d }));
        const links = (this.networkData.links || this.networkData.edges || []).map(d => ({
            source: d.source || d.from,
            target: d.target || d.to,
            strength: d.strength || 0.5,
            trust: d.trust || Math.floor(Math.random() * 60) + 40,
            relationship: d.relationship || 'connection',
            yearsKnown: d.yearsKnown || Math.floor(Math.random() * 5) + 1,
            ...d
        }));
        
        // Create force simulation optimized for large networks (65+ nodes)
        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(d => this.getLinkDistance(d)).strength(0.1))
            .force('charge', d3.forceManyBody().strength(d => this.getNodeCharge(d)).distanceMax(400))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.getNodeRadius(d) + 8).strength(0.9))
            .force('x', d3.forceX(this.width / 2).strength(0.1))
            .force('y', d3.forceY(this.height / 2).strength(0.1))
            .alpha(0.3)
            .alphaDecay(0.01)
            .velocityDecay(0.4);
        
        // Create link elements
        const link = this.svg.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke', d => this.getLinkColor(d))
            .attr('stroke-width', d => this.getLinkWidth(d))
            .attr('stroke-opacity', 0.8)
            .style('cursor', 'pointer');
        
        // Create node elements
        const node = this.svg.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(nodes)
            .enter().append('g')
            .attr('class', 'node')
            .style('cursor', 'grab')
            .call(this.createDragBehavior());
        
        // Add circles to nodes
        node.append('circle')
            .attr('r', d => this.getNodeRadius(d))
            .attr('fill', d => d.id === 'maria_santos' ? '#22c55e' : `url(#gradient-${d.type})`)
            .attr('stroke', d => d.id === 'maria_santos' ? '#16a34a' : (this.colors[d.type] || '#666'))
            .attr('stroke-width', d => d.type === 'primary_user' ? 3 : 2);
        // node.append('circle')
        //     .attr('r', d => this.getNodeRadius(d))
        //     .attr('fill', d => `url(#gradient-${d.type})`)
        //     .attr('stroke', d => this.colors[d.type] || '#666')
        //     .attr('stroke-width', d => d.type === 'primary_user' ? 3 : 2);
        
        // Add labels to nodes
        node.append('text')
            .text(d => d.label)
            .attr('x', 0)
            .attr('y', d => this.getNodeRadius(d) + 15)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('font-weight', d => d.type === 'primary_user' ? 'bold' : 'normal')
            .attr('fill', '#2d3748');
        
        // Add trust score indicators for important nodes
        node.filter(d => d.trust_score > 0.9)
            .append('circle')
            .attr('r', 4)
            .attr('cx', d => this.getNodeRadius(d) - 8)
            .attr('cy', -8)
            .attr('fill', '#10B981')
            .attr('stroke', 'white')
            .attr('stroke-width', 1);
        
        // Add tooltips
        this.addTooltips(node, link);
        
        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }
    
    getNodeRadius(d) {
        return this.nodeRadius[d.type] || 12;
    }
    
    getNodeCharge(d) {
        const baseCharge = -300;
        const multiplier = d.type === 'primary_user' ? 2 : 1;
        return baseCharge * multiplier;
    }
    
    getLinkDistance(d) {
        const baseDistance = 120; // Increased for larger network
        
        // Calculate distance based on relationship strength
        let strengthValue = 0.5; // Default
        
        if (typeof d.strength === 'string') {
            const strengthMap = { strong: 0.9, medium: 0.6, weak: 0.3 };
            strengthValue = strengthMap[d.strength] || 0.5;
        } else if (typeof d.strength === 'number') {
            strengthValue = d.strength;
        }
        
        // Closer distance for stronger relationships
        const strengthMultiplier = (2 - strengthValue);
        return Math.max(60, baseDistance * strengthMultiplier);
    }
    
    getLinkColor(d) {
        if (d.color) return d.color;
        
        const relationshipColors = {
            'spouse': '#C41E3A',
            'parent_child': '#8B0000',
            'sibling': '#DC143C',
            'business_partnership': '#800020',
            'loyal_customer': '#C41E3A',
            'community_member': '#8B0000',
            'neighbor': '#DC143C',
            'member': '#C41E3A',
            'compadrazgo': '#800020'
        };
        
        return relationshipColors[d.relationship] || '#C41E3A';
    }
    
    getLinkWidth(d) {
        if (d.width) return d.width;
        
        let strengthValue = 0.5; // Default
        
        if (typeof d.strength === 'string') {
            const strengthMap = { strong: 0.9, medium: 0.6, weak: 0.3 };
            strengthValue = strengthMap[d.strength] || 0.5;
        } else if (typeof d.strength === 'number') {
            strengthValue = d.strength;
        }
        
        return Math.max(1, strengthValue * 5);
    }
    
    addTooltips(nodeSelection, linkSelection) {
        // Create tooltip div
        const tooltip = d3.select('body').append('div')
            .attr('class', 'network-tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '10px')
            .style('border-radius', '8px')
            .style('font-size', '12px')
            .style('max-width', '200px')
            .style('z-index', '1000');
        
        // Node tooltips
        nodeSelection
            .on('mouseover', (event, d) => {
                tooltip.style('visibility', 'visible')
                    .html(`
                        <strong>${d.label}</strong><br/>
                        Type: ${this.formatNodeType(d.type)}<br/>
                        Trust Score: ${Math.round((d.trust_score || 0) * 100)}%<br/>
                        <em>${d.description || ''}</em>
                    `);
            })
            .on('mousemove', (event) => {
                tooltip.style('top', (event.pageY - 10) + 'px')
                    .style('left', (event.pageX + 10) + 'px');
            })
            .on('mouseout', () => {
                tooltip.style('visibility', 'hidden');
            });
        
        // Link tooltips
        linkSelection
            .on('mouseover', (event, d) => {
                let strengthPercent = 50; // Default
                if (typeof d.strength === 'string') {
                    const strengthMap = { strong: 90, medium: 60, weak: 30 };
                    strengthPercent = strengthMap[d.strength] || 50;
                } else if (typeof d.strength === 'number') {
                    strengthPercent = Math.round(d.strength * 100);
                }
                
                tooltip.style('visibility', 'visible')
                    .html(`
                        <strong>${this.formatRelationshipType(d.relationship)}</strong><br/>
                        Trust Level: ${d.trust || 'Unknown'}%<br/>
                        Relationship Strength: ${strengthPercent}%<br/>
                        Years Known: ${d.yearsKnown || 'N/A'} years
                    `);
            })
            .on('mousemove', (event) => {
                tooltip.style('top', (event.pageY - 10) + 'px')
                    .style('left', (event.pageX + 10) + 'px');
            })
            .on('mouseout', () => {
                tooltip.style('visibility', 'hidden');
            });
    }
    
    formatNodeType(type) {
        const typeLabels = {
            'primary_user': 'Primary User',
            'family_core': 'Family Core',
            'family_extended': 'Extended Family',
            'godparent': 'Godparent',
            'neighbor': 'Neighbor',
            'community_leader': 'Community Leader',
            'bpi_client': 'BPI Client',
            'high_net_worth': 'High Net Worth Individual',
            'business_partner': 'Business Partner',
            'mentor': 'Mentor',
            'business_supplier': 'Business Supplier',
            'loyal_customer': 'Loyal Customer',
            'financial_institution': 'Financial Partner'
        };
        return typeLabels[type] || type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    formatRelationshipType(relationship) {
        const relationshipLabels = {
            'spouse': 'Spouse',
            'parent_child': 'Parent-Child',
            'sibling': 'Sibling',
            'business_partnership': 'Business Partner',
            'loyal_customer': 'Loyal Customer',
            'community_member': 'Community Member',
            'neighbor': 'Neighbor',
            'member': 'Member',
            'compadrazgo': 'Godparent Bond'
        };
        return relationshipLabels[relationship] || relationship.replace('_', ' ');
    }
    
    createDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    animateTrustFlow() {
        const links = this.svg.selectAll('.links line');
        
        // Create trust flow animation
        links
            .transition()
            .duration(2000)
            .ease(d3.easeLinear)
            .attr('stroke-dasharray', '5,5')
            .attr('stroke-dashoffset', 0)
            .transition()
            .duration(2000)
            .attr('stroke-dashoffset', -10)
            .on('end', function() {
                d3.select(this)
                    .attr('stroke-dasharray', null)
                    .attr('stroke-dashoffset', null);
            });
    }
    
    highlightCommunityType(type) {
        const nodes = this.svg.selectAll('.nodes g');
        const links = this.svg.selectAll('.links line');
        
        // Fade all elements first
        nodes.style('opacity', 0.3);
        links.style('opacity', 0.1);
        
        // Highlight selected type
        nodes.filter(d => d.type === type)
            .style('opacity', 1);
        
        // Highlight related links
        links.filter(d => d.source.type === type || d.target.type === type)
            .style('opacity', 0.8);
    }
    
    resetHighlight() {
        this.svg.selectAll('.nodes g').style('opacity', 1);
        this.svg.selectAll('.links line').style('opacity', 0.8);
    }
    
    showFallback() {
        if (!this.container) return;
        
        this.container.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center p-8">
                    <div class="text-4xl mb-4">🏘️</div>
                    <h3 class="text-lg font-semibold text-sea-blue mb-2">Community Network</h3>
                    <p class="text-rich-brown opacity-75">
                        Maria's trust network includes 25 community connections<br/>
                        spanning family, business partners, and neighbors.
                    </p>
                    <div class="mt-4 space-y-2 text-sm">
                        <div class="flex justify-between items-center">
                            <span>Family Core:</span>
                            <span class="font-semibold text-tropical-green">95% Trust</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span>Business Partners:</span>
                            <span class="font-semibold text-sea-blue">89% Trust</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span>Community Leaders:</span>
                            <span class="font-semibold text-warm-orange">87% Trust</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    bindEvents() {
        // Listen for animation button
        document.addEventListener('click', (e) => {
            if (e.target.id === 'animateNetwork') {
                this.animateTrustFlow();
            }
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }
    
    handleResize() {
        if (!this.container || !this.svg) return;
        
        const rect = this.container.getBoundingClientRect();
        const newWidth = rect.width || 800;
        const newHeight = rect.height || 600;
        
        if (newWidth !== this.width || newHeight !== this.height) {
            this.width = newWidth;
            this.height = newHeight;
            
            this.svg
                .attr('width', this.width)
                .attr('height', this.height)
                .attr('viewBox', `0 0 ${this.width} ${this.height}`);
            
            if (this.simulation) {
                this.simulation
                    .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                    .alpha(0.3)
                    .restart();
            }
        }
    }
}

// Initialize the network visualization
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the network section
    if (document.getElementById('networkVisualization')) {
        window.networkVisualization = new CommunityNetworkVisualization();
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CommunityNetworkVisualization;
}