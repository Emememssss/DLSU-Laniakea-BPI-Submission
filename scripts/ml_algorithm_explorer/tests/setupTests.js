/**
 * Jest setup file for HARAYA ML Algorithm Explorer tests
 */

import '@testing-library/jest-dom';

// Mock global objects that would be available in browser
global.d3 = {
    select: jest.fn(() => ({
        html: jest.fn(() => ({ append: jest.fn() })),
        append: jest.fn(() => ({ 
            attr: jest.fn(() => ({ attr: jest.fn(), style: jest.fn(), text: jest.fn() }))
        })),
        selectAll: jest.fn(() => ({ data: jest.fn(() => ({ enter: jest.fn() })) }))
    })),
    scaleLinear: jest.fn(() => ({
        domain: jest.fn(() => ({ range: jest.fn() })),
        range: jest.fn()
    })),
    scaleOrdinal: jest.fn(),
    schemeCategory10: ['#1f77b4', '#ff7f0e', '#2ca02c'],
    axisBottom: jest.fn(),
    axisLeft: jest.fn()
};

global.Chart = jest.fn().mockImplementation(() => ({
    update: jest.fn(),
    destroy: jest.fn(),
    data: { labels: [], datasets: [] }
}));

global.tf = {
    sequential: jest.fn(() => ({
        compile: jest.fn(),
        fit: jest.fn(() => Promise.resolve({ history: { loss: [0.5], acc: [0.8] } })),
        predict: jest.fn(() => ({ data: jest.fn(() => Promise.resolve([0.85])) })),
        dispose: jest.fn(),
        countParams: jest.fn(() => 1024),
        summary: jest.fn()
    })),
    layers: {
        dense: jest.fn(),
        dropout: jest.fn()
    },
    train: {
        adam: jest.fn()
    },
    tensor2d: jest.fn(() => ({
        slice: jest.fn(() => ({ dispose: jest.fn() })),
        dispose: jest.fn(),
        data: jest.fn(() => Promise.resolve([0.1, 0.2, 0.3]))
    })),
    grad: jest.fn()
};

// Mock DOM elements
global.document = {
    ...global.document,
    getElementById: jest.fn((id) => {
        const mockElements = {
            'toastContainer': { appendChild: jest.fn() },
            'loadingOverlay': { classList: { add: jest.fn(), remove: jest.fn() } },
            'trainingProgress': { style: { width: '0%' } },
            'kmeansCanvas': { 
                innerHTML: '',
                appendChild: jest.fn()
            }
        };
        return mockElements[id] || { 
            textContent: '',
            style: {},
            classList: { add: jest.fn(), remove: jest.fn(), contains: jest.fn() },
            addEventListener: jest.fn(),
            appendChild: jest.fn()
        };
    }),
    createElement: jest.fn(() => ({
        id: '',
        className: '',
        innerHTML: '',
        appendChild: jest.fn(),
        setAttribute: jest.fn(),
        style: {},
        classList: { add: jest.fn(), remove: jest.fn() }
    })),
    querySelector: jest.fn(),
    querySelectorAll: jest.fn(() => []),
    addEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
    body: { appendChild: jest.fn() }
};

global.window = {
    ...global.window,
    requestAnimationFrame: jest.fn((cb) => setTimeout(cb, 16)),
    localStorage: {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn()
    }
};

// Mock console methods to reduce test noise
global.console = {
    ...console,
    log: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
};

// Setup global test helpers
global.createMockDataset = () => [
    {
        persona_id: 'test_1',
        name: 'Test Person 1',
        kapwa_score: 0.8,
        bayanihan_participation: 0.7,
        utang_na_loob_integrity: 0.9,
        monthly_income: 25000,
        trustworthiness_label: 'trustworthy'
    },
    {
        persona_id: 'test_2', 
        name: 'Test Person 2',
        kapwa_score: 0.6,
        bayanihan_participation: 0.5,
        utang_na_loob_integrity: 0.7,
        monthly_income: 35000,
        trustworthiness_label: 'untrustworthy'
    }
];

global.createMockMLState = () => ({
    initialized: false,
    activeAlgorithms: new Set(),
    sharedData: global.createMockDataset(),
    models: {},
    charts: {},
    status: {
        dataLoaded: true,
        modelsReady: false,
        trainingInProgress: false
    }
});