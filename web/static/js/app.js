/**
 * Ansel Crypto Rank App - Main JavaScript
 */

// Highlight current nav tab based on URL
document.addEventListener('DOMContentLoaded', function() {
    const currentPath = window.location.pathname;
    const navTabs = document.querySelectorAll('.nav-tab');

    navTabs.forEach(tab => {
        const isCurrentPage = tab.getAttribute('href') === currentPath;
        const dropdown = tab.closest('.nav-dropdown');
        const hasCurrentChild = dropdown && Array.from(dropdown.querySelectorAll('.dropdown-item'))
            .some(link => link.getAttribute('href') === currentPath);
        tab.classList.toggle('active', isCurrentPage || !!hasCurrentChild);
        if (isCurrentPage) tab.setAttribute('aria-current', 'page');
        else tab.removeAttribute('aria-current');
    });
});

// Utility functions
const Utils = {
    // API call helper
    async fetchAPI(endpoint, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const response = await fetch(`/api${endpoint}`, {
            ...defaultOptions,
            ...options,
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return response.json();
    },

    // Get API key from localStorage
    getApiKey() {
        return localStorage.getItem('anthropic_api_key') || '';
    },

    // Save API key to localStorage
    saveApiKey(key) {
        localStorage.setItem('anthropic_api_key', key);
    },

    // Format date
    formatDate(date) {
        return new Date(date).toLocaleString();
    },

    // Show loading state
    showLoading(element) {
        element.classList.add('loading');
        element.disabled = true;
    },

    // Hide loading state
    hideLoading(element) {
        element.classList.remove('loading');
        element.disabled = false;
    }
};

// Export for use in other scripts
window.Utils = Utils;
