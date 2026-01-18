/**
 * TimeLine Activity Tracker - Popup Script
 */

const statusText = document.getElementById('status-text');
const scenarioBadge = document.getElementById('scenario-badge');
const currentUrl = document.getElementById('current-url');
const toggleEnabled = document.getElementById('toggle-enabled');

/**
 * Update UI with current status
 */
function updateStatus(status) {
    statusText.textContent = status.enabled ? 'Active' : 'Paused';
    statusText.style.color = status.enabled ? '#2ecc71' : '#e74c3c';

    toggleEnabled.checked = status.enabled;

    if (status.lastScenario) {
        scenarioBadge.textContent = status.lastScenario;
        scenarioBadge.className = `scenario-badge scenario-${status.lastScenario}`;
    } else {
        scenarioBadge.textContent = '--';
        scenarioBadge.className = 'scenario-badge scenario-OTHER';
    }

    if (status.lastUrl) {
        // Truncate long URLs
        const displayUrl = status.lastUrl.length > 60
            ? status.lastUrl.substring(0, 60) + '...'
            : status.lastUrl;
        currentUrl.textContent = displayUrl;
    } else {
        currentUrl.textContent = '--';
    }
}

/**
 * Fetch status from background script
 */
function fetchStatus() {
    chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
        if (response) {
            updateStatus(response);
        }
    });
}

/**
 * Handle toggle change
 */
toggleEnabled.addEventListener('change', () => {
    chrome.runtime.sendMessage({
        type: 'SET_ENABLED',
        enabled: toggleEnabled.checked
    }, (response) => {
        if (response && response.success) {
            fetchStatus();
        }
    });
});

// Initial fetch
fetchStatus();

// Refresh every second while popup is open
setInterval(fetchStatus, 1000);
