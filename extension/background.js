/**
 * TimeLine Activity Tracker - Background Service Worker
 * 
 * Detects user activity based on active tab URL and reports to backend.
 * Runs on a 3-second interval.
 */

// Configuration
const CONFIG = {
    BACKEND_URL: 'http://localhost:5000/api/heartbeat',
    POLL_INTERVAL_MS: 3000,
    IDLE_DETECTION_SECONDS: 60
};

// Scenario detection patterns
const SCENARIO_PATTERNS = {
    VIDEO: [
        /youtube\.com\/watch/i,
        /vimeo\.com\/\d+/i,
        /netflix\.com\/watch/i,
        /twitch\.tv\//i,
        /dailymotion\.com\/video/i,
        /coursera\.org\/learn\/.*\/lecture/i,
        /udemy\.com\/course\/.*\/learn\/lecture/i,
        /edx\.org\/course\/.*\/video/i,
        /skillshare\.com\/classes\//i,
        /lynda\.com\/.*\/.*-tutorial/i,
        /linkedin\.com\/learning\//i,
        /pluralsight\.com\/course\//i
    ],
    DOC: [
        // PDF URLs (various patterns)
        /\.pdf(\?.*)?$/i,
        /\.pdf\b/i,  // .pdf anywhere in URL
        /\/pdf\//i,  // /pdf/ in path
        /viewer.*\.pdf/i,
        /pdfviewer/i,

        // Chrome PDF viewer
        /chrome-extension:\/\/.*pdf/i,

        // Google services
        /docs\.google\.com\/document/i,
        /docs\.google\.com\/spreadsheets/i,
        /docs\.google\.com\/presentation/i,
        /drive\.google\.com\/file/i,
        /drive\.google\.com\/.*\/view/i,

        // Cloud document services
        /notion\.so\//i,
        /dropbox\.com\/.*(\.pdf|\/view)/i,
        /onedrive\.live\.com/i,
        /sharepoint\.com/i,
        /box\.com\/s\//i,

        // Academic/Research papers
        /arxiv\.org\/(abs|pdf)/i,
        /researchgate\.net\/publication/i,
        /academia\.edu\//i,
        /sciencedirect\.com\/.*\/article/i,
        /ieee\.org\/document/i,
        /springer\.com\/.*\/chapter/i,
        /nature\.com\/articles/i,
        /jstor\.org\//i,
        /scholar\.google\.com/i,

        // Office Online
        /office\.com\/.*\/(word|excel|powerpoint)/i,
        /officeapps\.live\.com/i,

        // Other document platforms
        /scribd\.com\/(doc|document|read)/i,
        /slideshare\.net\//i,
        /issuu\.com\//i,
        /overleaf\.com\/project/i,
        /readthedocs\.io/i,
        /gitbook\.io/i
    ]
};

// State
let isEnabled = true;
let lastScenario = null;
let lastUrl = null;
let pollIntervalId = null;

/**
 * Detect scenario from URL
 * @param {string} url - The URL to analyze
 * @returns {string} - Scenario type: VIDEO, DOC, WEB, or OTHER
 */
function detectScenario(url) {
    if (!url) return 'OTHER';

    // Check for VIDEO patterns
    for (const pattern of SCENARIO_PATTERNS.VIDEO) {
        if (pattern.test(url)) {
            return 'VIDEO';
        }
    }

    // Check for DOC patterns
    for (const pattern of SCENARIO_PATTERNS.DOC) {
        if (pattern.test(url)) {
            return 'DOC';
        }
    }

    // Default to WEB for http/https URLs
    if (url.startsWith('http://') || url.startsWith('https://')) {
        return 'WEB';
    }

    return 'OTHER';
}

/**
 * Send heartbeat to backend
 * @param {Object} data - Heartbeat data
 */
async function sendHeartbeat(data) {
    try {
        const response = await fetch(CONFIG.BACKEND_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            console.warn('[TimeLine] Heartbeat failed:', response.status);
            return null;
        }

        const result = await response.json();

        // Handle any commands from backend
        if (result.command) {
            handleBackendCommand(result.command);
        }

        return result;
    } catch (error) {
        // Backend might not be running - this is expected during development
        console.debug('[TimeLine] Backend unavailable:', error.message);
        return null;
    }
}

/**
 * Handle commands from backend
 * @param {string} command - Command to execute
 */
function handleBackendCommand(command) {
    console.log('[TimeLine] Received command:', command);
    // Future: Handle commands like 'scrape_page', 'get_content', etc.
}

/**
 * Query active tab and send heartbeat
 */
async function pollActiveTab() {
    if (!isEnabled) return;

    try {
        // Check idle state first
        const idleState = await chrome.idle.queryState(CONFIG.IDLE_DETECTION_SECONDS);

        if (idleState === 'locked' || idleState === 'idle') {
            // User is idle/locked - send IDLE heartbeat
            await sendHeartbeat({
                url: null,
                title: null,
                scenario: 'IDLE',
                idle_state: idleState,
                timestamp: Date.now()
            });
            return;
        }

        // Query active tab in current window
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tabs || tabs.length === 0) {
            return;
        }

        const activeTab = tabs[0];
        const url = activeTab.url || '';
        const title = activeTab.title || '';

        // Skip chrome:// and extension pages
        if (url.startsWith('chrome://') || url.startsWith('chrome-extension://')) {
            return;
        }

        // Detect scenario
        const scenario = detectScenario(url);

        // Only send if URL or scenario changed (or periodically for VIDEO)
        const hasChanged = url !== lastUrl || scenario !== lastScenario;
        const isVideo = scenario === 'VIDEO';

        if (hasChanged || isVideo) {
            await sendHeartbeat({
                url: url,
                title: title,
                scenario: scenario,
                timestamp: Date.now()
            });

            lastUrl = url;
            lastScenario = scenario;
        }
    } catch (error) {
        console.error('[TimeLine] Poll error:', error);
    }
}

/**
 * Start the polling loop
 */
function startPolling() {
    if (pollIntervalId) {
        clearInterval(pollIntervalId);
    }

    // Initial poll
    pollActiveTab();

    // Set up interval
    pollIntervalId = setInterval(pollActiveTab, CONFIG.POLL_INTERVAL_MS);
    console.log('[TimeLine] Polling started');
}

/**
 * Stop the polling loop
 */
function stopPolling() {
    if (pollIntervalId) {
        clearInterval(pollIntervalId);
        pollIntervalId = null;
    }
    console.log('[TimeLine] Polling stopped');
}

/**
 * Toggle tracking enabled state
 */
function setEnabled(enabled) {
    isEnabled = enabled;
    if (enabled) {
        startPolling();
    } else {
        stopPolling();
    }
}

// Listen for tab updates (immediate detection)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (!isEnabled) return;

    // Only trigger on complete load of active tab
    if (changeInfo.status === 'complete' && tab.active) {
        pollActiveTab();
    }
});

// Listen for tab activation (switching tabs)
chrome.tabs.onActivated.addListener((activeInfo) => {
    if (!isEnabled) return;
    pollActiveTab();
});

// Listen for window focus changes
chrome.windows.onFocusChanged.addListener((windowId) => {
    if (!isEnabled) return;
    if (windowId !== chrome.windows.WINDOW_ID_NONE) {
        pollActiveTab();
    }
});

// Handle messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'GET_STATUS') {
        sendResponse({
            enabled: isEnabled,
            lastUrl: lastUrl,
            lastScenario: lastScenario
        });
    } else if (message.type === 'SET_ENABLED') {
        setEnabled(message.enabled);
        sendResponse({ success: true });
    }
    return true;
});

// Start polling when service worker loads
startPolling();

console.log('[TimeLine] Activity Tracker initialized');
