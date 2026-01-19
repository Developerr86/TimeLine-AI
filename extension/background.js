/**
 * TimeLine Activity Tracker - Background Service Worker
 * 
 * Detects user activity based on active tab URL and reports to backend.
 * Also handles video frame capture forwarding to backend.
 * Runs on a 3-second interval.
 */

// Configuration
const CONFIG = {
    BACKEND_URL: 'http://localhost:5000/api/heartbeat',
    FRAME_INGEST_URL: 'http://localhost:5000/api/ingest/frame',
    POLL_INTERVAL_MS: 3000,
    IDLE_DETECTION_SECONDS: 60
};

// Frame capture state
let isFrameCapturing = false;
let captureTabId = null;
let framesSent = 0;

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

    switch (command) {
        case 'START_CAPTURE':
            startFrameCapture();
            break;
        case 'STOP_CAPTURE':
            stopFrameCapture();
            break;
        default:
            console.log('[TimeLine] Unknown command:', command);
    }
}

/**
 * Start frame capture on current video tab
 */
async function startFrameCapture() {
    // Prevent duplicate start commands
    if (isFrameCapturing) {
        return;
    }

    try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tabs || tabs.length === 0) {
            console.warn('[TimeLine] No active tab for frame capture');
            return;
        }

        captureTabId = tabs[0].id;
        isFrameCapturing = true;
        framesSent = 0;

        console.log('[TimeLine] Starting frame capture on tab:', captureTabId);

        // Send message to content script to start capturing
        chrome.tabs.sendMessage(captureTabId, { type: 'START_CAPTURE' }, (response) => {
            if (chrome.runtime.lastError) {
                console.error('[TimeLine] Failed to start capture:', chrome.runtime.lastError.message);
                isFrameCapturing = false;
            } else {
                console.log('[TimeLine] Frame capture started successfully');
            }
        });
    } catch (error) {
        console.error('[TimeLine] Error starting frame capture:', error);
    }
}

/**
 * Stop frame capture
 */
function stopFrameCapture() {
    if (!isFrameCapturing || !captureTabId) {
        return;
    }

    chrome.tabs.sendMessage(captureTabId, { type: 'STOP_CAPTURE' }, (response) => {
        if (chrome.runtime.lastError) {
            console.error('[TimeLine] Failed to stop capture:', chrome.runtime.lastError.message);
        } else {
            console.log('[TimeLine] Frame capture stopped, total frames sent:', framesSent);
        }
    });

    isFrameCapturing = false;
    captureTabId = null;
}

/**
 * Send a captured frame to the backend
 * @param {Object} frameData - Frame data from content script
 */
async function sendFrameToBackend(frameData) {
    if (!frameData || !frameData.data) {
        return;
    }

    try {
        const response = await fetch(CONFIG.FRAME_INGEST_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frame_data: frameData.data,
                frame_number: frameData.frameNumber,
                video_time: frameData.videoTime,
                timestamp: frameData.timestamp
            })
        });

        if (!response.ok) {
            console.warn('[TimeLine] Frame upload failed:', response.status);
            return;
        }

        framesSent++;
        const result = await response.json();

        if (result.saved) {
            console.log(`[TimeLine] Frame ${framesSent} saved (similarity: ${result.similarity?.toFixed(2) || 'N/A'})`);
        } else {
            console.log(`[TimeLine] Frame ${framesSent} skipped (similarity: ${result.similarity?.toFixed(2) || 'N/A'})`);
        }
    } catch (error) {
        console.debug('[TimeLine] Frame upload error:', error.message);
    }
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

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
        case 'GET_STATUS':
            sendResponse({
                enabled: isEnabled,
                lastUrl: lastUrl,
                lastScenario: lastScenario,
                isCapturing: isFrameCapturing,
                framesSent: framesSent
            });
            break;

        case 'SET_ENABLED':
            setEnabled(message.enabled);
            sendResponse({ success: true });
            break;

        // Frame capture commands from popup
        case 'START_FRAME_CAPTURE':
            startFrameCapture();
            sendResponse({ success: true });
            break;

        case 'STOP_FRAME_CAPTURE':
            stopFrameCapture();
            sendResponse({ success: true, framesSent: framesSent });
            break;

        case 'GET_CAPTURE_STATUS':
            sendResponse({
                isCapturing: isFrameCapturing,
                captureTabId: captureTabId,
                framesSent: framesSent
            });
            break;

        // Messages from content script
        case 'FRAME_CAPTURED':
            sendFrameToBackend(message);
            sendResponse({ received: true });
            break;

        case 'CAPTURE_STARTED':
            console.log('[TimeLine] Content script started capture:', message.url);
            sendResponse({ acknowledged: true });
            break;

        case 'CAPTURE_STOPPED':
            console.log('[TimeLine] Content script stopped capture, frames:', message.frameCount);
            isFrameCapturing = false;
            sendResponse({ acknowledged: true });
            break;

        case 'CAPTURE_ERROR':
            console.error('[TimeLine] Capture error:', message.error);
            isFrameCapturing = false;
            sendResponse({ acknowledged: true });
            break;

        case 'VIDEO_PLAY_DETECTED':
            console.log('[TimeLine] Video playback detected:', message.title);
            sendResponse({ acknowledged: true });
            break;

        case 'VIDEO_PAUSE_DETECTED':
            console.log('[TimeLine] Video paused:', message.url);
            sendResponse({ acknowledged: true });
            break;

        default:
            sendResponse({ error: 'Unknown message type' });
    }

    return true; // Keep channel open for async response
});

// Start polling when service worker loads
startPolling();

console.log('[TimeLine] Activity Tracker initialized');
