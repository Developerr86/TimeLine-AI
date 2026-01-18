/**
 * TimeLine Content Script - Video Frame Extraction
 * 
 * Injected into video sites (YouTube, Vimeo, etc.) to extract frames
 * from <video> elements and send them to the background script for
 * forwarding to the backend.
 */

// State
let isCapturing = false;
let captureInterval = null;
let canvas = null;
let ctx = null;
let frameCount = 0;

// Configuration
const CONFIG = {
    CAPTURE_FPS: 1,  // 1 frame per second
    JPEG_QUALITY: 0.8,
    MAX_WIDTH: 1280,  // Max frame width to reduce data size
    VIDEO_SELECTORS: [
        'video',
        '.html5-main-video',  // YouTube
        '.vp-video video',    // Vimeo
        '.player-video video' // Generic players
    ]
};

/**
 * Find the main video element on the page
 */
function findVideoElement() {
    for (const selector of CONFIG.VIDEO_SELECTORS) {
        const video = document.querySelector(selector);
        if (video && video.readyState >= 2) { // HAVE_CURRENT_DATA or higher
            return video;
        }
    }
    return null;
}

/**
 * Initialize canvas for frame extraction
 */
function initCanvas() {
    if (!canvas) {
        canvas = document.createElement('canvas');
        ctx = canvas.getContext('2d');
    }
    return canvas;
}

/**
 * Extract a single frame from the video element
 */
function captureFrame(video) {
    if (!video || video.paused || video.ended) {
        return null;
    }

    initCanvas();

    // Calculate dimensions (scale down if needed)
    let width = video.videoWidth;
    let height = video.videoHeight;

    if (width > CONFIG.MAX_WIDTH) {
        const scale = CONFIG.MAX_WIDTH / width;
        width = CONFIG.MAX_WIDTH;
        height = Math.round(height * scale);
    }

    canvas.width = width;
    canvas.height = height;

    try {
        ctx.drawImage(video, 0, 0, width, height);
        const dataUrl = canvas.toDataURL('image/jpeg', CONFIG.JPEG_QUALITY);
        return dataUrl;
    } catch (error) {
        console.error('[TimeLine] Frame capture error:', error);
        return null;
    }
}

/**
 * Start capturing frames from the video
 */
function startCapture() {
    if (isCapturing) {
        console.log('[TimeLine] Already capturing');
        return;
    }

    const video = findVideoElement();
    if (!video) {
        console.warn('[TimeLine] No video element found');
        chrome.runtime.sendMessage({
            type: 'CAPTURE_ERROR',
            error: 'No video element found on page'
        });
        return;
    }

    isCapturing = true;
    frameCount = 0;
    console.log('[TimeLine] Starting frame capture at', CONFIG.CAPTURE_FPS, 'fps');

    captureInterval = setInterval(() => {
        if (!isCapturing) {
            stopCapture();
            return;
        }

        const currentVideo = findVideoElement();
        if (!currentVideo) {
            console.warn('[TimeLine] Video element lost');
            return;
        }

        // Only capture if video is playing
        if (currentVideo.paused || currentVideo.ended) {
            return;
        }

        const dataUrl = captureFrame(currentVideo);
        if (dataUrl) {
            frameCount++;

            // Send frame to background script
            chrome.runtime.sendMessage({
                type: 'FRAME_CAPTURED',
                data: dataUrl,
                frameNumber: frameCount,
                timestamp: Date.now(),
                videoTime: currentVideo.currentTime
            });

            console.log(`[TimeLine] Frame ${frameCount} captured at ${currentVideo.currentTime.toFixed(1)}s`);
        }
    }, 1000 / CONFIG.CAPTURE_FPS);

    // Notify background script
    chrome.runtime.sendMessage({
        type: 'CAPTURE_STARTED',
        url: window.location.href,
        title: document.title
    });
}

/**
 * Stop capturing frames
 */
function stopCapture() {
    if (!isCapturing) {
        return;
    }

    isCapturing = false;

    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    console.log(`[TimeLine] Stopped capturing after ${frameCount} frames`);

    // Notify background script
    chrome.runtime.sendMessage({
        type: 'CAPTURE_STOPPED',
        frameCount: frameCount
    });

    frameCount = 0;
}

/**
 * Handle messages from background script
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('[TimeLine Content] Received message:', message.type);

    switch (message.type) {
        case 'START_CAPTURE':
            startCapture();
            sendResponse({ success: true, capturing: isCapturing });
            break;

        case 'STOP_CAPTURE':
            stopCapture();
            sendResponse({ success: true, frameCount: frameCount });
            break;

        case 'GET_STATUS':
            const video = findVideoElement();
            sendResponse({
                capturing: isCapturing,
                frameCount: frameCount,
                hasVideo: !!video,
                videoPlaying: video ? !video.paused && !video.ended : false
            });
            break;

        case 'PING':
            sendResponse({ pong: true, url: window.location.href });
            break;

        default:
            sendResponse({ error: 'Unknown message type' });
    }

    return true; // Keep channel open for async response
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (isCapturing) {
        stopCapture();
    }
});

// Notify that content script is ready
console.log('[TimeLine] Content script loaded on:', window.location.href);

// Auto-detect when video starts playing (for auto-start feature if needed)
document.addEventListener('play', (event) => {
    if (event.target.tagName === 'VIDEO') {
        console.log('[TimeLine] Video playback detected');
        chrome.runtime.sendMessage({
            type: 'VIDEO_PLAY_DETECTED',
            url: window.location.href,
            title: document.title
        });
    }
}, true);

document.addEventListener('pause', (event) => {
    if (event.target.tagName === 'VIDEO') {
        console.log('[TimeLine] Video paused');
        chrome.runtime.sendMessage({
            type: 'VIDEO_PAUSE_DETECTED',
            url: window.location.href
        });
    }
}, true);
