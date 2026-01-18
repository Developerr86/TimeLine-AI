const { app, BrowserWindow, ipcMain, Notification } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess = null;
let statusPollingInterval = null;
let lastState = null;

const isDev = !app.isPackaged;
const PYTHON_PORT = 5000;
const BACKEND_URL = `http://localhost:${PYTHON_PORT}`;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#1a1a2e',
      symbolColor: '#ffffff',
      height: 32
    },
    backgroundColor: '#1a1a2e',
    show: false
  });

  // Load the React app
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startPythonBackend() {
  const pythonPath = 'python';
  const scriptPath = isDev
    ? path.join(__dirname, '..', 'main.py')
    : path.join(process.resourcesPath, 'main.py');

  console.log(`Starting Python backend from: ${scriptPath}`);

  pythonProcess = spawn(pythonPath, [scriptPath], {
    cwd: isDev ? path.join(__dirname, '..') : process.resourcesPath,
    stdio: ['pipe', 'pipe', 'pipe']
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data.toString().trim()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonProcess = null;
  });

  pythonProcess.on('error', (error) => {
    console.error('Failed to start Python backend:', error);
  });
}

function stopPythonBackend() {
  if (pythonProcess) {
    console.log('Stopping Python backend...');

    // On Windows, we need to kill the process tree
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
    } else {
      pythonProcess.kill('SIGTERM');
    }

    pythonProcess = null;
  }
}

// Scenario notification handling
async function checkBackendStatus() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/state`);
    if (!response.ok) return;

    const status = await response.json();
    const currentState = status.state;

    // Check if state changed to AWAITING_USER_INPUT
    if (currentState === 'AWAITING_USER_INPUT' && lastState !== 'AWAITING_USER_INPUT') {
      const pendingScenario = status.pending_scenario;
      if (pendingScenario) {
        showScenarioNotification(pendingScenario);
      }
    }

    lastState = currentState;
  } catch (error) {
    // Backend not available, ignore
  }
}

function showScenarioNotification(scenario) {
  const scenarioLabels = {
    'WEB': '🌐 Web Article Detected',
    'DOC': '📄 Document Detected',
    'VIDEO': '🎬 Video Lecture Detected'
  };

  const title = scenarioLabels[scenario.scenario_type] || '📋 Activity Detected';
  const confidence = Math.round(scenario.confidence * 100);

  const notification = new Notification({
    title: title,
    body: `Confidence: ${confidence}%\nClick to capture or dismiss`,
    icon: path.join(__dirname, '../src/assets/icon.png'),
    silent: true,  // Suppress notification sounds
    urgency: 'normal',
    timeoutType: 'never'  // Stay until user interacts
  });

  notification.on('click', () => {
    // Bring window to front and show scenario dialog
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
      mainWindow.webContents.send('show-scenario-dialog', scenario);
    }
  });

  notification.on('close', () => {
    // User dismissed notification - dismiss the scenario
    dismissScenario();
  });

  notification.show();
  console.log(`🔔 Notification shown: ${title}`);
}

async function confirmScenario() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/scenario/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const result = await response.json();
    console.log('Scenario confirmed:', result);
    return result;
  } catch (error) {
    console.error('Failed to confirm scenario:', error);
    return { status: 'error', message: error.message };
  }
}

async function dismissScenario() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/scenario/dismiss`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const result = await response.json();
    console.log('Scenario dismissed:', result);
    return result;
  } catch (error) {
    console.error('Failed to dismiss scenario:', error);
    return { status: 'error', message: error.message };
  }
}

function startStatusPolling() {
  // Poll every 2 seconds
  statusPollingInterval = setInterval(checkBackendStatus, 2000);
  console.log('📡 Status polling started');
}

function stopStatusPolling() {
  if (statusPollingInterval) {
    clearInterval(statusPollingInterval);
    statusPollingInterval = null;
    console.log('📡 Status polling stopped');
  }
}

// App lifecycle
app.whenReady().then(() => {
  // Python backend is NOT auto-started
  // Run it separately with: python main.py
  // startPythonBackend();

  // Create window immediately (no Python startup delay needed)
  createWindow();

  // Start polling for scenario notifications
  startStatusPolling();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopStatusPolling();
  stopPythonBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopStatusPolling();
  stopPythonBackend();
});

// IPC handlers
ipcMain.handle('get-backend-url', () => {
  return BACKEND_URL;
});

ipcMain.handle('window-minimize', () => {
  mainWindow?.minimize();
});

ipcMain.handle('window-maximize', () => {
  if (mainWindow?.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow?.maximize();
  }
});

ipcMain.handle('window-close', () => {
  mainWindow?.close();
});

// Scenario handling from renderer
ipcMain.handle('confirm-scenario', async () => {
  return await confirmScenario();
});

ipcMain.handle('dismiss-scenario', async () => {
  return await dismissScenario();
});

ipcMain.handle('get-pending-scenario', async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/api/state`);
    if (response.ok) {
      const status = await response.json();
      return status.pending_scenario;
    }
  } catch (error) {
    console.error('Failed to get pending scenario:', error);
  }
  return null;
});
