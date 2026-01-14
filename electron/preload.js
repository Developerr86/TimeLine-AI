const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Window controls
    minimize: () => ipcRenderer.invoke('window-minimize'),
    maximize: () => ipcRenderer.invoke('window-maximize'),
    close: () => ipcRenderer.invoke('window-close'),

    // Backend URL
    getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),

    // Scenario handling
    confirmScenario: () => ipcRenderer.invoke('confirm-scenario'),
    dismissScenario: () => ipcRenderer.invoke('dismiss-scenario'),
    getPendingScenario: () => ipcRenderer.invoke('get-pending-scenario'),

    // Listen for scenario dialog events from main process
    onShowScenarioDialog: (callback) => {
        ipcRenderer.on('show-scenario-dialog', (event, scenario) => callback(scenario));
    },

    // Platform info
    platform: process.platform
});
