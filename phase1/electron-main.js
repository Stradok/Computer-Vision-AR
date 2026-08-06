const { app, BrowserWindow } = require('electron');

// Starts the Express server (see server.js) in this same process -- the
// packaged app is just a browser window pointed at its own local server,
// same page as `npm start` serves for the browser-only dev workflow.
require('./server');

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadURL('http://localhost:4173');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
