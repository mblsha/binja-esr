// PC-E500 Web Emulator Frontend

const API_BASE = '/api/v1';
const POLL_INTERVAL = 100; // 100ms = 10fps

let pollTimer = null;
let isRunning = false;

// PC-E500 Keyboard Layout
// Based on the physical keyboard layout of the Sharp PC-E500
const KEYBOARD_LAYOUT = [
    // Function keys row
    [
        { key: 'KEY_CALC', label: 'CALC', class: 'special' },
        { key: 'KEY_BASIC', label: 'BASIC', class: 'special' },
        { key: 'KEY_F1', label: 'F1' },
        { key: 'KEY_F2', label: 'F2' },
        { key: 'KEY_F3', label: 'F3' },
        { key: 'KEY_F4', label: 'F4' },
        { key: 'KEY_F5', label: 'F5' },
        { key: 'KEY_F6', label: 'F6' },
        { key: 'KEY_SHIFT', label: 'SHIFT', class: 'special wide' },
        { key: 'KEY_ON', label: 'ON', class: 'special' }
    ],
    // Numbers row
    [
        { key: 'KEY_1', label: '1' },
        { key: 'KEY_2', label: '2' },
        { key: 'KEY_3', label: '3' },
        { key: 'KEY_4', label: '4' },
        { key: 'KEY_5', label: '5' },
        { key: 'KEY_6', label: '6' },
        { key: 'KEY_7', label: '7' },
        { key: 'KEY_8', label: '8' },
        { key: 'KEY_9', label: '9' },
        { key: 'KEY_0', label: '0' },
        { key: 'KEY_MINUS', label: '-' },
        { key: 'KEY_EQUALS', label: '=' },
        { key: 'KEY_BACKSPACE', label: '←', class: 'wide' }
    ],
    // QWERTY row 1
    [
        { key: 'KEY_TAB', label: 'TAB', class: 'wide' },
        { key: 'KEY_Q', label: 'Q' },
        { key: 'KEY_W', label: 'W' },
        { key: 'KEY_E', label: 'E' },
        { key: 'KEY_R', label: 'R' },
        { key: 'KEY_T', label: 'T' },
        { key: 'KEY_Y', label: 'Y' },
        { key: 'KEY_U', label: 'U' },
        { key: 'KEY_I', label: 'I' },
        { key: 'KEY_O', label: 'O' },
        { key: 'KEY_P', label: 'P' },
        { key: 'KEY_LBRACKET', label: '[' },
        { key: 'KEY_RBRACKET', label: ']' }
    ],
    // QWERTY row 2
    [
        { key: 'KEY_CAPS', label: 'CAPS', class: 'wide' },
        { key: 'KEY_A', label: 'A' },
        { key: 'KEY_S', label: 'S' },
        { key: 'KEY_D', label: 'D' },
        { key: 'KEY_F', label: 'F' },
        { key: 'KEY_G', label: 'G' },
        { key: 'KEY_H', label: 'H' },
        { key: 'KEY_J', label: 'J' },
        { key: 'KEY_K', label: 'K' },
        { key: 'KEY_L', label: 'L' },
        { key: 'KEY_SEMICOLON', label: ';' },
        { key: 'KEY_QUOTE', label: "'" },
        { key: 'KEY_ENTER', label: 'ENTER', class: 'wide special' }
    ],
    // QWERTY row 3
    [
        { key: 'KEY_Z', label: 'Z' },
        { key: 'KEY_X', label: 'X' },
        { key: 'KEY_C', label: 'C' },
        { key: 'KEY_V', label: 'V' },
        { key: 'KEY_B', label: 'B' },
        { key: 'KEY_N', label: 'N' },
        { key: 'KEY_M', label: 'M' },
        { key: 'KEY_COMMA', label: ',' },
        { key: 'KEY_PERIOD', label: '.' },
        { key: 'KEY_SLASH', label: '/' },
        { key: 'KEY_UP', label: '↑', class: 'special' },
        { key: 'KEY_DOWN', label: '↓', class: 'special' }
    ],
    // Space bar row
    [
        { key: 'KEY_CTRL', label: 'CTRL', class: 'wide' },
        { key: 'KEY_SPACE', label: 'SPACE', class: 'space' },
        { key: 'KEY_LEFT', label: '←', class: 'special' },
        { key: 'KEY_RIGHT', label: '→', class: 'special' },
        { key: 'KEY_INS', label: 'INS' },
        { key: 'KEY_DEL', label: 'DEL' }
    ]
];

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    setupKeyboard();
    setupControls();
    startPolling();
});

// Set up the virtual keyboard
function setupKeyboard() {
    const keyboardContainer = document.getElementById('virtual-keyboard');
    
    KEYBOARD_LAYOUT.forEach(row => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'keyboard-row';
        
        row.forEach(keyDef => {
            const keyButton = document.createElement('button');
            keyButton.className = 'key ' + (keyDef.class || '');
            keyButton.textContent = keyDef.label;
            keyButton.dataset.keyCode = keyDef.key;
            
            // Handle key press
            keyButton.addEventListener('mousedown', () => handleKeyPress(keyDef.key));
            keyButton.addEventListener('mouseup', () => handleKeyRelease(keyDef.key));
            keyButton.addEventListener('mouseleave', () => handleKeyRelease(keyDef.key));
            
            // Prevent focus on click
            keyButton.addEventListener('click', (e) => e.preventDefault());
            
            rowDiv.appendChild(keyButton);
        });
        
        keyboardContainer.appendChild(rowDiv);
    });
}

// Set up control buttons
function setupControls() {
    document.getElementById('btn-run').addEventListener('click', handleRun);
    document.getElementById('btn-pause').addEventListener('click', handlePause);
    document.getElementById('btn-step').addEventListener('click', handleStep);
    document.getElementById('btn-reset').addEventListener('click', handleReset);
}

// Control button handlers
async function handleRun() {
    try {
        const response = await fetch(`${API_BASE}/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'run' })
        });
        
        if (response.ok) {
            isRunning = true;
            updateControlButtons();
        }
    } catch (error) {
        console.error('Error starting emulator:', error);
    }
}

async function handlePause() {
    try {
        const response = await fetch(`${API_BASE}/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'pause' })
        });
        
        if (response.ok) {
            isRunning = false;
            updateControlButtons();
        }
    } catch (error) {
        console.error('Error pausing emulator:', error);
    }
}

async function handleStep() {
    try {
        await fetch(`${API_BASE}/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'step' })
        });
    } catch (error) {
        console.error('Error stepping emulator:', error);
    }
}

async function handleReset() {
    try {
        const response = await fetch(`${API_BASE}/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'reset' })
        });
        
        if (response.ok) {
            isRunning = false;
            updateControlButtons();
        }
    } catch (error) {
        console.error('Error resetting emulator:', error);
    }
}

// Update control button states
function updateControlButtons() {
    document.getElementById('btn-run').disabled = isRunning;
    document.getElementById('btn-pause').disabled = !isRunning;
    document.getElementById('btn-step').disabled = isRunning;
}

// Handle keyboard input
async function handleKeyPress(keyCode) {
    const keyElement = document.querySelector(`[data-key-code="${keyCode}"]`);
    if (keyElement) {
        keyElement.classList.add('pressed');
    }
    
    try {
        await fetch(`${API_BASE}/key`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                key_code: keyCode,
                action: 'press'
            })
        });
    } catch (error) {
        console.error('Error sending key press:', error);
    }
}

async function handleKeyRelease(keyCode) {
    const keyElement = document.querySelector(`[data-key-code="${keyCode}"]`);
    if (keyElement) {
        keyElement.classList.remove('pressed');
    }
    
    try {
        await fetch(`${API_BASE}/key`, {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                key_code: keyCode,
                action: 'release'
            })
        });
    } catch (error) {
        console.error('Error sending key release:', error);
    }
}

// Start polling for state updates
function startPolling() {
    pollTimer = setInterval(updateState, POLL_INTERVAL);
    // Initial update
    updateState();
}

// Stop polling
function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

// Update emulator state from API
async function updateState() {
    try {
        const response = await fetch(`${API_BASE}/state`);
        if (!response.ok) return;
        
        const state = await response.json();
        
        // Update screen
        if (state.screen) {
            document.getElementById('lcd-display').src = state.screen;
        }
        
        // Update registers
        if (state.registers) {
            document.getElementById('reg-pc').textContent = `0x${state.registers.pc.toString(16).padStart(6, '0').toUpperCase()}`;
            document.getElementById('reg-a').textContent = `0x${state.registers.a.toString(16).padStart(2, '0').toUpperCase()}`;
            document.getElementById('reg-b').textContent = `0x${state.registers.b.toString(16).padStart(2, '0').toUpperCase()}`;
            document.getElementById('reg-ba').textContent = `0x${state.registers.ba.toString(16).padStart(4, '0').toUpperCase()}`;
            document.getElementById('reg-i').textContent = `0x${state.registers.i.toString(16).padStart(4, '0').toUpperCase()}`;
            document.getElementById('reg-x').textContent = `0x${state.registers.x.toString(16).padStart(6, '0').toUpperCase()}`;
            document.getElementById('reg-y').textContent = `0x${state.registers.y.toString(16).padStart(6, '0').toUpperCase()}`;
            document.getElementById('reg-u').textContent = `0x${state.registers.u.toString(16).padStart(6, '0').toUpperCase()}`;
            document.getElementById('reg-s').textContent = `0x${state.registers.s.toString(16).padStart(6, '0').toUpperCase()}`;
        }
        
        // Update flags
        if (state.flags) {
            document.getElementById('flag-z').textContent = state.flags.z;
            document.getElementById('flag-c').textContent = state.flags.c;
        }
        
        // Update instruction count
        if (state.instruction_count !== undefined) {
            document.getElementById('instruction-count').textContent = state.instruction_count.toLocaleString();
        }
        
        // Update running state
        isRunning = state.is_running || false;
        updateControlButtons();
        
    } catch (error) {
        console.error('Error fetching state:', error);
    }
}

// Handle physical keyboard mapping (optional enhancement)
document.addEventListener('keydown', (e) => {
    // Map physical keys to virtual keys
    const keyMap = {
        'Enter': 'KEY_ENTER',
        'Backspace': 'KEY_BACKSPACE',
        'Tab': 'KEY_TAB',
        'Escape': 'KEY_ON',
        ' ': 'KEY_SPACE',
        'ArrowUp': 'KEY_UP',
        'ArrowDown': 'KEY_DOWN',
        'ArrowLeft': 'KEY_LEFT',
        'ArrowRight': 'KEY_RIGHT',
        'Delete': 'KEY_DEL',
        'Insert': 'KEY_INS',
        // Add letter keys
        ...Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ').reduce((acc, letter) => {
            acc[letter] = `KEY_${letter}`;
            acc[letter.toLowerCase()] = `KEY_${letter}`;
            return acc;
        }, {}),
        // Add number keys
        ...'0123456789'.split('').reduce((acc, num) => {
            acc[num] = `KEY_${num}`;
            return acc;
        }, {})
    };
    
    const virtualKey = keyMap[e.key];
    if (virtualKey) {
        e.preventDefault();
        handleKeyPress(virtualKey);
    }
});

document.addEventListener('keyup', (e) => {
    // Map physical keys to virtual keys (same mapping as keydown)
    const keyMap = {
        'Enter': 'KEY_ENTER',
        'Backspace': 'KEY_BACKSPACE',
        'Tab': 'KEY_TAB',
        'Escape': 'KEY_ON',
        ' ': 'KEY_SPACE',
        'ArrowUp': 'KEY_UP',
        'ArrowDown': 'KEY_DOWN',
        'ArrowLeft': 'KEY_LEFT',
        'ArrowRight': 'KEY_RIGHT',
        'Delete': 'KEY_DEL',
        'Insert': 'KEY_INS',
        // Add letter keys
        ...Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ').reduce((acc, letter) => {
            acc[letter] = `KEY_${letter}`;
            acc[letter.toLowerCase()] = `KEY_${letter}`;
            return acc;
        }, {}),
        // Add number keys
        ...'0123456789'.split('').reduce((acc, num) => {
            acc[num] = `KEY_${num}`;
            return acc;
        }, {})
    };
    
    const virtualKey = keyMap[e.key];
    if (virtualKey) {
        e.preventDefault();
        handleKeyRelease(virtualKey);
    }
});