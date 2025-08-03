// PC-E500 Web Emulator Frontend

const API_BASE = '/api/v1';
const POLL_INTERVAL = 100; // 100ms = 10fps

// PC Address Component - Creates a clickable PC address element
class PCAddress {
    constructor(address, options = {}) {
        this.address = address;
        this.className = options.className || 'pc-address';
        this.showTooltip = options.showTooltip !== false;
    }
    
    render() {
        const span = document.createElement('span');
        span.className = this.className;
        span.textContent = this.address;
        
        if (this.showTooltip) {
            span.title = 'Click to copy';
        }
        
        // Click to copy functionality
        span.addEventListener('click', async (e) => {
            e.stopPropagation();
            
            try {
                await navigator.clipboard.writeText(this.address);
                
                // Visual feedback
                span.classList.add('copied');
                const originalText = span.textContent;
                span.textContent = 'Copied!';
                
                setTimeout(() => {
                    span.textContent = originalText;
                    span.classList.remove('copied');
                }, 1000);
                
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        });
        
        return span;
    }
    
    static create(address, options = {}) {
        const component = new PCAddress(address, options);
        return component.render();
    }
}

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
    setupLCDInteraction();
    startPolling();
    startRegisterWatchPolling();
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
            // Use PCAddress component for PC register
            const pcElement = document.getElementById('reg-pc');
            const pcValue = `0x${state.registers.pc.toString(16).padStart(6, '0').toUpperCase()}`;
            if (pcElement.textContent !== pcValue) {
                pcElement.innerHTML = '';
                pcElement.appendChild(PCAddress.create(pcValue, { className: 'pc-address reg-value' }));
            }
            
            // Regular registers
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
        
        // Update instruction history
        if (state.instruction_history) {
            updateInstructionHistory(state.instruction_history);
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

// Update instruction history display
function updateInstructionHistory(history) {
    const historyContainer = document.getElementById('instruction-history');
    
    // Clear existing content
    historyContainer.innerHTML = '';
    
    // Display in reverse order (most recent first)
    const reversedHistory = history.slice().reverse();
    
    reversedHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        // Use PCAddress component
        const pcElement = PCAddress.create(item.pc, { className: 'pc-address history-pc' });
        
        const instrSpan = document.createElement('span');
        instrSpan.className = 'history-instr';
        instrSpan.textContent = item.disassembly;
        
        historyItem.appendChild(pcElement);
        historyItem.appendChild(instrSpan);
        historyContainer.appendChild(historyItem);
    });
}

// Set up LCD interaction
function setupLCDInteraction() {
    const lcdDisplay = document.getElementById('lcd-display');
    const screenFrame = document.querySelector('.screen-frame');
    let tooltip = null;
    let columnHighlight = null;
    
    // Create tooltip element
    tooltip = document.createElement('div');
    tooltip.className = 'lcd-tooltip';
    tooltip.style.display = 'none';
    document.body.appendChild(tooltip);
    
    // Create column highlight element
    columnHighlight = document.createElement('div');
    columnHighlight.className = 'lcd-column-highlight';
    columnHighlight.style.display = 'none';
    screenFrame.appendChild(columnHighlight);
    
    lcdDisplay.addEventListener('mousemove', async (e) => {
        // Get mouse position relative to the image
        const rect = lcdDisplay.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Convert to actual pixel coordinates (accounting for 2x zoom)
        const pixelX = Math.floor(x / 2);
        const pixelY = Math.floor(y / 2);
        
        // Ensure coordinates are in bounds
        if (pixelX >= 0 && pixelX < 240 && pixelY >= 0 && pixelY < 32) {
            try {
                // Fetch PC source for this pixel
                const response = await fetch(`${API_BASE}/lcd_debug?x=${pixelX}&y=${pixelY}`);
                const data = await response.json();
                
                if (data.pc) {
                    // Show tooltip with clickable PC address
                    tooltip.innerHTML = '';
                    const label = document.createElement('span');
                    label.textContent = 'PC: ';
                    tooltip.appendChild(label);
                    
                    const pcElement = PCAddress.create(data.pc, { 
                        className: 'pc-address lcd-pc',
                        showTooltip: false  // Don't show nested tooltip
                    });
                    tooltip.appendChild(pcElement);
                    
                    tooltip.style.display = 'block';
                    tooltip.style.left = `${e.pageX + 10}px`;
                    tooltip.style.top = `${e.pageY - 30}px`;
                    
                    // Highlight the column (1x8 pixels)
                    const columnX = Math.floor(pixelX) * 2; // Convert back to display coordinates
                    columnHighlight.style.display = 'block';
                    columnHighlight.style.left = `${columnX}px`;
                } else {
                    tooltip.style.display = 'none';
                    columnHighlight.style.display = 'none';
                }
            } catch (error) {
                console.error('Error fetching LCD debug info:', error);
                tooltip.style.display = 'none';
                columnHighlight.style.display = 'none';
            }
        } else {
            tooltip.style.display = 'none';
            columnHighlight.style.display = 'none';
        }
    });
    
    lcdDisplay.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
        columnHighlight.style.display = 'none';
    });
}

// Start polling for register watch updates
let registerWatchTimer = null;
function startRegisterWatchPolling() {
    // Poll less frequently than main state (every 500ms)
    registerWatchTimer = setInterval(updateRegisterWatch, 500);
    // Initial update
    updateRegisterWatch();
}

// Register descriptions for tooltips
const REGISTER_DESCRIPTIONS = {
    // RAM Pointers
    'BP': 'RAM Base Pointer - Base address for (BP) addressing mode',
    'PX': 'RAM PX Pointer - X pointer for indexed addressing',
    'PY': 'RAM PY Pointer - Y pointer for indexed addressing',
    'AMC': 'Address Modify Control - Virtually joins CE0/CE1 RAM regions',
    
    // Keyboard I/O
    'KOL': 'Key Output Low (KO0-KO7) - Controls keyboard matrix output pins',
    'KOH': 'Key Output High (KO8-KO15) - Controls keyboard matrix output pins',
    'KIL': 'Key Input (KI0-KI7) - Reads keyboard matrix input pins',
    
    // E Port I/O
    'EOL': 'E Port Output Low (E0-E7) - General purpose I/O output',
    'EOH': 'E Port Output High (E8-E15) - General purpose I/O output',
    'EIL': 'E Port Input Low (E0-E7) - General purpose I/O input',
    'EIH': 'E Port Input High (E8-E15) - General purpose I/O input',
    
    // UART
    'UCR': 'UART Control Register - Baud rate, parity, data/stop bits',
    'USR': 'UART Status Register - RX/TX ready, error flags',
    'RXD': 'UART Receive Buffer - Last received character',
    'TXD': 'UART Transmit Buffer - Character to transmit',
    
    // Interrupts
    'IMR': 'Interrupt Mask Register - Enable/disable interrupt sources',
    'ISR': 'Interrupt Status Register - Pending interrupt flags',
    
    // System Control
    'SCR': 'System Control Register - Timer control, buzzer, display',
    'LCC': 'LCD Contrast Control - Display contrast and key strobe',
    'SSR': 'System Status Register - ON key, reset flag, test inputs'
};

// Update register watch display
async function updateRegisterWatch() {
    try {
        const response = await fetch(`${API_BASE}/imem_watch`);
        if (!response.ok) return;
        
        const data = await response.json();
        const tbody = document.getElementById('register-watch-body');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Create rows for each register that has been accessed
        for (const [regName, accesses] of Object.entries(data)) {
            const row = document.createElement('tr');
            
            // Register name cell with tooltip
            const nameCell = document.createElement('td');
            nameCell.className = 'register-name';
            nameCell.textContent = regName;
            
            // Add tooltip if description exists
            if (REGISTER_DESCRIPTIONS[regName]) {
                nameCell.title = REGISTER_DESCRIPTIONS[regName];
                nameCell.style.cursor = 'help';
            }
            
            row.appendChild(nameCell);
            
            // Writes cell
            const writesCell = document.createElement('td');
            writesCell.className = 'pc-list';
            if (accesses.writes.length > 0) {
                accesses.writes.forEach(pc => {
                    const pcElement = PCAddress.create(pc, { className: 'pc-address pc-list-item' });
                    writesCell.appendChild(pcElement);
                });
            } else {
                writesCell.textContent = '-';
            }
            row.appendChild(writesCell);
            
            // Reads cell
            const readsCell = document.createElement('td');
            readsCell.className = 'pc-list';
            if (accesses.reads.length > 0) {
                accesses.reads.forEach(pc => {
                    const pcElement = PCAddress.create(pc, { className: 'pc-address pc-list-item' });
                    readsCell.appendChild(pcElement);
                });
            } else {
                readsCell.textContent = '-';
            }
            row.appendChild(readsCell);
            
            tbody.appendChild(row);
        }
        
        // If no registers have been accessed, show a message
        if (Object.keys(data).length === 0) {
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 3;
            cell.textContent = 'No register accesses recorded yet';
            cell.style.textAlign = 'center';
            cell.style.fontStyle = 'italic';
            row.appendChild(cell);
            tbody.appendChild(row);
        }
        
    } catch (error) {
        console.error('Error fetching register watch data:', error);
    }
}