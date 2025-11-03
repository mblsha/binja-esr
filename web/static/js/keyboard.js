// Keyboard layout, rendering, and input handling.
(function (global) {
  const g = global.PCE500 || (global.PCE500 = {});
  const { State, sendKey } = g;

  const KEYBOARD_LEFT = [
    [
      { key: 'KEY_UP_DOWN', label: '↕', class: 'special' },
      { key: 'KEY_F1', label: 'PF1', superscript: 'BASIC' },
      { key: 'KEY_F2', label: 'PF2' },
      { key: 'KEY_F3', label: 'PF3' },
      { key: 'KEY_F4', label: 'PF4' },
      { key: 'KEY_F5', label: 'PF5' },
      { key: 'KEY_2NDF', label: '2ndF', class: 'special' },
    ],
    [
      { key: 'KEY_MENU', label: 'MENU', class: 'special' },
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
      { key: 'KEY_STO', label: 'STO', class: 'special' },
    ],
    [
      { key: 'KEY_BASIC', label: 'BASIC', class: 'special' },
      { key: 'KEY_A', label: 'A' },
      { key: 'KEY_S', label: 'S' },
      { key: 'KEY_D', label: 'D' },
      { key: 'KEY_F', label: 'F' },
      { key: 'KEY_G', label: 'G' },
      { key: 'KEY_H', label: 'H' },
      { key: 'KEY_J', label: 'J' },
      { key: 'KEY_K', label: 'K' },
      { key: 'KEY_L', label: 'L' },
      { key: 'KEY_RCL', label: 'RCL', class: 'special' },
    ],
    [
      { key: 'KEY_OFF', label: 'OFF', class: 'special off-on' },
      { key: 'KEY_ON', label: 'ON', class: 'special off-on' },
      { key: 'KEY_Z', label: 'Z' },
      { key: 'KEY_X', label: 'X' },
      { key: 'KEY_C', label: 'C' },
      { key: 'KEY_V', label: 'V' },
      { key: 'KEY_B', label: 'B' },
      { key: 'KEY_N', label: 'N' },
      { key: 'KEY_M', label: 'M' },
      { key: 'KEY_COMMA', label: ',' },
      { key: 'KEY_SEMICOLON', label: ';' },
      { key: 'KEY_EQUALS', label: '↵', class: 'tall-enter', rowspan: 2 },
    ],
    [
      { key: 'KEY_SHIFT', label: 'SHIFT', class: 'special' },
      { key: 'KEY_CTRL', label: 'CTRL', class: 'special' },
      { key: 'KEY_SPACE', label: 'SPACE', class: 'wide-space' },
      { key: 'KEY_TAB', label: 'TAB', class: 'special' },
      { key: 'KEY_ESCAPE', label: 'ESC', class: 'special' },
      { key: 'KEY_RETURN', label: 'RETURN', class: 'special' },
    ],
  ];

  const KEYBOARD_RIGHT = [
    [
      { key: 'KEY_INS', label: 'INS', class: 'control' },
      { key: 'KEY_DEL', label: 'DEL', class: 'control' },
      { key: 'KEY_UP', label: '↑', class: 'arrow' },
      { key: 'KEY_CAL', label: 'CAL', class: 'special' },
      { key: 'KEY_MEMO', label: 'MEMO', class: 'special' },
    ],
    [
      { key: 'KEY_MODE', label: 'MODE', class: 'special' },
      { key: 'KEY_LEFT', label: '←', class: 'arrow' },
      { key: 'KEY_DOWN', label: '↓', class: 'arrow' },
      { key: 'KEY_RIGHT', label: '→', class: 'arrow' },
      { key: 'KEY_CANCEL', label: 'C-CE', class: 'special' },
    ],
    [
      { key: 'KEY_EXP', label: 'EXP', class: 'num' },
      { key: 'KEY_LN', label: 'LN', class: 'num' },
      { key: 'KEY_LOG', label: 'LOG', class: 'num' },
      { key: 'KEY_SIN', label: 'SIN', class: 'num' },
      { key: 'KEY_COS', label: 'COS', class: 'num' },
      { key: 'KEY_TAN', label: 'TAN', class: 'num' },
    ],
    [
      { key: 'KEY_E', label: 'e', class: 'num' },
      { key: 'KEY_PI', label: 'π', class: 'num' },
      { key: 'KEY_LPAREN', label: '(', class: 'num' },
      { key: 'KEY_RPAREN', label: ')', class: 'num' },
      { key: 'KEY_DIVIDE', label: '÷', class: 'op' },
      { key: 'KEY_MULTI', label: '×', class: 'op' },
    ],
    [
      { key: 'KEY_7', label: '7', class: 'num' },
      { key: 'KEY_8', label: '8', class: 'num' },
      { key: 'KEY_9', label: '9', class: 'num' },
      { key: 'KEY_MINUS', label: '-', class: 'op' },
      { key: 'KEY_PERCENT', label: '%', class: 'op' },
    ],
    [
      { key: 'KEY_4', label: '4', class: 'num' },
      { key: 'KEY_5', label: '5', class: 'num' },
      { key: 'KEY_6', label: '6', class: 'num' },
      { key: 'KEY_MULTIPLY', label: '×', class: 'op' },
      { key: 'KEY_BACKSPACE', label: 'BS', class: 'control' },
    ],
    [
      { key: 'KEY_1', label: '1', class: 'num' },
      { key: 'KEY_2', label: '2', class: 'num' },
      { key: 'KEY_3', label: '3', class: 'num' },
      { key: 'KEY_MINUS_ALT', label: '-', class: 'op' },
      { key: 'KEY_INSERT', label: 'INS', class: 'control' },
    ],
    [
      { key: 'KEY_0', label: '0', class: 'num' },
      { key: 'KEY_PLUSMINUS', label: '+/-', class: 'num' },
      { key: 'KEY_PERIOD', label: '•', class: 'num' },
      { key: 'KEY_PLUS', label: '+', class: 'op' },
      { key: 'KEY_EQUALS', label: '=', class: 'equals' },
    ],
  ];

  function buildPhysicalKeyMap() {
    const map = {
      Enter: 'KEY_ENTER',
      Backspace: 'KEY_BACKSPACE',
      Tab: 'KEY_TAB',
      Escape: 'KEY_ON',
      ' ': 'KEY_SPACE',
      ArrowUp: 'KEY_UP',
      ArrowDown: 'KEY_DOWN',
      ArrowLeft: 'KEY_LEFT',
      ArrowRight: 'KEY_RIGHT',
      Delete: 'KEY_DELETE',
      Insert: 'KEY_INSERT',
    };

    for (const letter of Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) {
      map[letter] = `KEY_${letter}`;
      map[letter.toLowerCase()] = `KEY_${letter}`;
    }

    for (const num of '0123456789') {
      map[num] = `KEY_${num}`;
    }

    return map;
  }

  const physicalKeyMap = buildPhysicalKeyMap();

  function handleKeyPress(keyCode) {
    const keyElement = document.querySelector(`[data-key-code="${keyCode}"]`);
    if (keyElement) {
      keyElement.classList.add('pressed');
    }
    if (State.pressedKeysClient.has(keyCode)) return;
    State.pressedKeysClient.add(keyCode);
    return sendKey(keyCode, 'press').catch((err) => {
      // eslint-disable-next-line no-console
      console.error('Error sending key press:', err);
    });
  }

  function handleKeyRelease(keyCode) {
    const keyElement = document.querySelector(`[data-key-code="${keyCode}"]`);
    if (keyElement) {
      keyElement.classList.remove('pressed');
    }
    if (!State.pressedKeysClient.has(keyCode)) return;
    State.pressedKeysClient.delete(keyCode);
    return sendKey(keyCode, 'release').catch((err) => {
      // eslint-disable-next-line no-console
      console.error('Error sending key release:', err);
    });
  }

  function setupKeyboardSection(layout, container) {
    let enterButton = null;

    layout.forEach((row) => {
      const rowDiv = document.createElement('div');
      rowDiv.className = 'keyboard-row';

      row.forEach((keyDef) => {
        const keyButton = document.createElement('button');
        keyButton.className = `key ${keyDef.class || ''}`.trim();
        keyButton.dataset.keyCode = keyDef.key;

        if (keyDef.rowspan) {
          keyButton.style.gridRow = `span ${keyDef.rowspan}`;
          enterButton = keyButton;
        }

        if (keyDef.superscript) {
          const wrapper = document.createElement('div');
          wrapper.className = 'key-wrapper';
          const superscript = document.createElement('div');
          superscript.className = 'key-superscript';
          superscript.textContent = keyDef.superscript;
          const mainLabel = document.createElement('div');
          mainLabel.className = 'key-main';
          mainLabel.textContent = keyDef.label;
          wrapper.appendChild(superscript);
          wrapper.appendChild(mainLabel);
          keyButton.appendChild(wrapper);
        } else {
          keyButton.textContent = keyDef.label;
        }

        keyButton.addEventListener('mousedown', () => handleKeyPress(keyDef.key));
        keyButton.addEventListener('mouseup', () => handleKeyRelease(keyDef.key));
        keyButton.addEventListener('mouseleave', () => handleKeyRelease(keyDef.key));
        keyButton.addEventListener('click', (e) => e.preventDefault());

        rowDiv.appendChild(keyButton);
      });

      container.appendChild(rowDiv);

      if (enterButton) {
        enterButton.classList.add('tall-enter');
      }
    });
  }

  function setupKeyboard() {
    const keyboardContainer = document.getElementById('virtual-keyboard');
    if (!keyboardContainer) return;

    const leftSection = document.createElement('div');
    leftSection.className = 'keyboard-left';
    const rightSection = document.createElement('div');
    rightSection.className = 'keyboard-right';

    setupKeyboardSection(KEYBOARD_LEFT, leftSection);
    setupKeyboardSection(KEYBOARD_RIGHT, rightSection);

    keyboardContainer.appendChild(leftSection);
    keyboardContainer.appendChild(rightSection);
  }

  function registerPhysicalKeyboardHandlers() {
    document.addEventListener('keydown', (e) => {
      const virtualKey = physicalKeyMap[e.key];
      if (!virtualKey) return;
      e.preventDefault();
      handleKeyPress(virtualKey);
    });

    document.addEventListener('keyup', (e) => {
      const virtualKey = physicalKeyMap[e.key];
      if (!virtualKey) return;
      e.preventDefault();
      handleKeyRelease(virtualKey);
    });
  }

  Object.assign(g, {
    KEYBOARD_LEFT,
    KEYBOARD_RIGHT,
    handleKeyPress,
    handleKeyRelease,
    setupKeyboard,
    registerPhysicalKeyboardHandlers,
  });
})(window);
