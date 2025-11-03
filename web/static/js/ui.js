// UI wiring, polling, and DOM update helpers.
(function (global) {
  const g = global.PCE500 || (global.PCE500 = {});
  const { Config, State, polls, control, apiGet, setRunning } = g;
  const { PCAddress } = g;

  const REGISTER_DESCRIPTIONS = {
    RAMS: 'Scratchpad RAM window (0xB8000-0xB87FF)',
    BA: 'Accumulator BA register',
    BC: 'Accumulator BC register',
    DE: 'Accumulator DE register',
    HL: 'Accumulator HL register',
    IX: 'Index register X',
    IY: 'Index register Y',
    SP: 'Stack pointer',
    PC: 'Program counter',
    IMR: 'Interrupt mask register',
    ISR: 'Interrupt status register',
    TM0: 'Timer 0 counter',
    TM1: 'Timer 1 counter',
  };

  const INTERRUPT_BITS = {
    IMR: {
      7: { name: 'IRM', desc: 'Global interrupt mask: 0 disables all sources' },
      6: { name: 'EXM', desc: 'External IRQ mask (EXI enable)' },
      5: { name: 'RXRM', desc: 'UART Receiver Ready interrupt mask' },
      4: { name: 'TXRM', desc: 'UART Transmitter Ready interrupt mask' },
      3: { name: 'ONKM', desc: 'ON-Key interrupt mask' },
      2: { name: 'KEYM', desc: 'Key matrix interrupt mask (KI pins)' },
      1: { name: 'STM', desc: 'SEC (sub-CG) timer interrupt mask' },
      0: { name: 'MTM', desc: 'MSEC (main-CG) timer interrupt mask' },
    },
    ISR: {
      7: { name: 'RES', desc: 'Reserved (unused)' },
      6: { name: 'EXI', desc: 'External IRQ pending' },
      5: { name: 'RXRI', desc: 'UART Receiver Ready pending' },
      4: { name: 'TXRI', desc: 'UART Transmitter Ready pending' },
      3: { name: 'ONKI', desc: 'ON-Key pending' },
      2: { name: 'KEYI', desc: 'Key matrix pending (any enabled KI high)' },
      1: { name: 'STI', desc: 'SEC (sub-CG) timer pending' },
      0: { name: 'MTI', desc: 'MSEC (main-CG) timer pending' },
    },
  };

  function updateControlButtons() {
    const runBtn = document.getElementById('btn-run');
    const pauseBtn = document.getElementById('btn-pause');
    const stepBtn = document.getElementById('btn-step');
    if (!runBtn || !pauseBtn || !stepBtn) return;
    runBtn.disabled = State.isRunning;
    pauseBtn.disabled = !State.isRunning;
    stepBtn.disabled = State.isRunning;
  }

  async function handleRun() {
    try {
      const result = await control('run');
      if (result.status === 'running') {
        setRunning(true);
        updateControlButtons();
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error starting emulator:', error);
    }
  }

  async function handlePause() {
    try {
      const result = await control('pause');
      if (result.status === 'paused') {
        setRunning(false);
        updateControlButtons();
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error pausing emulator:', error);
    }
  }

  async function handleStep() {
    try {
      await control('step');
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error stepping emulator:', error);
    }
  }

  async function handleReset() {
    try {
      const result = await control('reset');
      if (result.status === 'reset') {
        setRunning(true);
        updateControlButtons();
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error resetting emulator:', error);
    }
  }

  async function handleTraceStart() {
    try {
      const response = await fetch('/trace/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      if (data.ok) {
        const startBtn = document.getElementById('btn-trace-start');
        const stopBtn = document.getElementById('btn-trace-stop');
        const download = document.getElementById('trace-download');
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        if (download) download.style.display = 'none';
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error starting trace:', error);
    }
  }

  async function handleTraceStop() {
    try {
      const response = await fetch('/trace/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      if (data.ok) {
        const startBtn = document.getElementById('btn-trace-start');
        const stopBtn = document.getElementById('btn-trace-stop');
        const download = document.getElementById('trace-download');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        if (download) download.style.display = 'inline-block';
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error stopping trace:', error);
    }
  }

  function setupControls() {
    const runBtn = document.getElementById('btn-run');
    const pauseBtn = document.getElementById('btn-pause');
    const stepBtn = document.getElementById('btn-step');
    const resetBtn = document.getElementById('btn-reset');
    const traceStartBtn = document.getElementById('btn-trace-start');
    const traceStopBtn = document.getElementById('btn-trace-stop');

    if (runBtn) runBtn.addEventListener('click', handleRun);
    if (pauseBtn) pauseBtn.addEventListener('click', handlePause);
    if (stepBtn) stepBtn.addEventListener('click', handleStep);
    if (resetBtn) resetBtn.addEventListener('click', handleReset);
    if (traceStartBtn) traceStartBtn.addEventListener('click', handleTraceStart);
    if (traceStopBtn) traceStopBtn.addEventListener('click', handleTraceStop);
  }

  async function updateOcr() {
    try {
      const data = await apiGet('/ocr');
      const el = document.getElementById('ocr-text');
      if (!el) return;
      if (data && data.ok) {
        const txt = (data.text || '').trim();
        el.textContent = txt || '(no text)';
      } else {
        el.textContent = '(ocr unavailable)';
      }
    } catch (error) {
      const el = document.getElementById('ocr-text');
      if (el) el.textContent = '(ocr error)';
    }
  }

  function updateInstructionHistory(history) {
    const historyContainer = document.getElementById('instruction-history');
    if (!historyContainer) return;
    historyContainer.innerHTML = '';
    const reversedHistory = history.slice().reverse();
    reversedHistory.forEach((item) => {
      const historyItem = document.createElement('div');
      historyItem.className = 'history-item';
      const pcElement = PCAddress.create(item.pc, {
        className: 'pc-address history-pc',
      });
      const instrSpan = document.createElement('span');
      instrSpan.className = 'history-instr';
      instrSpan.textContent = item.disassembly;
      historyItem.appendChild(pcElement);
      historyItem.appendChild(instrSpan);
      historyContainer.appendChild(historyItem);
    });
  }

  async function updateState() {
    try {
      const state = await apiGet('/state');
      const lcd = document.getElementById('lcd-display');
      if (state.screen && lcd) {
        lcd.src = state.screen;
      }

      if (state.registers) {
        const pcElement = document.getElementById('reg-pc');
        const pcValue = `0x${state.registers.pc.toString(16).padStart(6, '0').toUpperCase()}`;
        if (pcElement && pcElement.textContent !== pcValue) {
          pcElement.innerHTML = '';
          pcElement.appendChild(
            PCAddress.create(pcValue, { className: 'pc-address reg-value' }),
          );
        }
        const regNames = ['a', 'b', 'ba', 'i', 'x', 'y', 'u', 's'];
        const widths = { a: 2, b: 2, ba: 4, i: 4, x: 6, y: 6, u: 6, s: 6 };
        for (const name of regNames) {
          const el = document.getElementById(`reg-${name}`);
          if (!el) continue;
          const width = widths[name] || 2;
          const value = state.registers[name] ?? 0;
          el.textContent = `0x${value.toString(16).padStart(width, '0').toUpperCase()}`;
        }
      }

      if (state.flags) {
        const zEl = document.getElementById('flag-z');
        const cEl = document.getElementById('flag-c');
        if (zEl) zEl.textContent = state.flags.z;
        if (cEl) cEl.textContent = state.flags.c;
      }

      if (state.instruction_count !== undefined) {
        const countEl = document.getElementById('instruction-count');
        if (countEl) countEl.textContent = state.instruction_count.toLocaleString();
      }

      const speedElement = document.getElementById('emulation-speed');
      if (speedElement) {
        if (state.emulation_speed !== null && state.emulation_speed !== undefined) {
          let speedText;
          if (state.emulation_speed >= 1_000_000) {
            speedText = `${(state.emulation_speed / 1_000_000).toFixed(2)}M ips`;
          } else if (state.emulation_speed >= 1_000) {
            speedText = `${(state.emulation_speed / 1_000).toFixed(1)}K ips`;
          } else {
            speedText = `${Math.round(state.emulation_speed)} ips`;
          }
          if (state.speed_ratio !== null && state.speed_ratio !== undefined) {
            const percentage = (state.speed_ratio * 100).toFixed(1);
            speedText += ` (${percentage}%)`;
          }
          speedElement.textContent = speedText;
        } else {
          speedElement.textContent = '-';
        }
      }

      if (state.interrupts) {
        const ints = state.interrupts;
        const by = ints.by_source || {};
        const last = ints.last || {};
        const fmtHex = (v, w) =>
          typeof v === 'number'
            ? `0x${v.toString(16).padStart(w, '0').toUpperCase()}`
            : '-';
        const setText = (id, value) => {
          const el = document.getElementById(id);
          if (el) el.textContent = value;
        };
        setText('irq-total', `${ints.total ?? 0}`);
        setText('irq-key', `${by.KEY ?? 0}`);
        setText('irq-mti', `${by.MTI ?? 0}`);
        setText('irq-sti', `${by.STI ?? 0}`);
        const lastText = `${last.src || '-'}${
          last.pc != null ? ` @ ${fmtHex(last.pc, 6)}` : ''
        }${last.vector != null ? ` → ${fmtHex(last.vector, 6)}` : ''}`;
        setText('irq-last', lastText);
        setText('irq-imr-hex', (ints.imr || '0x00').toString());
        setText('irq-isr-hex', (ints.isr || '0x00').toString());
        setText('irq-irm', `${ints.irm ?? 0}`);
        setText('irq-keym', `${ints.keym ?? 0}`);
        setText('irq-isr-key', `${ints.isr_key ?? 0}`);
        setText('irq-pending', ints.pending ? 'true' : 'false');

        const watch = ints.watch || {};
        const tbody = document.getElementById('irq-bit-watch-body');
        if (tbody) {
          tbody.innerHTML = '';
          const imrHex = ints.imr || '0x00';
          const isrHex = ints.isr || '0x00';
          const imrVal = parseInt(imrHex, 16) || 0;
          const isrVal = parseInt(isrHex, 16) || 0;
          const renderRows = (regName, bitsMap, watchMap) => {
            const bits = Object.keys(bitsMap)
              .map(Number)
              .sort((a, b) => b - a);
            for (const bit of bits) {
              const info = bitsMap[bit] || { name: `b${bit}`, desc: '' };
              const row = document.createElement('tr');
              const ledCell = document.createElement('td');
              ledCell.className = 'irq-bit-cell';
              const led = document.createElement('span');
              const isSet =
                regName === 'IMR'
                  ? (imrVal >> bit) & 1
                  : (isrVal >> bit) & 1;
              led.className = `bit-led${regName === 'ISR' ? ' isr' : ''}${isSet ? ' on' : ''}`;
              led.title = `${regName}.${bit} (${info.name})${
                info.desc ? `\n${info.desc}` : ''
              }`;
              ledCell.appendChild(led);
              const labelCell = document.createElement('td');
              labelCell.textContent = `${regName}.${bit} (${info.name})`;
              if (info.desc) {
                labelCell.title = info.desc;
                labelCell.style.cursor = 'help';
              }
              const countCell = document.createElement('td');
              let countText = '–';
              if (regName === 'ISR') {
                const map = {
                  0: 'MTI',
                  1: 'STI',
                  2: 'KEY',
                  3: 'ONK',
                  4: 'TXR',
                  5: 'RXR',
                  6: 'EXI',
                  7: 'RES',
                };
                const key = map[bit];
                const count = key && by[key] ? by[key] : 0;
                countText = count ? `×${count}` : '0';
              }
              countCell.textContent = countText;
              const setCell = document.createElement('td');
              setCell.className = 'pc-list';
              const clearCell = document.createElement('td');
              clearCell.className = 'pc-list';
              const rec = watchMap[bit] || { set: [], clear: [] };
              const addList = (cell, arr) => {
                if (Array.isArray(arr) && arr.length) {
                  arr.forEach((pc) => {
                    const pcText = `0x${Number(pc)
                      .toString(16)
                      .padStart(6, '0')
                      .toUpperCase()}`;
                    cell.appendChild(
                      PCAddress.create(pcText, {
                        className: 'pc-address pc-list-item',
                      }),
                    );
                  });
                } else {
                  cell.textContent = '-';
                }
              };
              addList(setCell, rec.set);
              addList(clearCell, rec.clear);
              row.appendChild(ledCell);
              row.appendChild(labelCell);
              row.appendChild(countCell);
              row.appendChild(setCell);
              row.appendChild(clearCell);
              tbody.appendChild(row);
            }
          };
          renderRows('IMR', INTERRUPT_BITS.IMR || {}, watch.IMR || {});
          renderRows('ISR', INTERRUPT_BITS.ISR || {}, watch.ISR || {});
        }
      }

      if (state.instruction_history) {
        updateInstructionHistory(state.instruction_history);
      }

      setRunning(state.is_running || false);
      updateControlButtons();
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error fetching state:', error);
    }
  }

  async function updateRegisterWatch() {
    try {
      const data = await apiGet('/imem_watch');
      const tbody = document.getElementById('register-watch-body');
      if (!tbody) return;
      tbody.innerHTML = '';
      for (const [regName, accesses] of Object.entries(data)) {
        const row = document.createElement('tr');
        const nameCell = document.createElement('td');
        nameCell.textContent = regName;
        if (REGISTER_DESCRIPTIONS[regName]) {
          nameCell.title = REGISTER_DESCRIPTIONS[regName];
          nameCell.style.cursor = 'help';
        }
        row.appendChild(nameCell);

        const writeCell = document.createElement('td');
        writeCell.className = 'pc-list';
        if (Array.isArray(accesses.writes) && accesses.writes.length) {
          for (const access of accesses.writes) {
            writeCell.appendChild(
              PCAddress.create(access.pc, {
                className: 'pc-address pc-list-item',
              }),
            );
          }
        } else {
          writeCell.textContent = '-';
        }
        row.appendChild(writeCell);

        const readCell = document.createElement('td');
        readCell.className = 'pc-list';
        if (Array.isArray(accesses.reads) && accesses.reads.length) {
          for (const access of accesses.reads) {
            readCell.appendChild(
              PCAddress.create(access.pc, {
                className: 'pc-address pc-list-item',
              }),
            );
          }
        } else {
          readCell.textContent = '-';
        }
        row.appendChild(readCell);
        tbody.appendChild(row);
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error fetching register watch data:', error);
    }
  }

  async function updateLcdStats() {
    try {
      const data = await apiGet('/lcd_stats');
      if (!data) return;
      const csBoth = document.getElementById('cs-both-count');
      const csLeft = document.getElementById('cs-left-count');
      const csRight = document.getElementById('cs-right-count');
      if (csBoth) csBoth.textContent = data.chip_select?.both ?? 0;
      if (csLeft) csLeft.textContent = data.chip_select?.left ?? 0;
      if (csRight) csRight.textContent = data.chip_select?.right ?? 0;

      const chips = data.chips || {};
      for (const index of [0, 1]) {
        const prefix = index === 0 ? 'chip0' : 'chip1';
        const info = chips[index] || {};
        const onEl = document.getElementById(`${prefix}-on`);
        const instructionsEl = document.getElementById(`${prefix}-instructions`);
        const onOffEl = document.getElementById(`${prefix}-on-off`);
        const dataEl = document.getElementById(`${prefix}-data`);
        const pageEl = document.getElementById(`${prefix}-page`);
        if (onEl) onEl.textContent = info.display_on ? 'ON' : 'OFF';
        if (instructionsEl) instructionsEl.textContent = info.instructions ?? 0;
        if (onOffEl) onOffEl.textContent = info.on_off_commands ?? 0;
        if (dataEl) dataEl.textContent = info.data_written ?? 0;
        if (pageEl) pageEl.textContent = info.last_page ?? '-';
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error fetching LCD statistics:', error);
    }
  }

  async function updateKeyQueue() {
    try {
      const data = await apiGet('/key_queue');
      const queueBody = document.getElementById('key-queue-body');
      const kolEl = document.getElementById('key-reg-kol');
      const kohEl = document.getElementById('key-reg-koh');
      const kilEl = document.getElementById('key-reg-kil');
      if (kolEl) kolEl.textContent = data.registers?.kol ?? '0x00';
      if (kohEl) kohEl.textContent = data.registers?.koh ?? '0x00';
      if (kilEl) kilEl.textContent = data.registers?.kil ?? '0x00';
      if (!queueBody) return;
      queueBody.innerHTML = '';
      const queue = Array.isArray(data.queue) ? data.queue : [];
      if (!queue.length) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 8;
        cell.style.textAlign = 'center';
        cell.style.fontStyle = 'italic';
        cell.textContent = 'No keys in queue';
        row.appendChild(cell);
        queueBody.appendChild(row);
        return;
      }
      queue.forEach((key) => {
        const row = document.createElement('tr');
        const makeCell = (value) => {
          const cell = document.createElement('td');
          cell.textContent = value;
          return cell;
        };
        row.appendChild(makeCell(key.name || key.matrix_code || ''));
        row.appendChild(makeCell(key.column ?? '-'));
        row.appendChild(makeCell(key.row ?? '-'));
        row.appendChild(makeCell(key.kol ?? '0x00'));
        row.appendChild(makeCell(key.koh ?? '0x00'));
        row.appendChild(makeCell(key.kil ?? '0x00'));

        const progressCell = document.createElement('td');
        const progress = key.target_reads
          ? (key.read_count / key.target_reads) * 100
          : 0;
        const progressDiv = document.createElement('div');
        progressDiv.className = 'key-progress';
        const bar = document.createElement('div');
        bar.className = 'key-progress-bar';
        bar.style.width = `${progress}%`;
        const text = document.createElement('div');
        text.className = 'key-progress-text';
        text.textContent = key.progress;
        progressDiv.appendChild(bar);
        progressDiv.appendChild(text);
        progressCell.appendChild(progressDiv);
        row.appendChild(progressCell);

        const statusCell = document.createElement('td');
        const statusSpan = document.createElement('span');
        statusSpan.className = 'key-status';
        if (key.is_stuck) {
          statusSpan.classList.add('stuck');
          statusSpan.textContent = 'STUCK';
        } else if (key.released && key.read_count < key.target_reads) {
          statusSpan.classList.add('released');
          statusSpan.textContent = 'RELEASED';
        } else if (key.read_count > 0) {
          statusSpan.classList.add('active');
          statusSpan.textContent = 'ACTIVE';
        } else {
          statusSpan.classList.add('waiting');
          statusSpan.textContent = 'WAITING';
        }
        statusCell.appendChild(statusSpan);
        row.appendChild(statusCell);
        queueBody.appendChild(row);
      });
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Error fetching key queue:', error);
    }
  }

  function setupLCDInteraction() {
    const lcdDisplay = document.getElementById('lcd-display');
    if (!lcdDisplay) return;
    const tooltip = document.createElement('div');
    tooltip.className = 'lcd-tooltip';
    tooltip.style.display = 'none';
    document.body.appendChild(tooltip);
    const columnHighlight = document.createElement('div');
    columnHighlight.className = 'lcd-column-highlight';
    columnHighlight.style.display = 'none';
    columnHighlight.style.position = 'absolute';
    lcdDisplay.parentElement.appendChild(columnHighlight);

    lcdDisplay.addEventListener('mousemove', async (event) => {
      const rect = lcdDisplay.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const pixelX = Math.floor(x / 2);
      const pixelY = Math.floor(y / 2);
      if (pixelX < 0 || pixelX >= 240 || pixelY < 0 || pixelY >= 32) {
        tooltip.style.display = 'none';
        columnHighlight.style.display = 'none';
        return;
      }
      try {
        const data = await apiGet(`/lcd_debug?x=${pixelX}&y=${pixelY}`);
        if (data.pc) {
          tooltip.innerHTML = '';
          const label = document.createElement('span');
          label.textContent = 'PC: ';
          tooltip.appendChild(label);
          const pcElement = PCAddress.create(data.pc, {
            className: 'pc-address lcd-pc',
            showTooltip: false,
          });
          tooltip.appendChild(pcElement);
          tooltip.style.display = 'block';
          tooltip.style.left = `${event.pageX + 10}px`;
          tooltip.style.top = `${event.pageY - 30}px`;
          const imgRect = lcdDisplay.getBoundingClientRect();
          const containerRect = lcdDisplay.parentElement.getBoundingClientRect();
          const imgOffsetX = imgRect.left - containerRect.left;
          const imgOffsetY = imgRect.top - containerRect.top;
          const columnX = Math.floor(pixelX) * 2;
          columnHighlight.style.display = 'block';
          columnHighlight.style.left = `${imgOffsetX + columnX}px`;
          columnHighlight.style.top = `${imgOffsetY}px`;
          columnHighlight.style.height = `${imgRect.height}px`;
        } else {
          tooltip.style.display = 'none';
          columnHighlight.style.display = 'none';
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error fetching LCD debug info:', error);
        tooltip.style.display = 'none';
        columnHighlight.style.display = 'none';
      }
    });

    lcdDisplay.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
      columnHighlight.style.display = 'none';
    });
  }

  function registerPolls() {
    polls.register({
      name: 'state',
      fn: updateState,
      intervalMsRunning: Config.POLL_INTERVAL,
      intervalMsPaused: 10000,
      requiresRunning: false,
    });
    polls.register({
      name: 'ocr',
      fn: updateOcr,
      intervalMs: 500,
      requiresRunning: true,
    });
    polls.register({
      name: 'imem',
      fn: updateRegisterWatch,
      intervalMs: 500,
      requiresRunning: true,
    });
    polls.register({
      name: 'lcd',
      fn: updateLcdStats,
      intervalMs: 500,
      requiresRunning: true,
    });
    polls.register({
      name: 'keys',
      fn: updateKeyQueue,
      intervalMs: 200,
      requiresRunning: true,
    });
  }

  function initUI() {
    setupControls();
    setupLCDInteraction();
    registerPolls();
  }

  Object.assign(g, {
    updateControlButtons,
    updateOcr,
    updateState,
    updateRegisterWatch,
    updateLcdStats,
    updateKeyQueue,
    setupLCDInteraction,
    setupControls,
    registerPolls,
    initUI,
  });
})(window);

