// Core configuration, global state, and shared helpers for the PC-E500 web UI.
(function (global) {
  const Config = {
    API_BASE: '/api/v1',
    POLL_INTERVAL: 100, // 100ms = 10fps
  };

  const State = {
    isRunning: false,
    pressedKeysClient: new Set(),
  };

  class PollManager {
    constructor() {
      this.tasks = new Map(); // name -> task
      this.isRunning = false;
    }

    register(cfg) {
      const task = {
        name: cfg.name,
        fn: cfg.fn,
        intervalMs: cfg.intervalMs,
        intervalMsRunning: cfg.intervalMsRunning,
        intervalMsPaused: cfg.intervalMsPaused,
        requiresRunning: !!cfg.requiresRunning,
        immediate: cfg.immediate !== false,
        timer: null,
        busy: false,
        currentInterval: null,
      };
      this.tasks.set(task.name, task);
      this._maybeStart(task);
    }

    setRunning(running) {
      this.isRunning = !!running;
      for (const task of this.tasks.values()) {
        this._maybeStart(task);
      }
    }

    stopAll() {
      for (const task of this.tasks.values()) this._stop(task);
    }

    _maybeStart(task) {
      const shouldRun = !task.requiresRunning || this.isRunning;
      const interval = this._effectiveInterval(task);
      if (shouldRun) {
        if (!task.timer) {
          this._start(task, interval);
        } else if (task.currentInterval !== interval) {
          this._stop(task);
          this._start(task, interval);
        }
      } else {
        this._stop(task);
      }
    }

    _start(task, interval) {
      const invoke = async () => {
        if (task.busy) return;
        task.busy = true;
        try {
          await task.fn();
        } catch (err) {
          // Swallow errors to keep polling alive; log for debugging.
          // eslint-disable-next-line no-console
          console.error('Poll task error:', err);
        } finally {
          task.busy = false;
        }
      };
      task.currentInterval = interval;
      if (task.immediate) invoke();
      task.timer = setInterval(invoke, interval);
    }

    _stop(task) {
      if (task.timer) {
        clearInterval(task.timer);
        task.timer = null;
        task.currentInterval = null;
      }
    }

    _effectiveInterval(task) {
      if (
        typeof task.intervalMsRunning === 'number' ||
        typeof task.intervalMsPaused === 'number'
      ) {
        return this.isRunning
          ? task.intervalMsRunning ?? task.intervalMs ?? 1000
          : task.intervalMsPaused ?? task.intervalMs ?? 1000;
      }
      return task.intervalMs ?? 1000;
    }
  }

  const polls = new PollManager();

  async function apiGet(path) {
    const response = await fetch(`${Config.API_BASE}${path}`);
    if (!response.ok) {
      throw new Error(`GET ${path} failed with status ${response.status}`);
    }
    return response.json();
  }

  async function apiPost(path, body) {
    const response = await fetch(`${Config.API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`POST ${path} failed with status ${response.status}: ${text}`);
    }
    return response.json();
  }

  async function control(command) {
    return apiPost('/control', { command });
  }

  async function sendKey(keyCode, action) {
    return apiPost('/key', { key_code: keyCode, action });
  }

  function setRunning(running) {
    State.isRunning = !!running;
    polls.setRunning(State.isRunning);
  }

  global.PCE500 = global.PCE500 || {};
  Object.assign(global.PCE500, {
    Config,
    State,
    PollManager,
    polls,
    control,
    sendKey,
    apiGet,
    apiPost,
    setRunning,
  });
})(window);

