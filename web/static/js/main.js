// Application bootstrap.
(function (global) {
  const g = global.PCE500 || (global.PCE500 = {});

  function bootstrap() {
    document.addEventListener('DOMContentLoaded', () => {
      if (g.setupKeyboard) g.setupKeyboard();
      if (g.registerPhysicalKeyboardHandlers) g.registerPhysicalKeyboardHandlers();
      if (g.initUI) g.initUI();
      if (typeof g.updateState === 'function') {
        g.updateState();
      }
      if (typeof g.updateOcr === 'function') {
        g.updateOcr();
      }
    });
  }

  g.bootstrap = bootstrap;
  bootstrap();
})(window);

