// Click-to-copy PC address component used throughout the UI.
(function (global) {
  const g = global.PCE500 || (global.PCE500 = {});

  class PCAddress {
    constructor(displayText, options = {}) {
      this.displayText = displayText;
      this.copyAddress = displayText.split(' ')[0];
      this.className = options.className || 'pc-address';
      this.showTooltip = options.showTooltip !== false;
    }

    render() {
      const span = document.createElement('span');
      span.className = this.className;
      span.textContent = this.displayText;

      if (this.showTooltip) {
        span.title = 'Click to copy';
      }

      span.addEventListener('click', async (event) => {
        event.stopPropagation();
        try {
          await navigator.clipboard.writeText(this.copyAddress);
          span.classList.add('copied');
          const originalText = span.textContent;
          span.textContent = 'Copied!';
          setTimeout(() => {
            span.textContent = originalText;
            span.classList.remove('copied');
          }, 1000);
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error('Failed to copy:', err);
        }
      });
      return span;
    }

    static create(displayText, options = {}) {
      const component = new PCAddress(displayText, options);
      return component.render();
    }
  }

  g.PCAddress = PCAddress;
})(window);

