function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }

  return new Promise((resolve, reject) => {
    try {
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.setAttribute('readonly', '');
      textArea.style.position = 'absolute';
      textArea.style.left = '-9999px';
      document.body.appendChild(textArea);
      textArea.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(textArea);
      ok ? resolve() : reject(new Error('execCommand copy failed'));
    } catch (e) {
      reject(e);
    }
  });
}

document.addEventListener('click', async (event) => {
  const button = event.target instanceof Element ? event.target.closest('[data-copy-target]') : null;
  if (!button) return;

  const targetId = button.getAttribute('data-copy-target');
  if (!targetId) return;
  const code = document.getElementById(targetId);
  if (!code) return;

  try {
    await copyTextToClipboard(code.textContent ?? '');
    const original = button.textContent ?? 'Copy';
    button.textContent = 'Copied';
    button.classList.add('copied');
    window.setTimeout(() => {
      button.textContent = original;
      button.classList.remove('copied');
    }, 1500);
  } catch (e) {
    // Silence errors to avoid noisy console on restricted environments.
  }
});
