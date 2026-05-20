const test = require('node:test');
const assert = require('node:assert');
const normalize = require('./normalize_response');

test('returns unchanged when no reasoning markers', () => {
  assert.strictEqual(normalize('pong', {}), 'pong');
  assert.strictEqual(normalize('hello world', {}), 'hello world');
});

test('strips a complete <think>...</think> pair', () => {
  assert.strictEqual(normalize('<think>reasoning here</think>pong', {}), 'pong');
});

test('strips multi-line <think> block and surrounding whitespace', () => {
  assert.strictEqual(
    normalize('<think>multi\nline\nstuff</think>\n\npong', {}),
    'pong'
  );
});

test('strips <thinking>...</thinking> variant', () => {
  assert.strictEqual(normalize('<thinking>foo</thinking>bar', {}), 'bar');
});

test('leaves unclosed <think> mention alone', () => {
  assert.strictEqual(
    normalize('What does <think> mean in HTML?', {}),
    'What does <think> mean in HTML?'
  );
});

test('handles empty string', () => {
  assert.strictEqual(normalize('', {}), '');
});

test('returns non-string inputs unchanged', () => {
  assert.strictEqual(normalize(null, {}), null);
  assert.strictEqual(normalize(undefined, {}), undefined);
});

test('strips Thinking:...newlines rendered-template artifact', () => {
  const input = 'Thinking: \nOkay let me consider this.\nMore thinking.\n\n\n\npong';
  assert.strictEqual(normalize(input, {}), 'pong');
});

test('Thinking: prefix without 3+ newline run is left alone', () => {
  // Only two newlines after — not the rendered artifact pattern
  const input = 'Thinking: I should reply\n\npong';
  assert.strictEqual(normalize(input, {}), input.trim());
});

test('Thinking: not at start of string is left alone', () => {
  const input = 'Notice: Thinking: hard about this\n\n\n\nAnswer';
  assert.strictEqual(normalize(input, {}), input.trim());
});

test('combined <think> tag and Thinking: prefix both stripped', () => {
  const input = 'Thinking: foo\n\n\n\n<think>bar</think>pong';
  assert.strictEqual(normalize(input, {}), 'pong');
});
