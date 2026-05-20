// Strips reasoning artifacts from a model response so built-in promptfoo
// assertions see only the final answer. Reasoning content for custom
// assertions is sourced from structured response fields, not from this
// transform — see docs/superpowers/specs/2026-05-07-reasoning-aware-assertions-design.md.

const THINK_PAIR = /<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi;
const RENDERED_THINKING = /^Thinking:[\s\S]*?\n{3,}/;

function normalize(output, context) {
  try {
    if (typeof output !== 'string') return output;
    let cleaned = output.replace(THINK_PAIR, '');
    cleaned = cleaned.replace(RENDERED_THINKING, '');
    return cleaned.trim();
  } catch (err) {
    process.stderr.write(`normalize_response: ${err.message}\n`);
    return output;
  }
}

module.exports = normalize;
