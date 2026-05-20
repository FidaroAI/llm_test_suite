GOTCHAS

When running the plaintext docker gateway locally, don't forget to set the phala vllm endpoint and brave credentials environment variables in the docker-compose file or in the docker command if you run directly.

Always run tests with --no-cache. promptfoo will caches results if they were successful

Currently we lose pre-web_search reasoning because the plaintext API doesn't use streaming. I'm not even sure if promptfoo can use streaming

OPEN QUESTIONS

If I talk to vllm endpoint directly then what am I losing?

NOTES

The reason we have \n\n\n in our responses after thinking is that deepseek_r1's parse strips <thinking> tags but swaps them for new lines. So we basically get two newlines for the tags and one for the already existing whitespace.

