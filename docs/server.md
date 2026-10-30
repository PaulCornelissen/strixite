# What the server accepts

strixite's server speaks the OpenAI API on port 5300. This page covers the parts where it does something you might
not expect: how thinking is switched and limited, the thinking nudge, prefill progress in a stream, conversations
served together, and exactly what structured output enforces - and what it refuses.

| endpoint | what it is |
|---|---|
| `POST /v1/chat/completions` | chat, with streaming, tool calls, reasoning and structured output |
| `POST /v1/completions` | a raw prompt, no chat template (same engine, queue and prompt cache) |
| `GET /v1/models` | the one served model |
| `GET /health`, `GET /cache`, `GET /metrics` | liveness, the prompt cache's state, Prometheus counters |

Every response carries a `strix` object next to `usage` with what the server did: whether thinking was `on`, `off`
or `skipped`, a thinking budget that was hit, MTP acceptance, and the number of thinking nudges.

## Thinking

The model thinks before it answers, in a think block returned as `reasoning_content`. Thinking is **on** unless the
request turns it off. Any of these spellings works - clients differ, so the server accepts them all:

| to... | send any of |
|---|---|
| turn thinking off | `"enable_thinking": false`, `"chat_template_kwargs": {"enable_thinking": false}`, `"thinking": {"type": "disabled"}`, `"reasoning_effort": "none"` |
| turn it on explicitly | the same with `true` / `"enabled"`, or any other `reasoning_effort` |
| cap it | `"max_thinking_tokens": N`, `"reasoning": {"max_tokens": N}`, `"thinking": {"budget_tokens": N}`, `"chat_template_kwargs": {"thinking_budget": N}` |

Two spellings that disagree (on and off, or two different budgets) are a 400, never a guess.

When a budget runs out, the server closes the think block itself and the model goes on to answer. Without a budget,
thinking may use whatever `max_tokens` leaves after room for the answer (1024 tokens, or 15% of `max_tokens` if
that's more). A request that didn't ask for thinking and whose `max_tokens` leaves no room at all is answered with
thinking skipped, and `strix.thinking` says so.

Server setting: `thinking = off` in the config makes requests that don't say either way run without thinking. I
measured it on terminal-bench: the model took 4-7 times as many steps and was no faster, so the shipped default is on.

Server setting: `thinking-cap = N` caps the thinking of every request that sets no budget of its own: past N tokens
(and the end of the sentence it's in) the server closes the think block and the model answers. The shipped default
is `0`, no cap. v0.3 shipped 2,048, and people told me the model seemed less smart with it, so from v0.4.0 the cap is
yours to set. A request's own budget always wins.

## The thinking nudge

Sometimes the model's thinking goes in circles - the same calculation, the same doubt, again and again. When the
server sees that, it writes one sentence into the think block in the model's own voice, suggesting it commit to an
approach. It never closes the think block itself; the model decides whether to wrap up.

- **When:** at least 25% of the last 1024 thinking tokens repeat 8-token sequences seen earlier in the same thinking,
  and the thinking is at least 3072 tokens long. It waits for a paragraph break, so it never lands mid-sentence.
- **At most twice** per turn: a firmer second nudge if the circling goes on well past the first.
- **How often:** rarely. On a full terminal-bench run it fired in 14 of ~550 requests, and most of those turns
  finished thinking within ~1.4k tokens. On turns forked at the exact point it fires, no nudge meant no turn stopped
  within 2000 more tokens; with the nudge, a quarter did.
- **What I tried and dropped:** nudging earlier (10% repetition from 1024 tokens) - mostly ignored; and a nudge on
  length alone - ignored, or it made the model write its answer inside the think block.
- **Settings:** `think-nudge = on|off`, `think-nudge-rate` (0.25), `think-nudge-min-tokens` (3072). Turn it off for
  like-for-like comparisons with engines that don't have it. `strix.think_nudges` in a response counts the nudges.

## Streaming

A streamed response (`"stream": true`) sends server-sent events: one `data:` chunk per piece of reasoning, content
or tool call, then a chunk with the `finish_reason`, a last chunk with `usage` and `strix`, and `data: [DONE]`.

**Prefill progress.** A long prompt can take minutes to read before the first token. By default the stream says how
far it got only in SSE comment lines (`: prefill 24576/65536`), which keep the connection alive but which clients
don't show. A request that sends `"return_progress": true` gets it as `data:` chunks instead - one when the prefill
begins, one after each prefill chunk - with a top-level `prompt_progress` object and an empty delta:

```json
{"id": "chatcmpl-...", "object": "chat.completion.chunk", "created": 1760000000, "model": "...",
 "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
 "prompt_progress": {"total": 65536, "cache": 8192, "processed": 24576, "time_ms": 9120}}
```

`total` is the prompt's tokens, `cache` the ones reused from the prompt cache rather than read again, `processed`
the ones in the model's state so far (cached ones included; it reaches `total` when the prefill is done), `time_ms`
the time since the prefill began. llama.cpp's server streams the same field with the same keys for the same flag,
so a harness that asks llama.cpp for it (llama.cpp's own web UI does, when it streams) can ask strixite too.
`/v1/completions` takes the flag as well (its chunks carry empty `text`); without streaming it does nothing.

## Conversations at once

The server holds `total-slots` conversations at the same time (2 by default, up to 8; [how many fit a given
memory](memory.md#more-conversations-at-once)). Each slot is one conversation's
state at the full context, so a coding agent and its subagents, or two clients, don't evict one another - and their
decode steps share one forward through the model, so the weights are read once for all of them. Two conversations
cost nearly nothing extra; more of them share the step but read more expert weights each, so deep conversations
stay best in pairs and shallow ones run in crowds. Requests past the slots wait
in arrival order; `strix_busy` out of `strix_slots` in `/metrics` shows how many are taken, and while one slot reads
a long prompt the others keep decoding, because the prefill runs in small chunks when a second slot is busy. A slot's
memory grows with its conversation: [memory and the slot recipes](memory.md#slots-conversations-at-once).

## Structured output

`response_format` with `json_object` or `json_schema` (`{name, strict, schema}`) constrains the answer token by token:
the model can only produce text the schema allows. Thinking is unaffected - the constraint starts at the first token
after the think block (or the first token, with thinking off).

**Enforced:** `type`, `properties`, `required`, `additionalProperties: false`, `items`, `enum`, `const`, `anyOf` /
`oneOf`, `$ref` with `$defs`, `minLength` / `maxLength` (in code points) and `minItems` / `maxItems` (bounds up to
4096).

**Stricter than the schema in two ways, on purpose:**
- `strict: false` is constrained exactly like `strict: true`.
- Whitespace between JSON tokens is bounded, so the model can't run away with spaces.

**Ignored** (they describe, they don't constrain): `description`, `title`, `default`, `examples`, `$comment`.

**Refused with a 400 naming the keyword and where it is** - never accepted and silently left unenforced: numeric
bounds (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`), `pattern`, `format`,
`uniqueItems`, `allOf`, `not`, `if` / `then` / `else`, `prefixItems`, `patternProperties`, `propertyNames`,
`dependentRequired` / `dependentSchemas`, and a schema for `additionalProperties`. Also refused: `json_object` /
`json_schema` together with tools (the answer is one or the other), and `tool_choice` `required` or a named tool.

These wait for a real client that needs them. If one of these 400s gets in your way, please open an issue with the
schema - that's what moves a keyword up the list.

A schema is compiled once and cached (the 16 most recent), so repeating one costs nothing the second time. Long
`maxLength` bounds cost a little on a schema's first use, since every position in the string is its own state.

## Streamed tool calls

In a streamed response (`"stream": true`) a tool call goes out as the model writes it, the way OpenAI streams one:
the call's `id`, `type` and function `name` as soon as the model has written the name, then its `arguments` piece by
piece - a string parameter (a file's content, a command) as its characters come, a number, boolean, object or array
whole once the model closes it, and the closing `}` when the call ends. Joined, the pieces are exactly the
`arguments` a non-streamed response would have.

A streamed call can't be taken back. A call that turns out malformed - a tag out of place, text after
`</function>`, a parameter named twice - would come back as plain `content` without streaming (or, named twice,
with the last value); streamed, it stays a call whose `arguments` never get their closing `}`, so they don't parse
as JSON. A call cut off - by `max_tokens`, or by the model ending its turn inside it - is left the same way. Either
way `finish_reason` is `tool_calls` (`length` for a `max_tokens` cut), and `strix.malformed_tool_calls` in the last
chunk counts such calls (`strix.dropped_partial_tool_call` says one was cut off).

What a client does with unparseable arguments is up to it: most report an error back to the model, which then tries
again; openai-python's parsing stream helpers (strict tools) raise; a client that repairs partial JSON (LangChain)
can run the repaired call - complete when the call broke between parameters, wrong when it was cut off inside a
value. Sent back in the conversation, a call whose arguments were left open this way is shown to the model with no
parameters, rather than failing the request.

## Responses (`POST /v1/responses`) - the Codex wire

This endpoint speaks the OpenAI **Responses** API that Codex drives (the URL is versioned like OpenAI's,
`/v1/responses`, so a provider `base_url` of `http://<host>:5300/v1` with `wire_api = "responses"` is all a
client needs). It runs the same engine, queue, prompt cache, thinking and grammar machinery as Chat - it is a
protocol surface, not a second server. It is **stateless**: responses are not stored, every request carries
its full input (a `GET /v1/responses/{id}` gets a typed 404 that says so, `store: true` and
`previous_response_id` are typed 400s).

### The fields it serves

| field | behavior |
|---|---|
| `model` | must name the model this server runs - a Responses client keys behavior off the name, so this route never pretends (unlike the Chat route, which serves any name and reports the real id) |
| `input` | a string, or an array of items: `message` (roles `system`/`developer`/`user`/`assistant`), `function_call`, `custom_tool_call`, `function_call_output`, `custom_tool_call_output`, `reasoning` (validated, treated as display metadata - never replayed, never invented) |
| `instructions` | prepended as a `system` message |
| `tools` | `function` (flat `name`/`description`/`parameters`), `custom` (freeform, e.g. `apply_patch`), `namespace` groups; names checked against template/tag forgery (a control byte, `<`, `>` or `=` is a named 400) |
| `tool_choice` | `"auto"` (default) or `"none"`; a named tool choice is a 400 (the server cannot force a call the model's template does not constrain) |
| `parallel_tool_calls` | accepted as a shape marker: every call the model writes is published, there is no cap to lift |
| `stream` | typed Responses SSE (see below) |
| `store` | `false`/absent; `true` is a 400 (stateless) |
| `max_output_tokens` | clamped to the room the context leaves, as Chat's `max_tokens` (the answer never truncates the prompt) |
| `temperature`, `top_p`, `seed`, `stop` | the Chat sampling parameters, same ranges and clamps |
| `reasoning` | `effort` (see the table below), `summary` accepted and ignored (raw reasoning streams; nothing is summarized) |
| `include` | `["reasoning.encrypted_content"]` accepted - `encrypted_content` is always `null` (no key, no fake ciphertext); any other inclusion is a 400 |
| `text.format` | `text`, `json_object`, `json_schema` (see structured output) |
| `truncation` | `"disabled"` only: a prompt over the context is refused with a named 400, never silently cut |
| `prompt_cache_key`, `client_metadata`, `metadata`, `user`, `safety_identifier` | validated benign client metadata (accepted, shape-checked, not behavior-changing) |

**Any other field is a 400 naming it.** A field this endpoint would silently ignore is a semantic the client
did not agree to - the strict list is deliberate (it also keeps protocol additions visible the day they appear).

Hosted tools: `web_search` is **dropped from what the model sees, disclosed here** - installed Codex sends it
by default and this server executes no hosted tools, so a refusal would make Codex unusable; every other
hosted tool kind is a named 400.

### Tools in, tools out

Function tools replay and emit by their exact external name; namespaced tools by `namespace/child` (emitted
split as `{namespace, name}`, replayed either split - the flat model-facing name is restored - or whole); custom
freeform tools carry one string `input` returned byte-for-byte (line endings, Unicode, patch boundaries). Tool
call arguments are validated against the request's own registry before an item reaches the wire, and a call to
a tool the request did not offer is a typed failure (`500` non-stream, `response.failed` mid-stream) - never a
published call the client would run blind. Call ids (`call_*`) are preserved on every replay.

### Streaming and the terminal

`stream: true` emits typed Responses SSE: `response.created`, `response.in_progress`, per-item
`output_item.added` / `content_part.added` / `reasoning_text`+`output_text` deltas / `.done` frames with
contiguous `sequence_number` and matching `output_index`/`item_id`/`content_index`, SSE comments while the
request waits in queue or prefills - and exactly one terminal: `response.completed` (with usage),
`response.incomplete` (`incomplete_details.reason: "max_output_tokens"`, the last item marked `incomplete`),
or `response.failed`. No Chat chunk shapes, no `[DONE]` sentinel. Tool items are buffered and published only
complete - a partial call never reaches the wire.

### Errors

Every failure is one JSON object: `{"error": {"message", "type", "param", "code"}}` - `400`
(`invalid_request_error`, `param` names the offending field), `404` (`not_found`), `413` (body over the server's request-size limit
(64 MB default), refused before the handler), `500` (typed engine failure, `server_error`), `503`
(submit refused: shutting down, the queue's post-fatal state, or the image preprocess gate; retryable where
the message says so), and `408` (`api_error` with `x-should-retry: true`): a POST whose body stops arriving
within `request-read-timeout-s` is turned into the retryable timeout before any route runs. The route is never
a bare 404, and a Responses error body is never a Chat-shaped body.

### Limits that are this server's truth

- **No plaintext replay:** reasoning from earlier turns is not re-rendered (the template regenerates blocks for
  kept turns only); conversation context - calls, results, text - is preserved exactly.
- **No encrypted reasoning, no summaries:** the raw `reasoning_text` streams; `summary` is `[]`,
  `encrypted_content` is `null`.
- **No response store, no background mode, no WebSocket, no hosted tool execution.**
- Custom-tool payloads are passed through, not grammar-checked: the server constrains the *shape* it returns,
  not patch syntax (the client validates on apply, as Codex does).

### Raw reasoning (and what it is not)

The Responses endpoint (the wire Codex drives) streams the model's **raw** thinking, not a summary: the reasoning
item carries a `reasoning_text` content part with exactly what the model wrote, `summary` is always the empty
array and `encrypted_content` is always `null` - this server has no summarizer and no key, and invents neither.

`reasoning.effort` maps onto the thinking switch above (the same template settings and the same budget/cap
semantics the Chat route runs - the same functions):

| `reasoning.effort` | what the server does |
|---|---|
| absent | the server's `thinking` default (config), thinking on unless it says off |
| `"none"` | thinking off: no think block, no reasoning item, `reasoning_tokens` 0 |
| `"minimal"`, `"low"` | thinking on, template effort `low` |
| `"medium"` | thinking on, template effort `medium` |
| `"high"`, `"xhigh"` | thinking on, template effort `xhigh` (the model's default depth) |

An explicit effort (anything but `none`) marks thinking as requested: with `max_output_tokens` set, the budget is
`max_output_tokens - max(64, 15%)` - the Chat rule, so the answer always keeps its room. A `reasoning.summary`
value is accepted (it is part of what Codex sends) and changes nothing: the raw text is what streams.

Two honest limits:

- **No plaintext replay.** Thinking from an earlier turn is not re-rendered as reasoning on a later request (the
  template regenerates the block only for the kept assistant turns; `reasoning` input items are validated and
  skipped as display metadata). The conversation context - calls, results, text - is preserved exactly.
- **No encrypted reasoning.** `include: ["reasoning.encrypted_content"]` is accepted; the field is always `null`.

To see the thinking in Codex, set `show_raw_agent_reasoning = true` in `~/.codex/config.toml`.

### Structured output and images

`text.format` lands on the same structured-output machinery the Chat route runs - the same compiler, the same
grammar cache, the same keyword-level refusals:

| `text.format.type` | what happens |
|---|---|
| `"text"` (or absent) | free text |
| `"json_object"` | the answer is grammar-enforced to be one JSON object |
| `"json_schema"` | the answer is grammar-enforced to follow `schema` (with `name`, `strict` accepted) |

Enforcement is the grammar, so it is real: `strict: true` is accepted as the shape's marker, not as a promise -
output is constrained whether the client asked strictly or not. A schema the compiler cannot enforce (unsupported
keywords, schema-typed `additionalProperties`, tuple `items`) is a 400 naming the keyword - never a silent
downgrade to unconstrained output.

Images: a user message's content array may interleave `input_text` and `input_image` parts, and the images reach
the vision pipeline in document order - **when the server runs with a vision tower** (config `vision`). Every
other case is a named refusal, never a silent drop:

- no tower: `400 ... image part: this server runs without a vision tower`
- `image_url` must be a `data:` URL - the server fetches nothing, so `https://` URLs are refused
- `file_id` is refused: the Responses surface here is stateless, there is no file store
- `detail` is validated (`auto`, `low`, `high`)
- images in assistant messages or tool outputs, and audio/video/file parts, are refused as named

The vision path through `/v1/responses` is the Chat route's pipeline verbatim (one shared implementation: same
pad expansion, mRoPE walk, decode memo, preprocess gate); its behavior on a real tower is exercised on
vision-served builds (the host test suite covers the parser rules and the vision-off refusals).

### Trying it: the commands in this section, executed

Against a server on `localhost:5300` (the `qwen3.8-flash-next` id below is what the reference build serves -
`GET /v1/models` answers with the real id):

```bash
# the served model id (name it exactly in Responses requests)
curl -s localhost:5300/v1/models

# a plain turn
curl -s localhost:5300/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "input": "Say hello in five words."
}'

# streaming (typed SSE; Ctrl-C ends the tail, the terminal event closes a full turn)
curl -sN localhost:5300/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next", "stream": true,
  "input": "Count to three."
}'

# a function tool call
curl -s localhost:5300/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next", "input": "List files with exec_command.",
  "tools": [{"type": "function", "name": "exec_command",
             "description": "Run a shell command.",
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"]}}]
}'

# structured output (grammar-enforced)
curl -s localhost:5300/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next", "input": "A one-line haiku about GPUs.",
  "text": {"format": {"type": "json_schema", "name": "poem", "strict": true,
           "schema": {"type": "object", "properties": {"text": {"type": "string"}},
                      "required": ["text"], "additionalProperties": false}}}
}'

# the two named refusals worth seeing once
curl -s localhost:5300/v1/responses/resp_any_id            # 404: stateless, nothing to fetch
curl -s localhost:5300/v1/responses -H 'Content-Type: application/json' \
     -d '{"model": "gpt-6", "input": "hi"}'               # 400: this runs its own model, by name
```

### A runnable Codex profile (`~/.codex/config.toml`)

```toml
[model_providers.strixite]
name = "Strixite (local)"
base_url = "http://127.0.0.1:5300/v1"
wire_api = "responses"

[profiles.strixite]
model = "qwen3.8-flash-next"          # GET /v1/models answers the served id - name that
model_provider = "strixite"
show_raw_agent_reasoning = true       # the raw thinking streams; nothing is summarized here
```

`codex --profile strixite` then drives this endpoint: tools, streamed turns, multi-round calls. (Verified
against codex-cli 0.162.1 request shapes; the live end-to-end run of this profile against the model needs the
Strix Halo runtime - it is documented here, not claimed as executed on a host without weights.)

### What is tested where

The contract suites (`ctest -L contract`) cover the parser table, tool registry, history replay, the non-stream
body, the SSE state machine (its emitted streams validated against the recorded Codex stream fixtures), the
failure matrix and the live route on a scripted backend - all on any build host, no GPU. The same requests
against the real model (turn quality, the live probe, the Codex loop, vision pixels) run on a served Strix
Halo install: the gate B-series logs which of the two every claim rests on.
