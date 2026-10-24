// Native reasoning interoperability for /v1/responses, host only (STORY-09 / gate B9, host half): the effort
// table mapped onto the real thinking settings, the budget/cap semantics IDENTICAL to the Chat path (the same
// functions, driven by the normalized request), the reasoning-only turn, and the two hard invariants on BOTH
// emitters (non-streaming body and SSE): summary is always the empty array, encrypted_content is always null -
// raw reasoning is what streams, nothing is invented. The live-codex half of B9 (show_raw_agent_reasoning on a
// real model) is the GPU-gated story-12 environment; nothing here claims it.
//
//   check_responses_reasoning

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/openai.hpp"
#include "serve/output_parser.hpp"
#include "serve/responses.hpp"
#include "serve/responses_body.hpp"
#include "serve/responses_stream.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

int failures = 0, checks = 0;
template <typename... A>
void ok(bool cond, A &&...a) {
    ++checks;
    if (!cond) {
        ++failures;
        std::ostringstream os;
        (os << ... << a);
        std::printf("FAIL: %s\n", os.str().c_str());
    }
}

const SamplingParams kDefaults;
const std::string kModel = "qwen3.8-flash-next";

ResponsesRequest parse(const std::string &reasoning_json) {
    json::Value b = json::Value::parse("{\"model\":\"" + kModel + "\",\"input\":\"q\"" +
                                       (reasoning_json.empty() ? "" : ",\"reasoning\":" + reasoning_json) + "}");
    return parse_responses_request(b, kDefaults, kModel);
}

bool effort_of(const ResponsesRequest &r, const char *want) {
    const auto &v = r.chat.template_options.reasoning_effort;
    return v.has_value() && *v == want;
}

// --- the effort table (documented in docs/server.md) ----------------------------------------------------------

void check_effort_table() {
    ok(parse("{\"effort\":\"none\"}").chat.template_options.enable_thinking &&
               !*parse("{\"effort\":\"none\"}").chat.template_options.enable_thinking,
       "effort none -> thinking off");
    ok(!parse("{\"effort\":\"none\"}").chat.template_options.reasoning_effort.has_value(),
       "effort none sets no template effort");
    ok(effort_of(parse("{\"effort\":\"minimal\"}"), "low"), "minimal -> low");
    ok(effort_of(parse("{\"effort\":\"low\"}"), "low"), "low -> low");
    ok(effort_of(parse("{\"effort\":\"medium\"}"), "medium"), "medium -> medium");
    ok(effort_of(parse("{\"effort\":\"high\"}"), "xhigh"), "high -> xhigh");
    ok(effort_of(parse("{\"effort\":\"xhigh\"}"), "xhigh"), "xhigh -> xhigh");
    ResponsesRequest plain = parse("");
    ok(!plain.chat.template_options.enable_thinking.has_value() &&
               !plain.chat.template_options.reasoning_effort.has_value(),
       "no reasoning field -> the server's thinking default applies (as Chat)");
    ResponsesRequest with_summary = parse("{\"effort\":\"low\",\"summary\":\"auto\"}");
    ok(effort_of(with_summary, "low") &&
               !with_summary.chat.template_options.enable_thinking.has_value(),
       "summary accepted, does not touch thinking (raw reasoning is what streams)");
    ok(parse("{\"effort\":\"low\"}").reasoning_effort == "low", "the requested effort is kept on the request");
}

// --- budget/cap semantics: the SAME functions the Chat route runs, on the normalized request ---------------------

void check_budget_semantics() {
    // Explicit effort (thinking requested): the Chat answer room is kExplicitAnswerRoom (64).
    ResponsesRequest low = parse("{\"effort\":\"low\"}");
    ok(thinking_requested(low.chat), "an effort names thinking as requested");
    ok(thinking_budget_for(low.chat, 1000) == 1000 - std::max<int64_t>(kExplicitAnswerRoom, 1000 * 15 / 100),
       "budget = max_output_tokens - max(64, 15%) room, got ", thinking_budget_for(low.chat, 1000));
    ok(thinking_budget_for(low.chat, 200) == 200 - kExplicitAnswerRoom, "small caps use the 64-token answer room");
    ok(thinking_budget_for(low.chat, 60) == 0, "no room -> 0 (the engine's guard, as Chat)");
    // No reasoning field: the Chat default room max(1024, 15%).
    ResponsesRequest plain = parse("");
    ok(!thinking_requested(plain.chat), "no effort -> not explicitly requested");
    ok(thinking_budget_for(plain.chat, 4096) == 4096 - 1024, "small cap: the 1024 answer room");
    ok(thinking_budget_for(plain.chat, 16384) == 16384 - 16384 * 15 / 100, "big cap: the 15% answer room");
    // Effort none: thinking off, budget -1 (no think block at all).
    ResponsesRequest off = parse("{\"effort\":\"none\"}");
    ok(thinking_budget_for(off.chat, 1000) == -1, "effort none -> budget -1 (thinking off)");
    ok(!thinking_requested(off.chat), "effort none is not 'requested'");
    // The left-out rule (the Chat route re-renders without the block): identical inputs, identical verdicts.
    ChatRequest chat_like;  // the Chat shape the route builds for a request with no thinking ask
    chat_like.messages = plain.chat.messages;
    ok(thinking_left_out(plain.chat, 512) == thinking_left_out(chat_like, 512),
       "thinking_left_out parity with the Chat path (512 room, no ask)");
    ok(!thinking_left_out(low.chat, 512), "an explicit effort keeps the block even at 512 (Chat rule)");
}

// --- the reasoning-only turn + the two invariants, on BOTH emitters ----------------------------------------------

GenerationResult result(const std::string &finish, int64_t reasoning_toks = 7) {
    GenerationResult r;
    r.finish_reason = finish;
    r.prompt_tokens = 50, r.cached_tokens = 0, r.completion_tokens = 20, r.reasoning_tokens = reasoning_toks;
    return r;
}

void check_reasoning_only_and_invariants() {
    ResponsesRequest rr = parse("");
    // Non-streaming: reasoning-only (no content, no calls) is a full completed response.
    json::Value body = responses_body(rr, "resp_r1", 1, "the whole turn was thinking", "", {}, result("stop"));
    ok(body.find("status")->as_string("status") == "completed", "reasoning-only turn completes");
    const json::Value &out = *body.find("output");
    ok(out.as_array("o").size() == 1 && out.as_array("o")[0].find("type")->as_string("t") == "reasoning",
       "the single item is the completed raw reasoning");
    ok(out.as_array("o")[0].find("content")->as_array("c")[0].find("text")->as_string("x") ==
               "the whole turn was thinking",
       "raw reasoning text, unmodified");
    ok(out.as_array("o")[0].find("summary")->as_array("s").empty(), "body invariant: summary []");
    ok(out.as_array("o")[0].find("encrypted_content")->is_null(), "body invariant: encrypted_content null");
    ok(body.find("usage")->find("output_tokens_details")->find("reasoning_tokens")->as_int("rt") == 7,
       "reasoning tokens reported");

    // Streaming: the same invariants on the lifecycle frames.
    ResponsesStream s("resp_r2", kModel, 1, rr.tool_registry);
    std::string bytes = s.ev_created();
    s.on_reasoning("a long ");
    s.on_reasoning("thought with no answer");
    bytes += s.take() + s.ev_completed(result("stop"));
    size_t seq = 0;
    int deltas = 0;
    bool added = false, item_done = false;
    size_t pos = 0;
    while (pos < bytes.size()) {
        const size_t end = bytes.find("\n\n", pos);
        std::string frame = bytes.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = (end == std::string::npos) ? bytes.size() : end + 2;
        if (frame.rfind("data: ", 0) != 0) continue;
        json::Value e = json::Value::parse(frame.substr(6));
        ok(e.find("sequence_number")->as_int("q") == (long long)seq++, "stream seq contiguous");
        const std::string t = e.find("type")->as_string("t");
        if (t == "response.output_item.added") {
            added = true;
            ok(e.find("item")->find("type")->as_string("t") == "reasoning" &&
                       e.find("item")->find("summary")->as_array("s").empty() &&
                       e.find("item")->find("encrypted_content")->is_null(),
               "stream invariant: added reasoning item is summary [] + encrypted null");
        }
        if (t == "response.reasoning_text.delta") ++deltas;
        if (t == "response.output_item.done") {
            item_done = true;
            const json::Value &item = *e.find("item");
            ok(item.find("summary")->as_array("s").empty() && item.find("encrypted_content")->is_null() &&
                       item.find("content")->as_array("c")[0].find("text")->as_string("x") ==
                               "a long thought with no answer",
               "stream invariant: done item raw, summary [], encrypted null");
        }
        ok(t.find("summary_text") == std::string::npos, "no summary channel on the wire: ", t);
    }
    ok(added && deltas == 2 && item_done, "reasoning streamed added/2 deltas/done");
}

}  // namespace

int main() {
    try {
        check_effort_table();
        check_budget_semantics();
        check_reasoning_only_and_invariants();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
