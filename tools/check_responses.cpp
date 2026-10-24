// The /v1/responses request contract (for Codex), host only: normalizing real captured Codex requests and their
// negative cases through parse_responses_request, before any generation. No model, no GPU, no tokenizer.json:
// the checks run on JSON in, JSON out. The fixture tools/testdata/codex-request-sample.json is a sanitized request
// captured from codex-cli 0.162.1 (scripts/record_codex.py in the design pack); the inline cases vary it.
//
//   check_responses [codex-request-sample.json]     (default: tools/testdata/codex-request-sample.json)

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/openai.hpp"
#include "serve/responses.hpp"

#include <cstdio>
#include <fstream>
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

const SamplingParams kDefaults;  // generation_config defaults are not the point of these tests
const std::string kModel = "qwen3.8-flash-next";

// The served-model gate: the Responses route must validate the name, not serve any model as the Chat route does.
json::Value load_fixture(const std::string &path) {
    std::ifstream in(path);
    STRIX_CHECK(in.is_open(), "cannot read the Codex request fixture '", path, "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    json::Value v = json::Value::parse(ss.str());
    STRIX_CHECK(v.is_object() && v.find("input") != nullptr, "fixture is not a Responses request body");
    // Drop the fixture's own provenance keys: they are test metadata, not request fields, and the strict
    // top-level key check of parse_responses_request must see exactly what Codex sent.
    json::Value clean = json::Value::object();
    for (const auto &kv : v.as_object("fixture"))
        if (!(kv.first.empty() || kv.first[0] == '_' || kv.first == "codex_version")) clean.set(kv.first, kv.second);
    return clean;
}

const json::Value *field(const json::Value &obj, const char *name) { return obj.find(name); }

// One entry of the parsed request's tool registry, by the external (client-facing, flat) tool name.
const json::Value *registry_entry(const ResponsesRequest &r, const std::string &external) {
    for (const auto &e : r.tool_registry.as_array("tool_registry"))
        if (e.find("external")->as_string("external") == external) return &e;
    return nullptr;
}

// --- the captured Codex request, as sent ---------------------------------------------------------------

void check_real_codex_request(const json::Value &fixture) {
    ResponsesRequest r = parse_responses_request(fixture, kDefaults, kModel);
    ok(r.chat.stream, "the captured request streams");
    ok(r.chat.model == kModel, "the model echoes the served id");

    // input: [developer instructions, developer skills, user environment_context, user prompt] in order. The
    // template's roles are system/user/assistant/tool, so developer items normalize to system, order kept -
    // developer context is never dropped and never reordered.
    const auto &msgs = r.chat.messages.as_array("chat.messages");
    ok(msgs.size() == 4, "4 input items -> 4 chat messages, got ", msgs.size());
    ok(msgs[0].find("role")->as_string("role") == "system" &&
               msgs[1].find("role")->as_string("role") == "system",
       "developer items normalize to system messages in order");
    ok(msgs[2].find("role")->as_string("role") == "user" && msgs[3].find("role")->as_string("role") == "user",
       "user items keep their role and order");
    ok(msgs[3].find("content")->as_string("content").find("one word: hello") != std::string::npos,
       "the user's prompt text reaches the last message intact");

    // tools: 4 functions + 5 namespaced children = 9 model-callable tools; the hosted web_search is not executed
    // by this server and is dropped (disclosed in docs/server.md), never silently passed to the generator.
    ok(r.chat.tools.as_array("chat.tools").size() == 9, "flattened tool count, got ", r.chat.tools.as_array("chat.tools").size());
    ok(registry_entry(r, "web_search") == nullptr, "the hosted web_search tool is not registered");
    const json::Value *exec = registry_entry(r, "exec_command");
    ok(exec != nullptr, "exec_command registered");
    if (exec != nullptr) ok(exec->find("kind")->as_string("kind") == "function", "exec_command kind is function");
    const json::Value *child = registry_entry(r, "multi_agent_v1/close_agent");
    ok(child != nullptr, "namespaced close_agent registered under its namespaced external name");
    if (child != nullptr) {
        ok(child->find("kind")->as_string("kind") == "function", "namespace children are functions");
        ok(child->find("namespace")->as_string("namespace") == "multi_agent_v1",
           "the namespace is recorded for output item restore");
    }

    // The nested Chat schema of exec_command keeps every field Codex sent (STORY-04 acceptance).
    const json::Value *nested = nullptr;
    for (const auto &t : r.chat.tools.as_array("chat.tools"))
        if (t.find("function")->find("name")->as_string("name") == "exec_command") nested = &t;
    ok(nested != nullptr, "exec_command reaches the nested Chat tool form");
    if (nested != nullptr) {
        const json::Value &fn = *nested->find("function");
        ok(fn.find("description")->as_string("description").find("PTY") != std::string::npos,
           "the tool description survives");
        ok(fn.find("parameters")->find("properties")->find("cmd") != nullptr,
           "the parameters schema (cmd) survives");
        ok(fn.find("parameters")->find("properties")->find("sandbox_permissions")->find("enum") != nullptr,
           "deep schema fields (sandbox_permissions.enum) survive");
    }

    // Verified options: benign client metadata (prompt_cache_key, client_metadata, include, parallel_tool_calls)
    // parses; store:false is the stateless mode; tool_choice/reasoning carry through.
    ok(r.tool_choice == "auto", "tool_choice auto");
    ok(r.reasoning_effort == "medium", "reasoning effort medium, got ", r.reasoning_effort);
}

// --- negative cases: path-specific 400s, never 500, never silent ------------------------------------------

void expect_400(const json::Value &body, const std::string &param, const std::string &what) {
    try {
        parse_responses_request(body, kDefaults, kModel);
        ok(false, what, ": expected a 400, parsed fine");
    } catch (const ApiError &e) {
        ok(e.status == 400 && e.param == param, what, ": got status ", e.status, " param '", e.param,
           "' (want 400 / '", param, "')");
    }
}

json::Value with(json::Value base, const char *key, const std::string &raw_value) {
    json::Value v = json::Value::parse(raw_value);
    base.set(key, std::move(v));
    return base;
}

void check_negatives(const json::Value &fixture) {
    expect_400(with(fixture, "model", "\"gpt-6-astra\""), "model", "a model name the server is not");
    expect_400(with(fixture, "store", "true"), "store", "store:true (no response state store here)");
    expect_400(with(fixture, "tool_choice", "\"required\""), "tool_choice",
               "tool_choice:required (not served: the model is never forced)");
    expect_400(with(fixture, "input", "[]"), "input", "an empty input array");
    expect_400(with(fixture, "max_output_tokens", "0"), "max_output_tokens", "max_output_tokens 0");

    json::Value bad_input = json::Value::parse(
        "[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]},"
        "{\"type\":\"item_from_mars\",\"role\":\"user\",\"content\":[]}]");
    expect_400(with(fixture, "input", bad_input.dump()), "input", "an unrecognized input item type");

    json::Value bad_tools = json::Value::parse(
        "[{\"type\":\"function\",\"name\":\"dupe\",\"parameters\":{\"type\":\"object\",\"properties\":{}}},"
        "{\"type\":\"function\",\"name\":\"dupe\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]");
    expect_400(with(fixture, "tools", bad_tools.dump()), "tools", "duplicate tool names");

    json::Value bad_tool = json::Value::parse("[{\"type\":\"function\",\"name\":\"nosc\"}]");
    expect_400(with(fixture, "tools", bad_tool.dump()), "tools", "a function tool without a parameters schema");

    expect_400(with(fixture, "background", "true"), "background", "background:true (no background mode)");

    // benign metadata stays accepted
    ResponsesRequest r = parse_responses_request(with(fixture, "include", "[\"reasoning.encrypted_content\"]"),
                                                 kDefaults, kModel);
    ok(r.chat.stream, "include:[reasoning.encrypted_content] is accepted (raw reasoning; no encrypted content)");

    // tool_choice:none - the generator sees no tools, the registry keeps them for output mapping (STORY-04 rule).
    ResponsesRequest none = parse_responses_request(with(fixture, "tool_choice", "\"none\""), kDefaults, kModel);
    ok(none.chat.tools.is_null(), "tool_choice none: the generator gets no tool definitions");
    ok(none.tool_registry.as_array("tool_registry").size() == 9, "tool_choice none: the registry stays complete");
}

}  // namespace

int main(int argc, char **argv) {
    const std::string path = argc > 1 ? argv[1] : "tools/testdata/codex-request-sample.json";
    try {
        json::Value fixture = load_fixture(path);
        check_real_codex_request(fixture);
        check_negatives(fixture);
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
