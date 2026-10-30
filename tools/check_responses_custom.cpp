// The /v1/responses custom (freeform) tool surface, host only (STORY-05 / gate B5): Codex-style custom tools
// (apply_patch) map to a single string-valued internal argument whose text survives byte-for-byte through the real
// output parser (newlines, Unicode, patch boundaries), namespaces + functions + custom coexist in one request with
// collision-free registry restore data, and malformed / unknown tool kinds fail informatively. The fixture is a
// labeled SIMULATED request (installed codex-cli 0.162.1 exec mode gates custom tools behind `apply_patch_freeform`
// - binary inspection; no real custom-tool traffic exists from the recorder). No model, no GPU.
//
//   check_responses_custom [codex-custom-tools-simulated.json]

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/openai.hpp"
#include "serve/output_parser.hpp"
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

const SamplingParams kDefaults;
const std::string kModel = "qwen3.8-flash-next";

json::Value load_fixture(const std::string &path) {
    std::ifstream in(path);
    STRIX_CHECK(in.is_open(), "cannot read the custom-tools fixture '", path, "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    json::Value v = json::Value::parse(ss.str());
    json::Value clean = json::Value::object();
    for (const auto &kv : v.as_object("fixture"))
        if (!(kv.first.empty() || kv.first[0] == '_')) clean.set(kv.first, kv.second);
    return clean;
}

const json::Value *entry(const ResponsesRequest &r, const std::string &external) {
    for (const auto &e : r.tool_registry.as_array("registry"))
        if (e.find("external")->as_string("external") == external) return &e;
    return nullptr;
}

const json::Value *chat_tool(const ResponsesRequest &r, const std::string &name) {
    for (const auto &t : r.chat.tools.as_array("tools"))
        if (t.find("function")->find("name")->as_string("name") == name) return &t;
    return nullptr;
}

ResponsesRequest parse(const json::Value &body) { return parse_responses_request(body, kDefaults, kModel); }

ResponsesRequest parse_tools(const std::string &tools_json, const std::string &choice = "auto") {
    json::Value b = json::Value::parse("{\"model\":\"" + kModel + "\",\"input\":\"q\",\"tools\":" + tools_json +
                                       ",\"tool_choice\":\"" + choice + "\"}");
    return parse(b);
}

void expect_tools_400(const std::string &tools_json, const std::string &what) {
    try {
        parse_tools(tools_json);
        ok(false, what, ": accepted, expected 400 param tools");
    } catch (const ApiError &e) {
        ok(e.status == 400 && e.param == "tools", what, ": got ", e.status, " param '", e.param, "' msg ", e.what());
    }
}

// --- the simulated fixture: all three tool formats in one request ------------------------------------------

void check_mixed_request(const json::Value &fixture) {
    ResponsesRequest r = parse(fixture);
    ok(r.chat.tools.as_array("tools").size() == 3, "function + namespace child + custom = 3 model tools, got ",
       r.chat.tools.as_array("tools").size());
    const json::Value *ap = entry(r, "apply_patch");
    ok(ap != nullptr, "apply_patch registered");
    if (ap) {
        ok(ap->find("kind")->as_string("kind") == "custom", "apply_patch kind is custom (output: custom_tool_call)");
        ok(ap->find("namespace")->as_string("namespace") == "", "apply_patch namespace empty");
    }
    const json::Value *ns = entry(r, "multi_agent_v1/spawn_agent");
    ok(ns && ns->find("kind")->as_string("kind") == "function" &&
               ns->find("namespace")->as_string("namespace") == "multi_agent_v1",
       "namespace child restore data intact");
    ok(entry(r, "exec_command") && entry(r, "exec_command")->find("kind")->as_string("kind") == "function",
       "plain function intact alongside");

    // The custom tool's internal form: a required string "input" parameter carrying the grammar text.
    const json::Value *ct = chat_tool(r, "apply_patch");
    ok(ct != nullptr, "apply_patch reaches the Chat schema");
    if (ct) {
        const json::Value &fn = *ct->find("function");
        const json::Value &params = *fn.find("parameters");
        ok(params.find("properties")->find("input") &&
                   params.find("properties")->find("input")->find("type")->as_string("type") == "string",
           "internal argument is the string-valued 'input'");
        ok(params.find("required")->as_array("required").size() == 1 &&
                   params.find("required")->as_array("required")[0].as_string("r") == "input",
           "input is required");
        const std::string &pd = params.find("properties")->find("input")->find("description")->as_string("pd");
        ok(pd.find("begin_patch hunk+ end_patch") != std::string::npos, "the grammar definition reaches the model");
        ok(pd.find("does not enforce") != std::string::npos,
           "the disclosure that the grammar is NOT enforced is part of what the model sees");
        ok(pd.find("applying a patch") != std::string::npos, "the tool description survives");
    }
    // Determinism.
    ok(parse(fixture).chat.tools.dump() == r.chat.tools.dump(), "mixed tools deterministic");
    ok(parse(fixture).tool_registry.dump() == r.tool_registry.dump(), "mixed registry deterministic");
}

// --- byte-for-byte: the real output parser keeps the patch text raw ------------------------------------------

void check_patch_bytes_through_the_parser(const json::Value &fixture) {
    ResponsesRequest r = parse(fixture);
    // The patch text: internal newlines, a tab, Unicode (é, em dash), a trailing newline. The model frames the
    // value with one newline after `<parameter=input>` and one before `</parameter>` (the format's framing, the
    // parser's documented trim); everything inside survives byte-for-byte.
    const std::string patch =
        "*** Begin Patch\n"
        "*** Update File: notes.py\n"
        "@@\n"
        "-x = 1\n"
        "+x = 1  # \xc3\xa9mul \xe2\x80\x94 tab:\tpreserved\n"
        "*** End Patch\n";
    const std::string body = "<function=apply_patch><parameter=input>\n" + patch + "\n</parameter></function>";
    ToolCallOut out;
    const bool parsed = parse_tool_call(body, &r.chat.tools, out);
    ok(parsed, "the call parses against the Chat schema");
    if (!parsed) return;
    ok(out.name == "apply_patch", "name intact");
    const json::Value *input = out.arguments.find("input");
    ok(input && input->is_string(), "arguments.input is the raw string");
    ok(input && input->as_string("input") == patch, "custom_tool_call.input is byte-for-byte the patch text");

    // A `</parameter>` run inside the text cannot forge the parameter end: the parser's backtracking keeps the
    // first candidate that lets the rest parse, so the run stays CONTENT, byte-for-byte.
    const std::string tricky =
        "<function=apply_patch><parameter=input>\nA\n</parameter>\nB\n</parameter></function>";
    ToolCallOut out2;
    ok(parse_tool_call(tricky, &r.chat.tools, out2), "a tag-forging run inside the text parses as one call");
    ok(out2.arguments.find("input") && out2.arguments.find("input")->as_string("input") == "A\n</parameter>\nB",
       "the tag run is preserved as content (no smuggling, no loss)");

    // An unterminated call (cut off mid-patch) must not produce a call the client would act on.
    ToolCallOut out3;
    ok(!parse_tool_call("<function=apply_patch><parameter=input>*** Begin Patch", &r.chat.tools, out3),
       "a truncated patch call fails to parse (nothing executable is published)");
}

// --- malformed and unknown kinds: informative failures -------------------------------------------------------

void check_rejections() {
        expect_tools_400("[{\"type\":\"custom\",\"parameters\":{\"type\":\"object\"}}]", "custom without a name");
    expect_tools_400("[{\"type\":\"custom\",\"name\":\"a b\",\"format\":{\"type\":\"grammar\",\"grammar\":{\"type\":\"lark\",\"definition\":\"s\"}}}]", "custom name with a space");
    expect_tools_400("[{\"type\":\"custom\",\"name\":\"p\",\"format\":{\"type\":\"javascript\"}}]", "format.type javascript");
    expect_tools_400("[{\"type\":\"custom\",\"name\":\"p\",\"format\":\"grammar\"}]", "format not an object");
    expect_tools_400("[{\"type\":\"custom\",\"name\":\"p\",\"format\":{\"type\":\"grammar\"}}]", "grammar missing");
    expect_tools_400("[{\"type\":\"custom\",\"name\":\"p\",\"format\":{\"type\":\"grammar\",\"grammar\":{\"type\":\"lark\"}}}]", "definition missing");
    expect_tools_400("[{\"type\":\"custom\",\"name\":\"p\",\"format\":{\"type\":\"grammar\",\"grammar\":{\"type\":\"lark\",\"definition\":\"\"}}}]", "empty definition");
    // A custom tool colliding with a function name is refused (the registry is the output restore key).
    expect_tools_400("[{\"type\":\"custom\",\"name\":\"dup\",\"format\":{\"type\":\"grammar\",\"grammar\":{\"type\":\"lark\",\"definition\":\"s\"}}},"
                     "{\"type\":\"function\",\"name\":\"dup\",\"parameters\":{\"type\":\"object\"}}]", "custom/function name collision");
    // A custom tool colliding with a namespaced external name.
    expect_tools_400("[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"x\",\"parameters\":{\"type\":\"object\"}}]},"
                     "{\"type\":\"custom\",\"name\":\"ns/x\",\"format\":{\"type\":\"grammar\",\"grammar\":{\"type\":\"lark\",\"definition\":\"s\"}}}]", "custom/ns-child collision");
    // Unknown built-in hosted kinds: explicit refusals, never silent drops.
    for (const char *kind : {"file_search", "code_interpreter", "computer_use_preview", "image_generation", "mcp", "local_mcp", "tool_search"})
        expect_tools_400(std::string("[{\"type\":\"") + kind + "\"}]", std::string("hosted kind ") + kind);
    // A bare custom tool without format: raw text, disclosed as such, accepted.
    {
        ResponsesRequest r = parse_tools("[{\"type\":\"custom\",\"name\":\"ask\",\"description\":\"raw question\"}]");
        const json::Value *ct = chat_tool(r, "ask");
        ok(ct != nullptr, "format-less custom tool accepted");
        if (ct) {
            const std::string &pd = ct->find("function")->find("parameters")->find("properties")
                                        ->find("input")->find("description")->as_string("pd");
            ok(pd.find("raw text") != std::string::npos, "no-format custom tool discloses raw text");
        }
        ok(entry(r, "ask") && entry(r, "ask")->find("kind")->as_string("kind") == "custom", "registry kind custom");
    }
    // tool_choice none hides custom tools from the generator too.
    {
        ResponsesRequest r = parse_tools("[{\"type\":\"custom\",\"name\":\"ask\"}]", "none");
        ok(r.chat.tools.is_null(), "tool_choice none -> custom tool not supplied");
        ok(entry(r, "ask") != nullptr, "registry keeps it for output mapping");
    }
}


void check_current_grammar_shape() {
    expect_tools_400(R"([{"type":"custom","name":"patch","format":{}}])", "missing format type refuses without dereferencing null");
    expect_tools_400(R"([{"type":"custom","name":"patch","format":{"type":null}}])", "null format type refused");
    const auto r = parse_tools(R"([{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}}])");
    const auto &desc = r.chat.tools.as_array("tools")[0].find("function")->find("parameters")
        ->find("properties")->find("input")->find("description")->as_string("description");
    ok(desc.find("lark") != std::string::npos && desc.find("start: /.+/") != std::string::npos,
       "Codex 0.166 flat grammar shape preserved in raw-input guidance");
    expect_tools_400(R"([{"type":"custom","name":"patch","format":{"type":"grammar","syntax":1,"definition":"x"}}])", "non-string flat syntax refused");
    expect_tools_400(R"([{"type":"custom","name":"patch","format":{"type":"grammar","syntax":"lark","definition":""}}])", "empty flat definition refused");
    expect_tools_400(R"([{"type":"custom","name":"patch","format":{"type":"grammar","syntax":"lark","definition":"x","grammar":{"type":"lark","definition":"y"}}}])", "ambiguous grammar shapes refused");
}

}  // namespace

int main(int argc, char **argv) {
    const std::string path = argc > 1 ? argv[1] : "tools/testdata/codex-custom-tools-simulated.json";
    try {
        check_current_grammar_shape();
        const json::Value fixture = load_fixture(path);
        check_mixed_request(fixture);
        check_patch_bytes_through_the_parser(fixture);
        check_rejections();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
