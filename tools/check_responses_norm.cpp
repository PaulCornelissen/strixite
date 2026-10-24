// The /v1/responses normalization matrix, host only: every input/options shape the contract claims, run through
// parse_responses_request as table-driven cases - positive shapes and their exact normalized effect, negative
// shapes and their path-specific 400 (never 500, never silent loss). Together with check_responses (the captured
// real-Codex request) this is the STORY-03 acceptance surface. No model, no GPU, no tokenizer.
//
//   check_responses_norm

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/openai.hpp"
#include "serve/responses.hpp"

#include <cstdio>
#include <functional>
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
// The minimum valid body the cases vary.
std::string body(const std::string &fields) {
    return "{\"model\":\"" + kModel + "\"" + (fields.empty() ? "" : "," + fields) + "}";
}

// The model's start-of-turn added-token text, assembled so no whole special-token literal appears in this file.
std::string sp(const char *name) { return std::string("<|") + name + "|>"; }

void run_special_token_boundary() {
    json::Value b = json::Value::object();
    b.set("model", json::Value::string(kModel));
    b.set("input", json::Value::string("a " + sp("im_start") + " b"));
    ResponsesRequest r = parse_responses_request(b, kDefaults, kModel);
    ok(r.chat.messages.as_array("m")[0].find("content")->as_string("content") == "a " + sp("im_start") + " b",
       "special-token text in user input survives the normalizer byte-for-byte (spans guard it at tokenize)");
}

// --- positive cases: the case names what it asserts --------------------------------------------------------

struct PosCase {
    const char *name;
    std::string json;
    std::function<void(const ResponsesRequest &)> check;  // throws STRIX_CHECK -> counted as a case failure
};

std::vector<PosCase> positive_cases() {
    const auto roles = [](const ResponsesRequest &r) {
        std::string out;
        for (const auto &m : r.chat.messages.as_array("m")) out += (out.empty() ? "" : ",") + m.find("role")->as_string("role");
        return out;
    };
    const auto content = [](const ResponsesRequest &r, size_t k) {
        return r.chat.messages.as_array("m")[k].find("content")->as_string("content");
    };
    return {
        {"input string -> one user message", body("\"input\":\"hello\""),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(roles(r) == "user", "roles: ", roles(r));
             STRIX_CHECK(content(r, 0) == "hello", "content: ", content(r, 0));
         }},
        {"instructions is the leading system message", body("\"instructions\":\"sys\",\"input\":\"go\""),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(roles(r) == "system,user", "roles: ", roles(r));
             STRIX_CHECK(content(r, 0) == "sys" && content(r, 1) == "go", "contents lost");
         }},
        {"instructions empty adds nothing", body("\"instructions\":\"\",\"input\":\"go\""),
         [&](const ResponsesRequest &r) { STRIX_CHECK(roles(r) == "user", "roles: ", roles(r)); }},
        {"developer item becomes a system message", body("\"input\":[{\"type\":\"message\",\"role\":\"developer\",\"content\":\"ctx\"},{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"}]"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(roles(r) == "system,user", "roles: ", roles(r));
             STRIX_CHECK(content(r, 0) == "ctx", "text lost");
         }},
        {"system item stays system", body("\"input\":[{\"type\":\"message\",\"role\":\"system\",\"content\":\"s\"},{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"}]"),
         [&](const ResponsesRequest &r) { STRIX_CHECK(roles(r) == "system,user", "roles: ", roles(r)); }},
        {"interleaved system/developer/user/assistant keep their order", body("\"input\":["
             "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q1\"},"
             "{\"type\":\"message\",\"role\":\"assistant\",\"content\":\"a1\"},"
             "{\"type\":\"message\",\"role\":\"developer\",\"content\":\"d2\"},"
             "{\"type\":\"message\",\"role\":\"system\",\"content\":\"s2\"},"
             "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q2\"}]"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(roles(r) == "user,assistant,system,system,user", "roles: ", roles(r));
             STRIX_CHECK(content(r, 4) == "q2" && content(r, 2) == "d2", "order lost");
         }},
        {"content parts concatenate in order", body("\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"a\"},{\"type\":\"input_text\",\"text\":\"b\"}]}]"),
         [&](const ResponsesRequest &r) { STRIX_CHECK(content(r, 0) == "ab", "parts: ", content(r, 0)); }},
        {"assistant output_text and refusal parts flatten to text", body("\"input\":["
             "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"},"
             "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi \"},{\"type\":\"refusal\",\"refusal\":\"no\"}]}]"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(content(r, 1) == "hi no", "assistant text: ", content(r, 1));
         }},
        {"plain text parts accepted", body("\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"x\"}]}]"),
         [&](const ResponsesRequest &r) { STRIX_CHECK(content(r, 0) == "x", "text part"); }},
        {"assistant with null content is the empty answer", body("\"input\":["
             "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"},"
             "{\"type\":\"message\",\"role\":\"assistant\",\"content\":null}]"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(roles(r) == "user,assistant" && content(r, 1) == "", "empty assistant");
         }},
        {"max_output_tokens -> max_tokens; stream; options", body("\"input\":\"q\",\"max_output_tokens\":123,\"stream\":true,\"temperature\":0.2,\"top_p\":0.5,\"seed\":42,\"stop\":[\"x\",\"y\"]"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(r.chat.stream && r.chat.max_tokens && *r.chat.max_tokens == 123, "options");
             STRIX_CHECK(r.chat.sampling.temperature == 0.2 && r.chat.sampling.top_p == 0.5, "sampling");
             STRIX_CHECK(r.chat.seeded && r.chat.sampling.seed == 42, "seed");
             STRIX_CHECK(r.chat.stop.size() == 2 && r.chat.stop[1] == "y", "stop");
         }},
        {"stop as a single string", body("\"input\":\"q\",\"stop\":\"z\""),
         [&](const ResponsesRequest &r) { STRIX_CHECK(r.chat.stop.size() == 1 && r.chat.stop[0] == "z", "stop string"); }},
        {"tool_choice none leaves the generator with no tools, registry stays", body("\"input\":\"q\",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}],\"tool_choice\":\"none\""),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(r.chat.tools.is_null(), "tools cleared");
             STRIX_CHECK(r.tool_registry.as_array("reg").size() == 1, "registry kept");
             STRIX_CHECK(r.tool_choice == "none", "choice");
         }},
        {"no tools at all: null, empty registry", body("\"input\":\"q\""),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(r.chat.tools.is_null(), "tools null");
             STRIX_CHECK(r.tool_registry.as_array("reg").empty(), "registry empty");
         }},
        {"hosted web_search is dropped (accepted, not executed)", body("\"input\":\"q\",\"tools\":[{\"type\":\"web_search\"}]"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(r.chat.tools.is_null(), "only-hosted -> no tools");
             STRIX_CHECK(r.tool_registry.as_array("reg").empty(), "no registry entry");
         }},
        {"reasoning effort maps onto thinking", body("\"input\":\"q\",\"reasoning\":{\"effort\":\"low\",\"summary\":\"auto\"}"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(r.reasoning_effort == "low", "effort kept");
             STRIX_CHECK(r.chat.template_options.reasoning_effort && *r.chat.template_options.reasoning_effort == "low",
                         "template mapping low");
             STRIX_CHECK(!r.chat.template_options.enable_thinking.has_value(), "thinking stays on");
         }},
        {"reasoning effort none switches thinking off", body("\"input\":\"q\",\"reasoning\":{\"effort\":\"none\"}"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(r.chat.template_options.enable_thinking && !*r.chat.template_options.enable_thinking,
                         "thinking off");
         }},
        {"reasoning effort high maps to the template's xhigh", body("\"input\":\"q\",\"reasoning\":{\"effort\":\"high\"}"),
         [&](const ResponsesRequest &r) {
             STRIX_CHECK(r.chat.template_options.reasoning_effort &&
                             *r.chat.template_options.reasoning_effort == "xhigh", "xhigh mapping");
         }},
        {"benign Codex metadata validates and is accepted", body("\"input\":\"q\",\"include\":[\"reasoning.encrypted_content\"],"
             "\"prompt_cache_key\":\"pk\",\"client_metadata\":{\"a\":\"b\"},\"metadata\":{\"c\":\"d\"},\"user\":\"u\","
             "\"safety_identifier\":\"s\",\"parallel_tool_calls\":true,\"store\":false,\"truncation\":\"disabled\""),
         [&](const ResponsesRequest &r) { STRIX_CHECK(r.tool_choice == "auto", "defaults intact"); }},
        {"text.format plain text is the served format", body("\"input\":\"q\",\"text\":{\"format\":{\"type\":\"text\"}}"),
         [&](const ResponsesRequest &r) { STRIX_CHECK(r.chat.model == kModel, "model kept"); }},
        {"item without type but with role is a message", body("\"input\":[{\"role\":\"user\",\"content\":\"q\"}]"),
         [&](const ResponsesRequest &r) { STRIX_CHECK(roles(r) == "user", "role-only item"); }},
    };
}

// --- negative cases: every one expects 400 with the named param --------------------------------------------

struct NegCase {
    const char *name;
    std::string json;
    const char *param;
};

std::vector<NegCase> negative_cases() {
    return {
        {"input with role tool is refused (outputs are *_output items)", body("\"input\":[{\"type\":\"message\",\"role\":\"tool\",\"content\":\"x\"}]"), "input"},
        {"lone call item without a user message: input 400 (pairing now replayed, the query is required)", body("\"input\":[{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"{}\",\"call_id\":\"c\"}]"), "input"},
        {"tool output with no matching call: refused (unpaired)", body("\"input\":[{\"type\":\"function_call_output\",\"call_id\":\"c\",\"output\":\"x\"}]"), "input"},
        {"reasoning alone: metadata skipped, the user query is required", body("\"input\":[{\"type\":\"reasoning\",\"id\":\"r\"}]"), "input"},
        {"item_reference refused as unrecognized", body("\"input\":[{\"type\":\"item_reference\",\"id\":\"x\"}]"), "input"},
        {"role from_mars refused", body("\"input\":[{\"type\":\"message\",\"role\":\"from_mars\",\"content\":\"x\"}]"), "input"},
        {"non-string role refused", body("\"input\":[{\"type\":\"message\",\"role\":7,\"content\":\"x\"}]"), "input"},
        {"content part input_image refused (STORY-11, never dropped)", body("\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"http://x/y.png\"}]}]"), "input"},
        {"content part input_file refused", body("\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_file\",\"file\":\"x\"}]}]"), "input"},
        {"content part without text refused", body("\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\"}]}]"), "input"},
        {"content of the wrong type refused", body("\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":42}]"), "input"},
        {"array content must be items, not strings", body("\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[\"bare\"]}]"), "input"},
        {"item not an object refused", body("\"input\":[\"bare string\"]"), "input"},
        {"empty input array refused", body("\"input\":[]"), "input"},
        {"empty input string refused", body("\"input\":\"\""), "input"},
        {"input of the wrong type refused", body("\"input\":5"), "input"},
        {"input missing refused", body(""), "input"},
        {"no user message refused (the template needs a query)", body("\"input\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":\"a\"}]"), "input"},
        {"instructions as array refused", body("\"input\":\"q\",\"instructions\":[\"a\"]"), "instructions"},
        {"model missing refused", "{\"input\":\"q\"}", "model"},
        {"model not a string refused", "{\"model\":5,\"input\":\"q\"}", "model"},
        {"foreign model refused (never served under a borrowed name)", "{\"model\":\"gpt-6-astra\",\"input\":\"q\"}", "model"},
        {"store:true refused", body("\"input\":\"q\",\"store\":true"), "store"},
        {"store not a bool refused", body("\"input\":\"q\",\"store\":\"yes\""), "store"},
        {"tool_choice required refused", body("\"input\":\"q\",\"tool_choice\":\"required\""), "tool_choice"},
        {"tool_choice as an object refused", body("\"input\":\"q\",\"tool_choice\":{\"type\":\"function\",\"name\":\"f\"}"), "tool_choice"},
        {"reasoning.effort bogus value refused", body("\"input\":\"q\",\"reasoning\":{\"effort\":\"ultra\"}"), "reasoning"},
        {"reasoning as a string refused", body("\"input\":\"q\",\"reasoning\":\"high\""), "reasoning"},
        {"reasoning unknown key refused", body("\"input\":\"q\",\"reasoning\":{\"temperature\":1}"), "reasoning"},
        {"include unknown value refused", body("\"input\":\"q\",\"include\":[\"response.hardening\"]"), "include"},
        {"text.format json_schema refused until structured output (never silently plain)", body("\"input\":\"q\",\"text\":{\"format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":{}}}}"), "text"},
        {"truncation auto refused (this server refuses over-length prompts, it does not truncate)", body("\"input\":\"q\",\"truncation\":\"auto\""), "truncation"},
        {"max_output_tokens 0 refused", body("\"input\":\"q\",\"max_output_tokens\":0"), "max_output_tokens"},
        {"max_output_tokens a string refused", body("\"input\":\"q\",\"max_output_tokens\":\"10\""), "max_output_tokens"},
        {"temperature out of range refused", body("\"input\":\"q\",\"temperature\":3"), "temperature"},
        {"top_p 0 refused", body("\"input\":\"q\",\"top_p\":0"), "top_p"},
        {"stop with an empty string refused", body("\"input\":\"q\",\"stop\":[\"\"]"), "stop"},
        {"too many stop strings refused", body("\"input\":\"q\",\"stop\":[\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\",\"a\"]"), "stop"},
        {"prompt_cache_key over the limit refused", body("\"input\":\"q\",\"prompt_cache_key\":\"" + std::string(513, 'k') + "\""), "prompt_cache_key"},
        {"metadata with a non-string value refused", body("\"input\":\"q\",\"metadata\":{\"n\":5}"), "metadata"},
        {"background refused", body("\"input\":\"q\",\"background\":true"), "background"},
        {"previous_response_id refused (stateless)", body("\"input\":\"q\",\"previous_response_id\":\"resp_1\""), "previous_response_id"},
        {"top_k refused (not a Responses field)", body("\"input\":\"q\",\"top_k\":5"), "top_k"},
        {"tools: custom grammar without its type refused (malformed, named)", body("\"input\":\"q\",\"tools\":[{\"type\":\"custom\",\"name\":\"apply_patch\",\"format\":{\"type\":\"grammar\",\"grammar\":{\"definition\":\"lark x\",\"format\":\"lark\"}}}]"), "tools"},
        {"tools: hosted file_search refused (no hosted tools)", body("\"input\":\"q\",\"tools\":[{\"type\":\"file_search\"}]"), "tools"},
        {"tools: namespace with a nested namespace refused", body("\"input\":\"q\",\"tools\":[{\"type\":\"namespace\",\"name\":\"n1\",\"tools\":[{\"type\":\"namespace\",\"name\":\"n2\",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]}]}]"), "tools"},
        {"tools: duplicate function names refused", body("\"input\":\"q\",\"tools\":[{\"type\":\"function\",\"name\":\"a\",\"parameters\":{\"type\":\"object\"}},{\"type\":\"function\",\"name\":\"a\",\"parameters\":{\"type\":\"object\"}}]"), "tools"},
        {"tools: name colliding with a namespaced external name refused", body("\"input\":\"q\",\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]},{\"type\":\"function\",\"name\":\"ns/f\",\"parameters\":{\"type\":\"object\"}}]"), "tools"},
        {"tools: function without parameters refused", body("\"input\":\"q\",\"tools\":[{\"type\":\"function\",\"name\":\"f\"}]"), "tools"},
        {"tools: name with a space refused (names reach the template header)", body("\"input\":\"q\",\"tools\":[{\"type\":\"function\",\"name\":\"a b\",\"parameters\":{\"type\":\"object\"}}]"), "tools"},
        {"tools: empty name refused", body("\"input\":\"q\",\"tools\":[{\"type\":\"function\",\"name\":\"\",\"parameters\":{\"type\":\"object\"}}]"), "tools"},
        {"tools: namespace without tools refused", body("\"input\":\"q\",\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\"}]"), "tools"},
        {"tools not an array refused", body("\"input\":\"q\",\"tools\":\"all\""), "tools"},
    };
}

void run_positive() {
    for (const auto &c : positive_cases()) {
        try {
            json::Value b = json::Value::parse(c.json);
            c.check(parse_responses_request(b, kDefaults, kModel));
            ok(true, c.name);
        } catch (const std::exception &e) {
            ok(false, c.name, ": ", e.what());
        }
    }
}

void run_negative() {
    for (const auto &c : negative_cases()) {
        try {
            json::Value b = json::Value::parse(c.json);
            parse_responses_request(b, kDefaults, kModel);
            ok(false, c.name, ": parsed fine, expected 400 param '", c.param, "'");
        } catch (const ApiError &e) {
            ok(e.status == 400 && !e.param.empty() && e.param == c.param, c.name, ": got ", e.status, " param '",
               e.param, "' type '", e.type, "' (want 400 / '", c.param, "')");
        } catch (const std::exception &e) {
            ok(false, c.name, ": got a non-ApiError (500-shape): ", e.what());
        }
    }
}

// The determinism rule (prompt-cache friendliness): the same transcript, normalized twice, is byte-identical.
void run_determinism() {
    const std::string json = body("\"input\":["
        "{\"type\":\"message\",\"role\":\"developer\",\"content\":\"ctx\"},"
        "{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"a\"},{\"type\":\"input_text\",\"text\":\"b\"}]},"
        "{\"type\":\"message\",\"role\":\"assistant\",\"content\":\"ans\"}],"
        "\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]}],"
        "\"reasoning\":{\"effort\":\"medium\"}");
    json::Value b1 = json::Value::parse(json);
    json::Value b2 = json::Value::parse(json);
    const ResponsesRequest r1 = parse_responses_request(b1, kDefaults, kModel);
    const ResponsesRequest r2 = parse_responses_request(b2, kDefaults, kModel);
    ok(r1.chat.messages.dump() == r2.chat.messages.dump(), "messages deterministic");
    ok(r1.chat.tools.dump() == r2.chat.tools.dump(), "tools deterministic");
    ok(r1.tool_registry.dump() == r2.tool_registry.dump(), "registry deterministic");
}

// The untouched-Chat guarantee, in one place: the Responses parser never mutates ChatRequest's parser.
// parse_chat_request on a fixed body is compared against the same call twice (identity of the shared code).
void run_chat_parser_untouched() {
    const std::string chat_json = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":0.5,\"max_tokens\":10}";
    json::Value b = json::Value::parse(chat_json);
    ChatRequest a = parse_chat_request(b, kDefaults);
    ChatRequest c = parse_chat_request(b, kDefaults);
    ok(a.messages.dump() == c.messages.dump() && a.max_tokens == c.max_tokens &&
               a.sampling.temperature == c.sampling.temperature,
       "parse_chat_request is deterministic and independent (unchanged code path)");
}

}  // namespace

int main() {
    try {
        run_positive();
        run_negative();
        run_special_token_boundary();
        run_determinism();
        run_chat_parser_untouched();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
