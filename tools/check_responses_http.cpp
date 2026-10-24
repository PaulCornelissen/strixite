// The /v1/responses HTTP surface on a running HttpServer, host only: a scripted LmBackend and a synthetic
// byte-level tokenizer (all 256 bytes, no merges, the specials the engine looks up) stand in for the model, so the
// route contract - typed 400s, 501 while generation is unwired, never 404 for POST - and the untouched Chat route
// are proven end to end without a GPU. This is a route-contract test, not model behavior: nothing here shows what
// the real model answers.
//
// The special token texts are built from fragments (sp("im_start") == the model's start-of-turn added token):
// whole special-token literals must not appear in this file's text.
//
//   check_responses_http [tokenizer.json]   (default: writes a synthetic tokenizer.json to a temp dir)

#include "common/check.hpp"
#include "serve/engine.hpp"
#include "serve/host_buffer.hpp"
#include "serve/http_server.hpp"
#include "serve/json.hpp"
#include "serve/tokenizer.hpp"

#include <httplib.h>

#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <chrono>
#include <thread>
#include <unistd.h>
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

std::string jstr(std::string_view s) {
    std::string out;
    json::append_quoted(out, s);
    return out;
}

std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | cp >> 6); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xE0 | cp >> 12); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
}

std::string sp(const char *name) { return std::string("<|") + name + "|>"; }    // an added-token text
std::string tag(const char *name) { return std::string("<") + name + ">"; }     // a parser tag text
std::string tagc(const char *name) { return std::string("<") + "/" + name + ">"; }  // its closing tag

// The synthetic tokenizer.json: the loader's contract is structural (serve/tokenizer.cpp), not checkpoint-sized -
// the 256 byte-level code points as the vocab (GPT-2's printable map, the loader's own byte_to_cp walk), empty
// merges, and the added tokens the engine and OutputParser look up. BPE is then one token per byte of text.
const char *kPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::vector<std::string> specials() {
    return {sp("im_start"), sp("im_end"), sp("endoftext"), sp("im_sep"), tag("tool_call"), tagc("tool_call"),
            tagc("think"),  sp("endofprompt"), sp("vision_start"), sp("image_pad"), sp("vision_end")};
}

std::string synthetic_tokenizer_json() {
    uint32_t next = 256;
    uint32_t map256[256];
    for (uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        map256[b] = printable ? b : next++;
    }
    std::string out =
        "{\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"byte_fallback\":false,\"ignore_merges\":false,"
        "\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\",\"dropout\":null,\"vocab\":{";
    for (uint32_t b = 0; b < 256; ++b) {
        if (b) out += ",";
        out += jstr(cp_to_utf8(map256[b])) + ":" + std::to_string(b);
    }
    out += "},\"merges\":[]},\"normalizer\":{\"type\":\"NFC\"},"
           "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":";
    out += jstr(kPattern);
    out += "},\"behavior\":\"Isolated\",\"invert\":false},"
           "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false}]},"
           "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false},\"added_tokens\":[";
    const auto &added = specials();
    for (size_t k = 0; k < added.size(); ++k) {
        if (k) out += ",";
        out += "{\"content\":" + jstr(added[k]) + ",\"id\":" + std::to_string(256 + (int)k) +
               ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false}";
    }
    out += "]}";
    return out;
}

// --- the scripted backend -----------------------------------------------------------------------------
// Answers every request with the same tokens (reasoning, the think close, " Hello world", the end of turn); the
// engine asks only for the next row, so "next scripted token" is the whole model for this contract test.

class ScriptedLmBackend : public LmBackend {
public:
    ScriptedLmBackend(int32_t vocab, std::vector<int32_t> script, int32_t im_end)
        : vocab_(vocab), script_(std::move(script)), im_end_(im_end) {}

    int64_t capacity() const override { return 8192; }
    int64_t max_chunk() const override { return 512; }
    int64_t logits_row() const override { return vocab_; }
    int64_t pos() const override { return pos_; }
    void reset() override { pos_ = 0; at_ = 0; }
    std::vector<float> forward(const std::vector<int32_t> &ids, bool want_logits) override {
        pos_ += (int64_t)ids.size();
        if (!want_logits) return {};
        // The script answers every request of the session: replay it when spent (a resumed request re-opens
        // the think block and expects the full answer again); the end-of-turn token ends each turn.
        if (at_ >= (int64_t)script_.size()) at_ = 0;
        const int32_t tok = script_[at_++];
        std::vector<float> row((size_t)vocab_, 0.0f);
        row[(size_t)tok] = 10.0f;
        return row;
    }
    void save_snapshot(int slot) override {
        snap_[slot] = pos_;
        if (slot == kTurnSlot) at_ = 0;  // the prompt's turn snapshot: a new request starts - replay the script
    }
    int64_t snapshot_pos(int slot) const override { return snap_[slot]; }
    bool can_restore_snapshot(int slot) const override { return snap_[slot] >= 0 && snap_[slot] <= pos_; }
    void restore_snapshot(int slot) override {
        at_ = 0;  // a restore precedes a new request's suffix prefill: replay the script for it
        if (can_restore_snapshot(slot)) pos_ = snap_[slot];
    }
    void export_snapshot(int, HostBuffer &out, int64_t) override { out.resize(0); }
    uint64_t snapshot_bytes(int, int64_t) const override { return 0; }
    void import_state(const HostBuffer &, int64_t n, int64_t) override { pos_ = n; }
    std::string state_fingerprint() const override { return "scripted-host-test"; }
    std::string describe() const override { return "ScriptedLmBackend(host test)"; }
    // Swap the scripted answer (the route test drives text and tool-call turns on one backend).
    void set_script(std::vector<int32_t> script) {
        std::lock_guard<std::mutex> lk(mu_);
        script_ = std::move(script);
        at_ = 0;
    }

private:
    int32_t vocab_, im_end_;
    std::vector<int32_t> script_;
    int64_t at_ = 0, pos_ = 0;
    std::mutex mu_;
    int64_t snap_[4] = {-1, -1, -1, -1};
};

// --- the checks -----------------------------------------------------------------------------------------

// The error field of a JSON response body (null Value when the body is not an error object).
json::Value err_field(const httplib::Response &res, const char *key) {
    try {
        json::Value body = json::Value::parse(res.body);
        const json::Value *e = body.find("error");
        if (const json::Value *f = e ? e->find(key) : nullptr) return *f;
    } catch (const Error &) {
    }
    return json::Value();
}

void check_routes(httplib::Client &cli, const std::string &model_id, const Tokenizer &tok, ScriptedLmBackend &be) {
    // The text answer (default) and the tool-call answer, replayed per request; scripts swap between cases.
    std::vector<int32_t> text_script;
    for (char c : std::string_view("thinking\n")) text_script.push_back(tok.byte_token((uint8_t)c));
    text_script.push_back(tok.id_of(tagc("think").c_str()));
    for (char c : std::string_view(" Hello world")) text_script.push_back(tok.byte_token((uint8_t)c));
    const int32_t im_end = tok.id_of(sp("im_end").c_str());
    text_script.push_back(im_end);
    std::vector<int32_t> tool_script;
    tool_script.push_back(tok.id_of(tag("tool_call").c_str()));
    for (char c : std::string_view("\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n"))
        tool_script.push_back(tok.byte_token((uint8_t)c));
    tool_script.push_back(tok.id_of(tagc("tool_call").c_str()));
    tool_script.push_back(im_end);
    {  // GET on the route: not a registered method - the documented 404 (Codex sends POST).
        auto res = cli.Get("/v1/responses");
        ok(res && res->status == 404, "GET /v1/responses -> 404, got ", res ? std::to_string(res->status) : "none");
    }
    {  // POST, no model: 400 naming the field - never 404, never a Chat-shaped body.
        auto res = cli.Post("/v1/responses", "{\"input\":\"hi\"}", "application/json");
        ok(res && res->status == 400, "POST no model -> 400, got ", res ? std::to_string(res->status) : "none");
        ok(err_field(*res, "param").as_string("param") == "model", "the 400 names param 'model'");
    }
    {  // POST, a model this server is not: the Responses route refuses the name (Chat serves any).
        auto res = cli.Post("/v1/responses", "{\"model\":\"gpt-6-astra\",\"input\":\"hi\"}", "application/json");
        ok(res && res->status == 400, "POST foreign model -> 400, got ", res ? std::to_string(res->status) : "none");
        ok(err_field(*res, "param").as_string("param") == "model", "the refusal names 'model'");
    }
    {  // POST, valid + stream:true (STORY-08): the typed SSE state machine answers (it replaced the 501).
        auto res = cli.Post("/v1/responses", "{\"model\":\"" + model_id + "\",\"input\":\"hi\",\"stream\":true}",
                            "application/json");
        ok(res && res->status == 200, "valid streaming POST -> 200 (typed SSE), got ",
           res ? std::to_string(res->status) : "none");
        ok(res && res->headers.find("content-type") != res->headers.end() &&
                   res->headers.find("content-type")->second.find("text/event-stream") != std::string::npos,
           "Content-Type text/event-stream");
        ok(res && res->body.find("\"type\":\"response.created\"") != std::string::npos &&
                   res->body.find("\"type\":\"response.completed\"") != std::string::npos,
           "created + completed frames in the body");
        ok(res && res->body.find("[DONE]") == std::string::npos, "no Chat [DONE] sentinel");
    }
    {  // Unsupported semantics are refused on their path.
        auto res = cli.Post("/v1/responses", "{\"model\":\"" + model_id + "\",\"input\":\"hi\",\"store\":true}",
                            "application/json");
        ok(res && res->status == 400 && err_field(*res, "param").as_string("param") == "store",
           "store:true -> 400 param store");
        res = cli.Post("/v1/responses", "{\"model\":\"" + model_id + "\",\"input\":\"hi\",\"tool_choice\":\"required\"}",
                       "application/json");
        ok(res && res->status == 400 && err_field(*res, "param").as_string("param") == "tool_choice",
           "tool_choice:required -> 400 param tool_choice");
        res = cli.Post("/v1/responses", "{not json", "application/json");
        ok(res && res->status == 400, "a body that is not JSON -> 400");
    }
    {  // Non-streaming generation on the route (STORY-07): the scripted answer comes back as typed Responses items.
        auto res = cli.Post("/v1/responses", "{\"model\":\"" + model_id + "\",\"input\":\"hi\"}", "application/json");
        ok(res && res->status == 200, "valid non-streaming POST -> 200, got ", res ? std::to_string(res->status) : "none",
           res ? " body " + res->body.substr(0, 200) : "");
        if (!res || res->status != 200) return;
        json::Value b = json::Value::parse(res->body);
        ok(b.find("object") && b.find("object")->as_string("object") == "response", "object=response");
        ok(b.find("id") && b.find("id")->as_string("id").rfind("resp_", 0) == 0, "response id is resp_*");
        ok(b.find("model")->as_string("model") == model_id, "model is the served id");
        ok(b.find("status")->as_string("status") == "completed", "status completed");
        ok(b.find("created_at") && b.find("created_at")->as_int("created_at") > 1700000000, "created_at is now");
        const json::Value &out = *b.find("output");
        ok(out.is_array() && out.as_array("output").size() >= 2, "reasoning + message items, got ",
           (long)out.as_array("output").size());
        ok(out.as_array("output")[0].find("type")->as_string("type") == "reasoning" &&
                   out.as_array("output")[0].find("summary")->as_array("s").empty() &&
                   out.as_array("output")[0].find("encrypted_content")->is_null(),
           "raw reasoning item: summary [], encrypted_content null");
        bool found_msg = false;
        for (const auto &it : out.as_array("output"))
            if (it.find("type")->as_string("type") == "message") {
                found_msg = true;
                ok(it.find("content")->as_array("c")[0].find("text")->as_string("text") == "Hello world",
                   "message item carries the scripted answer");
            }
        ok(found_msg, "a message item exists");
        const json::Value &u = *b.find("usage");
        ok(u.find("input_tokens")->as_int("i") > 0 && u.find("output_tokens")->as_int("o") > 0 &&
                   u.find("input_tokens_details")->find("cached_tokens") != nullptr &&
                   u.find("output_tokens_details")->find("reasoning_tokens")->as_int("rt") > 0,
           "usage from the engine: input/cached/output/reasoning tokens real");
    }
    {  // A tool call the request offered: a completed function_call item, ids separate, registry-checked.
        std::fprintf(stderr, "DBG main: before tool POST\n");
        be.set_script(tool_script);
        auto res = cli.Post("/v1/responses",
                            "{\"model\":\"" + model_id + "\",\"input\":\"run f\",\"max_output_tokens\":64,"
                            "\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\"}}}}]}",
                            "application/json");
        ok(res && res->status == 200, "tool-call POST -> 200, got ", res ? std::to_string(res->status) : "none",
           res ? " body " + res->body.substr(0, 300) : "");
        if (!res || res->status != 200) return;
        json::Value b = json::Value::parse(res->body);
        ok(b.find("status")->as_string("status") == "completed", "completed with a call");
        const json::Value *fc = nullptr;
        for (const auto &it : b.find("output")->as_array("output"))
            if (it.find("type") && it.find("type")->as_string("type") == "function_call") fc = &it;
        ok(fc != nullptr, "function_call item published");
        if (fc) {
            ok(fc->find("name")->as_string("name") == "f", "name f");
            ok(fc->find("call_id")->as_string("call_id").rfind("call_", 0) == 0, "parser call_id (call_*)");
            ok(fc->find("id")->as_string("id") != fc->find("call_id")->as_string("call_id"), "item id separate");
            ok(fc->find("arguments")->as_string("arguments") == "{\"x\":1}", "arguments typed from the schema, as JSON text");
        }
    }
    {  // Streaming on the route (STORY-08): typed Responses SSE through the live pipeline.
        be.set_script(text_script);  // the tool POST above left the call script armed
        std::string acc;
        auto res = cli.Post("/v1/responses", httplib::Headers{},
                            "{\"model\":\"" + model_id + "\",\"input\":\"hi\",\"stream\":true}", "application/json",
                            [&](const char *data, size_t len) { acc.append(data, len); return true; });
        std::fprintf(stderr, "DBG case: responses streaming Post returned\n");
        ok(res && res->status == 200, "streaming POST completed, status ", res ? std::to_string(res->status) : "none");
        ok(acc.find("response.created") != std::string::npos && acc.find("response.in_progress") != std::string::npos,
           "created + in_progress");
        ok(acc.find("response.output_text.delta") != std::string::npos, "text deltas streamed");
        ok(acc.find("response.completed") != std::string::npos, "typed terminal");
        ok(acc.find("[DONE]") == std::string::npos, "no Chat [DONE] sentinel");
        ok(acc.find("chat.completion.chunk") == std::string::npos, "no Chat chunks");
        // Extract the data frames to jsonl for the pack stream validator.
        std::ofstream out("/tmp/opencode/stream-http_generated.jsonl", std::ios::binary);
        size_t pos = 0;
        while ((pos = acc.find("data: ", pos)) != std::string::npos) {
            const size_t end = acc.find("\n", pos);
            out << acc.substr(pos + 6, end - pos - 6) << "\n";
            pos = end;
        }
        ok(out.good(), "stream dumped");
        out.close();
        std::fprintf(stderr, "DBG main: streaming case done\n");
        if (getenv("STREAM_ONLY")) { ok(true, "stream-only probe"); return; }
    }
    {  // Chat untouched: any model name is served (the model refusal above is Responses-only), the scripted
       // answer comes back as Chat content with real usage.
        be.set_script(text_script);
        auto res = cli.Post("/v1/chat/completions",
                            "{\"model\":\"anything\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                            "\"temperature\":0,\"max_tokens\":2048}",
                            "application/json");
        ok(res && res->status == 200, "Chat serves any model name (unchanged), status ",
           res ? std::to_string(res->status) : "none");
        if (!res || res->status != 200) return;
        json::Value b = json::Value::parse(res->body);
        ok(b.find("model") && b.find("model")->as_string("model") == model_id, "Chat reports the real model id");
        const json::Value &choice = b.find("choices")->as_array("choices")[0];
        ok(choice.find("message")->find("content")->as_string("content") == "Hello world",
           "the scripted answer arrives as Chat content");
        ok(choice.find("finish_reason")->as_string("finish_reason") == "stop", "finish_reason stop");
        ok(b.find("usage") && b.find("usage")->find("completion_tokens")->as_int("completion_tokens") > 0,
           "usage reports real completion tokens");
    }
    {  // Chat streams (unchanged SSE with [DONE]).
        std::string acc;
        auto res = cli.Post("/v1/chat/completions", httplib::Headers{},
                            "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,"
                            "\"temperature\":0,\"max_tokens\":2048}",
                            "application/json",
                            [&](const char *data, size_t len) { acc.append(data, len); return true; });
        ok(res, "Chat stream request completed");
        ok(acc.find("data: [DONE]") != std::string::npos, "Chat SSE closes with [DONE]");
        // Deltas stream per token (one byte here): assert the pieces arrived, not the joined string.
        ok(acc.find("\"reasoning_content\":\"t\"") != std::string::npos, "Chat SSE streamed reasoning deltas");
        ok(acc.find("\"content\":\"d\"") != std::string::npos, "Chat SSE streamed content deltas");
        ok(acc.find("\"finish_reason\":\"stop\"") != std::string::npos, "Chat SSE carries the stop finish");
    }
    {  // /health and /v1/models keep answering (the shared server surface).
        auto res = cli.Get("/health");
        ok(res && res->status == 200, "GET /health -> 200");
        res = cli.Get("/v1/models");
        ok(res && res->status == 200, "GET /v1/models -> 200");
    }
}

std::string write_temp_tokenizer(const std::string &dir) {
    const std::string path = dir + "/tokenizer.json";
    std::ofstream f(path, std::ios::binary);
    STRIX_CHECK(f.good(), "can't write '", path, "'");
    f << synthetic_tokenizer_json();
    STRIX_CHECK(f.good(), "write of '", path, "' failed");
    f.close();
    return path;
}

}  // namespace

int main(int argc, char **argv) {
    std::string tok_path = argc > 1 ? argv[1] : "";
    char tmpdir[] = "/tmp/opencode/check_responses_http.XXXXXX";
    std::string keep_dir;
    try {
        if (tok_path.empty()) {
            const char *d = mkdtemp(tmpdir);
            STRIX_CHECK(d != nullptr, "mkdtemp failed");
            keep_dir = d;
            tok_path = write_temp_tokenizer(d);
        }
        Tokenizer tok(tok_path);
        // The scripted answer: reasoning, the think close (its own token), " Hello world", the end of turn.
        std::vector<int32_t> script;
        // The close tag counts only right after a newline (Config::literal_tags), so the reasoning text ends with one.
        for (char c : std::string_view("thinking\n")) script.push_back(tok.byte_token((uint8_t)c));
        script.push_back(tok.id_of(tagc("think").c_str()));  // the think close is an added token of the model
        for (char c : std::string_view(" Hello world")) script.push_back(tok.byte_token((uint8_t)c));
        const int32_t im_end = tok.id_of(sp("im_end").c_str());
        script.push_back(im_end);
        ScriptedLmBackend be(tok.size(), std::move(script), im_end);
        Engine::Options options;
        options.think_nudge = false;  // no real repetitions to watch in a scripted answer
        Engine engine(be, tok, options);

        ServerConfig cfg;
        cfg.port = 0;  // any free port
        cfg.model_id = "qwen3.8-flash-next";
        HttpServer server(engine, tok, cfg);
        const int port = server.bind();
        std::thread thr([&] { server.serve(); });

        if (getenv("SERVE_ONLY")) {  // keep the scripted server up for external probes (python/pytest side)
            std::printf("SERVING port=%d\n", port);
            std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::minutes(3));
            server.stop();
            thr.join();
            return 0;
        }
        httplib::Client cli("127.0.0.1", port);
        cli.set_read_timeout(10, 0);
        check_routes(cli, cfg.model_id, tok, be);

        server.stop();
        thr.join();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    if (!keep_dir.empty()) std::remove((keep_dir + "/tokenizer.json").c_str());
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
