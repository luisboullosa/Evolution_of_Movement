#include <filesystem>
#include <iostream>
#include <regex>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    fs::path build_dir = fs::current_path();
    fs::path project_root = build_dir.parent_path(); // tests run from build/

    std::vector<std::string> exts = {".tsv", ".csv", ".png"};
    std::regex sim_re("^(sim|scan)\\\d{4}$", std::regex::ECMAScript | std::regex::icase);

    std::vector<fs::path> offenders;

    // 1) Check top-level files in project root (non-recursive)
    for (auto &e : fs::directory_iterator(project_root)) {
        if (!e.is_regular_file()) continue;
        fs::path p = e.path();
        std::string ext = p.extension().string();
        for (auto &ex : exts) {
            if (ext == ex) {
                offenders.push_back(p);
                break;
            }
        }
    }

    // 2) Check files under webviewer/ recursively, but allow files inside simNNNN or scanNNNN directories
    fs::path webviewer = project_root / "webviewer";
    if (fs::exists(webviewer) && fs::is_directory(webviewer)) {
        for (auto &entry : fs::recursive_directory_iterator(webviewer)) {
            if (!entry.is_regular_file()) continue;
            fs::path p = entry.path();
            std::string ext = p.extension().string();
            bool match_ext = false;
            for (auto &ex : exts) if (ext == ex) { match_ext = true; break; }
            if (!match_ext) continue;

            // climb ancestors from the file's parent up to webviewer; if any ancestor matches simNNNN/scanNNNN, allow
            bool allowed = false;
            fs::path cur = p.parent_path();
            while (cur != webviewer && cur.has_relative_path()) {
                std::string name = cur.filename().string();
                if (std::regex_match(name, sim_re)) { allowed = true; break; }
                if (cur == cur.parent_path()) break; // root guard
                cur = cur.parent_path();
            }
            if (!allowed) offenders.push_back(p);
        }
    }

    if (!offenders.empty()) {
        std::cerr << "Found artifact files outside simNNNN/scanNNNN folders:\n";
        for (auto &p : offenders) std::cerr << "  " << p.string() << "\n";
        return 1;
    }

    return 0;
}
